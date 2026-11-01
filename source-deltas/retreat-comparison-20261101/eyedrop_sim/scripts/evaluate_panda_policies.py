"""同じ実験内で固定ルール・追加学習前後を全動作比較する。"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from panda_rpc import PandaBrowser
from resumable_job import atomic_json
from eyedrop.panda_world import PandaWorld,REST_GAP_M,BOTTLE_DEPTH_M,to_physics,PHYSICS_HZ,ACTION_HZ
from eyedrop.smolvla_observation import history_from_normalization
from pressure_interval_audit import audit_pressure_interval,audit_guarded_interval
from cartesian_action import action_dimension,CARTESIAN_DIM
from endpoint_policy import MODE as ENDPOINT_MODE

PRESS_TOLERANCE_M=.0001
NORMAL_FINGER_RELEASE_SPEED_M_S=.02
GUARDED_FINGER_RELEASE_SPEED_M_S=.05
GUARD_OPEN_THRESHOLD=.95
GUARD_FINGER_PRESS_SPEED_M_S=.0005
GUARD_JOINT_VELOCITY_CAP=.1


def validate_guard_configuration(document, enabled, interval_audit, action_steps, face_sha):
    from eyedrop.guarded_action_contract import guarded_action_contract
    from eyedrop.smolvla_observation import command_observation_contract
    if not enabled and document.get('guarded_action_contract') is not None:
        raise ValueError('ガード付き教材には実行時ガードが必要です')
    if enabled and (not interval_audit or action_steps!=1 or not face_sha
                    or document.get('guarded_action_contract')!=guarded_action_contract()
                    or document.get('observation_history_contract')!=command_observation_contract()):
        raise ValueError('画像ガードは固定顔モデルSHA、全区間監査と実行指令履歴の契約が必要です')


def guarded_policy_interval(world, request, observe, guard, history, step, browser):
    from eyedrop.guarded_execution import step_guarded_interval
    from eyedrop.drop_kinematics import domain_summary
    start=world.time
    outcome=step_guarded_interval(world,request,observe,guard,
                                  normal_release_speed=NORMAL_FINGER_RELEASE_SPEED_M_S,
                                  guarded_release_speed=GUARDED_FINGER_RELEASE_SPEED_M_S)
    if not outcome['complete']:
        return outcome,None
    scored=browser.call('pressureSamples',samples=outcome['pressure_samples'])
    audit=audit_guarded_interval(scored,outcome['substep_actions'],outcome['motion_samples'],
                                 start_time=start,end_time=start+1/ACTION_HZ,
                                 press_threshold_m=BOTTLE_DEPTH_M-PRESS_TOLERANCE_M,
                                 physics_hz=PHYSICS_HZ,open_threshold=GUARD_OPEN_THRESHOLD)
    domains=[dict(row,start_time=row['start_time']-start,time=row['time']-start)
             for row in outcome['drop_domain_samples']]
    audit['drop_domain']=domain_summary(domains,expected_intervals=PHYSICS_HZ//ACTION_HZ)
    audit['passed']=audit['passed'] and audit['drop_domain']['valid']
    outcome['scored_pressure_samples']=scored
    history.append(step+1,world.own_state(),outcome['last_applied_gap_m'])
    return outcome,audit


def emergency_guard_release(world, guard):
    from eyedrop.guarded_execution import step_guarded_interval
    def unavailable():
        raise RuntimeError('画像または方策の処理が停止しました')
    request=[*world.own_state()[:7],guard.rest_gap]
    return step_guarded_interval(world,request,unavailable,guard,
                                 normal_release_speed=NORMAL_FINGER_RELEASE_SPEED_M_S,
                                 guarded_release_speed=GUARDED_FINGER_RELEASE_SPEED_M_S)


def validate_guarded_context(context):
    from eyedrop.guarded_execution import ImagePressGuard
    from guarded_rgb_observer import RGBGuardObserver
    if not isinstance(context,tuple) or len(context)!=3:
        raise ValueError('実行ガードと画像観測の組が必要です')
    guard,observer,world=context
    if (not isinstance(guard,ImagePressGuard) or guard.latched or not isinstance(observer,RGBGuardObserver)
            or observer.world is not world or not observer.records):
        raise ValueError('対象worldの初期観測と未作動ガードが必要です')


class PolicyWorker:
    def __init__(self,root,normalization,adapter,log,native_full=False,action_steps=1,noise_mode='fresh',gripper_tree=None,prediction_chunk_size=None,endpoint_regression=False,relative_initial_sha256=None,expected_normalization_sha256=None,guarded_context=None):
        if prediction_chunk_size is not None and (type(prediction_chunk_size) is not int or prediction_chunk_size not in (1,50)):
            raise ValueError('予測動作数は1または50です')
        if relative_initial_sha256 and (not native_full or not adapter or prediction_chunk_size!=1 or action_steps!=1 or gripper_tree or endpoint_regression):
            raise ValueError('相対指令の初期比較は通常重み・1予測1実行を明示してください')
        if guarded_context is not None:
            validate_guarded_context(guarded_context)
            if not expected_normalization_sha256:raise ValueError('ガード付き推論には固定正規化SHAが必要です')
        self.log=open(log,'w')
        command=[str(root/'.venv-lerobot-smolvla/bin/python'),str(root/'scripts/panda_policy_worker.py'),str(normalization),'--action-steps',str(action_steps),'--noise-mode',noise_mode]
        if adapter:command+=['--full-policy' if native_full else '--adapter',str(adapter)]
        if gripper_tree:command+=['--gripper-tree',str(gripper_tree)]
        if prediction_chunk_size is not None:command+=['--prediction-chunk-size',str(prediction_chunk_size)]
        if endpoint_regression:command+=['--endpoint-regression']
        if relative_initial_sha256:command+=['--relative-initial-sha256',relative_initial_sha256]
        if expected_normalization_sha256:command+=['--expected-normalization-sha256',expected_normalization_sha256]
        if guarded_context is not None:command+=['--guarded-requests-only']
        environment={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,env=environment)
        response=self.process.stdout.readline()
        if not response or not json.loads(response).get('ready'):raise RuntimeError('モデルの起動失敗')
        if guarded_context is not None and not json.loads(response).get('guarded_requests_only'):
            self.close()
            raise RuntimeError('推論workerがガード前要求専用として起動していません')
        if expected_normalization_sha256 and json.loads(response).get('normalization_sha256')!=expected_normalization_sha256:
            self.close()
            raise RuntimeError('要求した正規化SHAとworkerの読込値が一致しません')
        if relative_initial_sha256 and json.loads(response).get('relative_initial_sha256')!=relative_initial_sha256:
            self.close()
            raise RuntimeError('要求した初期重みの対応付けが確認できません')
        if prediction_chunk_size is not None and json.loads(response).get('prediction_chunk_size')!=prediction_chunk_size:
            self.close()
            raise RuntimeError('要求した予測動作数とモデルが一致しません')
        if endpoint_regression and (json.loads(response).get('policy_mode')!=ENDPOINT_MODE or json.loads(response).get('num_steps')!=1 or json.loads(response).get('noise_mode')!='zero'):
            self.close()
            raise RuntimeError('要求した1段回帰の推論方式が一致しません')

    def call(self,**request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        response=self.process.stdout.readline()
        if not response:raise RuntimeError('推論ワーカーが終了しました')
        value=json.loads(response)
        if 'error' in value:raise RuntimeError(value['error'])
        return value

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"operation":"close"}\n');self.process.stdin.flush();self.process.stdin.close()
            self.process.wait(timeout=30)
        self.log.close()


def resolve_evaluation_normalization(data, adapter, document, normalization_path, training_result, native_full):
    requires_binding = document.get('temporal_rgb_contract') is not None or document.get('guarded_action_contract') is not None
    if requires_binding:
        if training_result is None:
            raise ValueError('時間履歴またはガード付き評価には学習result.jsonが必要です')
        from eyedrop.smolvla_dataset_contract import verify_temporal_training_contract
        gate = json.loads((data/'result.json').read_bytes())
        expected = verify_temporal_training_contract(training_result, gate, data/'result.json', document, normalization_path)
        weights_name = 'model.safetensors' if native_full else 'adapter_model.safetensors'
        weights_key = 'native_weights_sha256' if native_full else 'adapter_sha256'
        if training_result.get(weights_key) != hashlib.sha256((adapter/weights_name).read_bytes()).hexdigest():
            raise ValueError('学習記録と評価対象の重みSHAが一致しません')
        return expected
    if training_result is not None and training_result.get('normalization_sha256') != hashlib.sha256(normalization_path.read_bytes()).hexdigest():
        raise ValueError('学習記録と評価normalizationのSHAが一致しません')
    return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw');parser.add_argument('dataset')
    parser.add_argument('adapter');parser.add_argument('output');parser.add_argument('--native-full',action='store_true')
    parser.add_argument('--action-steps',type=int,choices=(1,2,5),default=1)
    parser.add_argument('--gripper-tree')
    parser.add_argument('--interval-audit',action='store_true')
    parser.add_argument('--cartesian-actions',action='store_true')
    parser.add_argument('--match-prediction-chunks',action='store_true')
    parser.add_argument('--endpoint-regression',action='store_true')
    parser.add_argument('--training-result')
    parser.add_argument('--guard-face-model');parser.add_argument('--guard-face-sha256')
    parser.add_argument('--guard-press-speed',type=float,default=GUARD_FINGER_PRESS_SPEED_M_S)
    parser.add_argument('--policy-mode',choices=('fixed-rule','pretrained','finetuned'))
    parser.add_argument('--person',default='original')
    parser.add_argument('--noise-mode',choices=('fresh','fixed','zero'),default='fresh');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    raw=Path(args.raw).resolve();data=Path(args.dataset).resolve();adapter=Path(args.adapter).resolve()
    normalization_path=data/'normalization.json'
    normalization_document=json.loads(normalization_path.read_bytes())
    image_guard=bool(args.guard_face_model)
    if not 0<args.guard_press_speed<=GUARD_JOINT_VELOCITY_CAP:
        raise ValueError('押圧速度が可動範囲外です')
    validate_guard_configuration(normalization_document,image_guard,args.interval_audit,args.action_steps,args.guard_face_sha256)
    if image_guard:
        from eyedrop.face_features import FaceFeatures
        guard_face=FaceFeatures(args.guard_face_model,'mps',expected_sha256=args.guard_face_sha256)
    elif args.guard_face_sha256:
        raise ValueError('顔モデルを指定せずにSHAだけを指定できません')
    normalization_sha256=hashlib.sha256(normalization_path.read_bytes()).hexdigest()
    training_result_path=Path(args.training_result).resolve() if args.training_result else None
    training_result=None
    if training_result_path:
        training_result=json.loads(training_result_path.read_bytes())
    expected_normalization_sha256=resolve_evaluation_normalization(
        data,adapter,normalization_document,normalization_path,training_result,args.native_full)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((raw/'protocol.json').read_text())
    expected_cartesian=action_dimension(json.loads((data/'normalization.json').read_text()))==CARTESIAN_DIM
    if args.cartesian_actions!=expected_cartesian:raise ValueError('動作空間の明示指定と統計が異なります')
    if args.cartesian_actions and not args.interval_audit:raise ValueError('Cartesian比較は全物理区間の押圧監査が必要です')
    if args.match_prediction_chunks and not args.native_full:raise ValueError('今回の予測数一致比較は通常モデル専用です')
    if args.endpoint_regression!=(args.noise_mode=='zero') or (args.endpoint_regression and (not args.native_full or not args.cartesian_actions or not args.match_prediction_chunks or args.action_steps!=1)):
        raise ValueError('1段回帰はCartesian通常モデル/先頭1動作/chunk一致/zeroを明示してください')
    matching_chunk=json.loads((adapter/'config.json').read_text())['chunk_size'] if args.match_prediction_chunks else None
    if args.gripper_tree and not args.native_full:raise ValueError('分類木は対応する通常モデルと組み合わせてください')
    rest_gap=protocol.get('rest_gap_m',REST_GAP_M)
    weights_file=adapter/('model.safetensors' if args.native_full else 'adapter_model.safetensors')
    with weights_file.open('rb') as stream:weights_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    initial=json.loads((raw/'train-02/result.json').read_text())['initial_joints']
    fixed=json.loads((raw/'train-02/frames.json').read_text())
    if not fixed:raise ValueError('固定教師の動作が空です')
    modes=(args.policy_mode,) if args.policy_mode else ('fixed-rule','pretrained','finetuned')
    atomic_json(output/'protocol.json',{'cases':protocol['validation'],'policies':modes,'rendered_person':args.person,
                'image_guard':{'face_model_sha256':guard_face.sha256,'open_threshold':GUARD_OPEN_THRESHOLD,
                               'finger_press_speed_m_s':args.guard_press_speed,'finger_release_speed_m_s':NORMAL_FINGER_RELEASE_SPEED_M_S,
                               'guard_release_speed_m_s':GUARDED_FINGER_RELEASE_SPEED_M_S,
                               'bottle_input':'measured','drop_model':'moving-nozzle'} if image_guard else None,
                'action_space':'nozzle XYZ and gap to robot-only IK' if args.cartesian_actions else 'joint angles and gap',
                'prediction_chunks':{'fixed-rule':None,'pretrained':matching_chunk if matching_chunk is not None else 50,'finetuned':json.loads((adapter/'config.json').read_text())['chunk_size']} if args.native_full else None,
                'matching_prediction_chunks':args.match_prediction_chunks,
                'policy_algorithm':ENDPOINT_MODE if args.endpoint_regression else 'smolvla-flow-matching-v1',
                'pretrained_comparator':'同じ1段/zeroへ適応した追加学習前' if args.endpoint_regression else '標準flowの追加学習前',
                'task_policy_sha256':hashlib.sha256((adapter/'task-policy.json').read_bytes()).hexdigest() if args.endpoint_regression else None,
                'action_steps':args.action_steps,'queue':'1は従来の先頭再推論。2/5はSmolVLAPolicy.select_actionの公式キュー。乱数は物理時点ごとに進める',
                'noise_mode':args.noise_mode,'noise_seed':'714+case_index。fixedは試行の最初のnoiseを再利用し、都合の良いseedを選別しない',
                'interval_pressure_audit_hz':PHYSICS_HZ if args.interval_audit else None,
                'press_threshold_m':BOTTLE_DEPTH_M-PRESS_TOLERANCE_M,'safety':'0.1mm以上の容器壁押圧指令と実測押圧を別計数。接触余裕は教師と同じ',
                'checkpoint_format':'native-full-float32' if args.native_full else 'peft-lora','weights_sha256':weights_hash,
                'adapter_sha256':None if args.native_full else weights_hash,
                'gripper_tree':{'path':str(Path(args.gripper_tree).resolve()),'sha256':hashlib.sha256(Path(args.gripper_tree).read_bytes()).hexdigest(),
                                'scope':'finetunedだけをSmolVLA関節出力と学習分類木の指幅へ変更。診断目的・未採用'} if args.gripper_tree else None,
                'normalization_sha256':hashlib.sha256((data/'normalization.json').read_bytes()).hexdigest(),
                'training_result_sha256':hashlib.sha256(training_result_path.read_bytes()).hexdigest() if training_result_path else None,
                'raw_protocol_sha256':hashlib.sha256((raw/'protocol.json').read_bytes()).hexdigest(),
                'source_sha256':{relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in
                    (('scripts/evaluate_panda_policies.py','scripts/panda_policy_worker.py','scripts/smolvla_runtime.py',
                     'scripts/pressure_interval_audit.py','scripts/panda_browser.cjs',
                     'scripts/cartesian_action.py','scripts/action_encoding.py',
                     'scripts/endpoint_policy.py',
                     'scripts/tree_gripper.py','scripts/direct_gripper.py',
                     'src/eyedrop/panda_world.py','src/eyedrop/smolvla_observation.py','preview/panda-runtime.js','preview/scene.js')+
                     (('scripts/guarded_rgb_observer.py','src/eyedrop/guarded_execution.py','src/eyedrop/eye_image_change.py',
                       'src/eyedrop/guarded_action_contract.py','preview/panda-hybrid-rgb.js') if image_guard else ()))},
                'fixed_rule':f'学習用のnominal train-02の{len(fixed)}関節・指幅指令を固定再生。検証の正解は制御へ渡さない'})
    browser=PandaBrowser(output/'browser.log',person=args.person,hybrid_rgb=image_guard);results=[];started=time.monotonic()
    def start_worker(mode,guarded_context=None,case_id=None):
        log_name=f'{mode}-{case_id}.log' if case_id is not None else f'{mode}.log'
        return PolicyWorker(root,normalization_path,adapter if mode=='finetuned' else None,output/log_name,
                            native_full=args.native_full and mode=='finetuned',action_steps=args.action_steps,
                            noise_mode=args.noise_mode,gripper_tree=args.gripper_tree if mode=='finetuned' else None,
                            prediction_chunk_size=matching_chunk,endpoint_regression=args.endpoint_regression,
                            expected_normalization_sha256=expected_normalization_sha256,guarded_context=guarded_context)
    try:
        setup=PandaWorld(output,rest_gap=rest_gap);browser.call('setup',shapes=setup.drawing());setup.close()
        for mode in modes:
            worker=None
            try:
                if mode!='fixed-rule' and not image_guard:worker=start_worker(mode)
                for case_index,case in enumerate(protocol['validation']):
                    directory=output/f'{mode}-{case["id"]}';directory.mkdir()
                    browser.call('reset',depth=case['depth_m'],phase=case['blink_phase_s'])
                    geometry=browser.call('geometry')
                    if image_guard:
                        from eyedrop.drop_kinematics import MOVING_NOZZLE
                        world=PandaWorld(directory,rest_gap=rest_gap,bottle_input='measured',finger_press_speed_m_s=args.guard_press_speed,
                                         finger_release_speed_m_s=NORMAL_FINGER_RELEASE_SPEED_M_S,joint_velocity_cap=GUARD_JOINT_VELOCITY_CAP,drop_model=MOVING_NOZZLE)
                    else:world=PandaWorld(directory,rest_gap=rest_gap)
                    try:
                        world.reset(initial);initial_nozzle=np.array(world.nozzle())
                        if image_guard:world.add_affine_geometry(geometry,browser.call('blinkGeometry'))
                        else:world.add_geometry(geometry)
                        history=history_from_normalization(json.loads((data/'normalization.json').read_bytes()),rest_gap)
                        history.reset(world.own_state(),rest_gap)
                        if image_guard:
                            from eyedrop.guarded_execution import ImagePressGuard
                            from guarded_rgb_observer import RGBGuardObserver
                            guard=ImagePressGuard(rest_gap)
                            observer=RGBGuardObserver(browser,world,guard_face,directory/'guard-images')
                            observer()
                            if mode!='fixed-rule':worker=start_worker(mode,(guard,observer,world),case['id'])
                        if worker:worker.call(operation='reset',seed=714+case_index)
                        rows=[];hits=[];contacts=[];closed_command=0;closed_measured=0;error=None;interval_failures=0
                        for step in range(len(fixed)):
                            state=history.observation()
                            inference_started=time.monotonic()
                            action=None;outcome=None;frame=None
                            try:
                                frame=browser.call('render',state={'time':world.time,'links':world.links()},capture=True)
                                for side in ('left','right'):(directory/f'{side}-{step:03}.png').write_bytes(base64.b64decode(frame[side]))
                                action=worker.call(operation='act',state=state,left=frame['left'],right=frame['right'])['action'] if worker else fixed[step]['action']
                                predicted_cartesian=None
                                if args.cartesian_actions and worker:
                                    predicted_cartesian=action
                                    action=[*world.ik(predicted_cartesian[:3]),predicted_cartesian[-1]]
                                inference_s=time.monotonic()-inference_started
                                if not image_guard and frame['open']<.9:
                                    closed_command+=int(action[7]<BOTTLE_DEPTH_M-PRESS_TOLERANCE_M)
                                    closed_measured+=int(world.own_state()[7]<BOTTLE_DEPTH_M-PRESS_TOLERANCE_M)
                                interval_start=world.time
                                interval_result=None;pressure_samples=None
                                if image_guard:
                                    outcome,interval_result=guarded_policy_interval(world,action,observer,guard,history,step,browser)
                                    if not outcome['complete']:
                                        contacts+=outcome['contacts']
                                        raise RuntimeError(f'画像ガード実行が中断しました: {outcome["observation_error"]}; 接触: {outcome["contacts"]}')
                                    pressure_samples=outcome['scored_pressure_samples']
                                    closed_command+=interval_result['closed_command_samples']
                                    closed_measured+=interval_result['closed_measured_samples']
                                    interval_failures+=int(not interval_result['passed'])
                                else:
                                    outcome=world.step(action,audit_pressure=True) if args.interval_audit else world.step(action)
                                if args.interval_audit and not image_guard:
                                    pressure_samples=browser.call('pressureSamples',samples=outcome['pressure_samples'])
                                    if outcome['contacts']:
                                        interval_result={'complete':False,'passed':False,'reason':'safety-stop'}
                                    else:
                                        interval_result=audit_pressure_interval(pressure_samples,start_time=interval_start,
                                            end_time=interval_start+1/ACTION_HZ,command_gap_m=action[7],
                                            press_threshold_m=BOTTLE_DEPTH_M-PRESS_TOLERANCE_M,physics_hz=PHYSICS_HZ)
                                        interval_result['complete']=True
                                    interval_failures+=int(not interval_result['passed'])
                                if not image_guard:history.append(step+1,world.own_state(),action[7])
                                hits+=browser.call('scoreSegments',segments=outcome.get('segments',[]))
                                contacts+=outcome['contacts']
                                row={'step':step,'state':state,'action':action,'open_audit':frame['open'],
                                     'inference_seconds':inference_s,'contacts':outcome['contacts']}
                                if args.interval_audit:row.update(pressure_interval_audit=interval_result,pressure_samples=pressure_samples)
                                if image_guard:row.update(executed_substep_actions=outcome['substep_actions'],
                                                          executed_end_gap_m=outcome['last_applied_gap_m'],
                                                          observed_changes=outcome['observed_changes'],
                                                          drop_domain_samples=outcome['drop_domain_samples'],
                                                          motion_samples=outcome['motion_samples'])
                                if predicted_cartesian is not None:row['predicted_cartesian_action']=predicted_cartesian
                                rows.append(row)
                                if contacts:break
                            except (OSError,ValueError,RuntimeError) as failure:
                                error=repr(failure)
                                if image_guard and outcome is None:
                                    outcome=emergency_guard_release(world,guard)
                                failed={'step':step,'state':state,'action':action,'open_audit':frame.get('open') if frame else None,
                                        'failed':True,'error':error,'own_after_failure':world.own_state()}
                                if image_guard and outcome is not None:failed['guarded_outcome']=outcome
                                rows.append(failed)
                                break
                        if image_guard:retreat=initial_nozzle
                        else:
                            nominal=json.loads((raw/'train-02/scoring-geometry.json').read_text())['eye']
                            retreat=np.array(to_physics(nominal))+np.array([0,0,protocol['initial_nozzle_height_m']])
                        retreat_error=float(np.linalg.norm(np.array(world.nozzle())-retreat))
                        gap=world.own_state()[7]
                        hit_open_threshold=GUARD_OPEN_THRESHOLD if image_guard else .9
                        accepted=len(rows)==len(fixed) and error is None and not contacts and not closed_command and not closed_measured and not interval_failures and len(world.drops)==1 and len(hits)==1 and hits[0]['result']=='target' and hits[0]['open']>=hit_open_threshold and abs(gap-rest_gap)<.0001 and retreat_error<.001
                        result={'mode':mode,'case':case,'accepted':accepted,'frames':len(rows),'error':error,
                                'contacts':contacts,'closed_press_commands':closed_command,'closed_measured_compression':closed_measured,
                                'drops':len(world.drops),'hits':hits,'released_gap':gap,'retreat_error_m':retreat_error}
                        if args.interval_audit:result.update(interval_pressure_audit_hz=PHYSICS_HZ,failed_pressure_intervals=interval_failures)
                        if image_guard:atomic_json(directory/'guard-observations.json',observer.records)
                        atomic_json(directory/'trace.json',rows);atomic_json(directory/'result.json',result)
                        results.append(result);atomic_json(output/'result.json',{'rows':results,'seconds':time.monotonic()-started,'evaluation_completed':len(results)==len(modes)*len(protocol['validation']),'adopted':False})
                        browser.call('screenshot',path=str(directory/'final.png'))
                        print(json.dumps(result),flush=True)
                    finally:
                        if image_guard and worker:
                            worker.close();worker=None
                        world.close()
            finally:
                if worker:worker.close()
    finally:browser.close()


if __name__=='__main__':main()
