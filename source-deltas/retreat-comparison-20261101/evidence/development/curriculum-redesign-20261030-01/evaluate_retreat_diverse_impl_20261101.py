"""未採用の退避候補を開発条件で診断する。"""
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from eye_crop_override import install_eye_crop_override
from eyedrop.eye_ready_head import ready_from_image
from evaluate_panda_policies import PolicyWorker
from hybrid_action import combine_action
from retreat_home_gate_20261101 import RetreatHomeGate
from learned_retreat_action_20261101 import retreat_action
from retreat_path_context_20261101 import OnlinePathContext
from retreat_diverse_runtime_20261101 import encode_runtime_state
from retreat_context_features_20261101 import model_state
from install_blink_head_mix import install
from late_fine_source import select_source
from dual_host_stabilize import should_stabilize
from image_motion_validity import ObservedReferencePosition
from learned_rollout_fluid import audit_fluid
from panda_rpc import PandaBrowser
from relative_joint_action import file_hash
from resumable_job import atomic_json
from eyedrop.model_integrity import restore_sklearn_model, verified_model_bytes
from eyedrop.panda_fixed_control import FixedEyeDrop
from eyedrop.panda_rgb import PandaRgb, predict_rgb_reference
from eyedrop.face_features import FaceFeatures
from eyedrop.panda_world import PandaWorld, to_physics, SUBSTEPS, synchronized_joint_target
from eyedrop.smolvla_observation import history_from_normalization
from eyedrop.drop_kinematics import MOVING_NOZZLE, domain_summary

BASE = Path('outputs/diagnostics/curriculum-redesign-20261030-01')
DATA = Path('outputs/learning/panda-synchronized-data-01')
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=('control','candidate'),required=True)
parser.add_argument('--person',choices=('dev-d','dev-e'),required=True)
parser.add_argument('--position-model',choices=('control','candidate'),required=True)
parser.add_argument('--diagnostic',action='store_true')
parser.add_argument('--retreat-mode',choices=('baseline','candidate'),required=True)
parser.add_argument('--retreat-case',choices=('e-train04','d-val00'),required=True)
parser.add_argument('--context-mode',choices=('coarse',),required=True)
parser.add_argument('--alignment-profile',choices=('original','exploratory'),default='original')
ARGS=parser.parse_args()
if not ARGS.diagnostic or ARGS.alignment_profile != 'exploratory':
    parser.error('この比較は2.5mm条件の診断のみです')
MODE=ARGS.mode
PERSON=ARGS.person
POSITION_MODEL=ARGS.position_model
TRAIN = BASE/('dual-host-perturbed-weighted-comparison-candidate-20261031-01' if MODE == 'control' else 'dual-host-e-recovery-half-20261101-01')
CHECKPOINT = TRAIN/'policy-final'
OUT = BASE/('dual-host-eye-crop-added-20261031-'+POSITION_MODEL+'-'+PERSON+'-01')
HOLD_TRAIN = BASE/'dual-host-person-batch-hold-20261031-01'
HOLD_MODEL = HOLD_TRAIN/'policy-final'
HOLD_SHA = 'de87bfa660eaa35e5a4371ebcc04dea18550818a8d39125af9756cba8b4ca4ee'
RUNTIME = Path('outputs/runtime/guarded-eval-person-20261029-01')
NORM_SHA = '80f84cecc060a4c186c5b5835bf453282f01c166f7226abfb75cd7f4bfc2dc28'
CASE = 'original-train-04' if ARGS.retreat_case == 'e-train04' else 'original-val-00'
SEED = 714
ENVIRONMENT_PERSON = {'dev-d':'vroid-beta-d','dev-e':'vroid-beta-e'}
MAX_STEPS = 1200
FINE_SWITCH_DISTANCE_M = .015
TARGET_HEIGHT_M = .05
ALIGN_M = .0025

def main():
    gate = json.loads((BASE/'dual-host-person-batch-hold-static-20261031.json').read_text())
    if not gate['completed'] or not ARGS.diagnostic:
        raise ValueError('固定画像比較が未完了、または診断指定がありません')
    training = json.loads((TRAIN/'result.json').read_text())
    normpath = CHECKPOINT/'normalization.json'
    if (not training['completed'] or (MODE == 'control' and training.get('updates') != 240) or (MODE == 'candidate' and (training.get('alpha') != 0.5 or not training.get('reload_exact'))) or file_hash(CHECKPOINT/'model.safetensors') != training['model_sha256'] or file_hash(normpath) != NORM_SHA):
        raise ValueError('モデルまたは正規化の同一性を確認できません')
    if ARGS.retreat_mode != 'candidate':
        raise ValueError('経路比較は追加学習モデルだけを使います')
    retreat_train = BASE / 'dual-host-retreat-diverse-coarse-20261101-01'
    retreat_checkpoint = retreat_train/'policy-final'
    retreat_training = json.loads((retreat_train/'result.json').read_text())
    if (not retreat_training['completed'] or file_hash(retreat_checkpoint/'model.safetensors') != retreat_training['model_sha256']
            or file_hash(retreat_checkpoint/'normalization.json') != retreat_training['normalization_sha256']
            or retreat_training.get('mode') != ARGS.context_mode or retreat_training.get('smoke')
            or (ARGS.retreat_mode == 'candidate' and (retreat_training.get('updates') != 320 or retreat_training.get('reload_max_abs_difference',1) > 1e-5))):
        raise ValueError('退避モデルまたは正規化を確認できません')
    retreat_norm = json.loads((retreat_checkpoint/'normalization.json').read_text())
    if retreat_norm.get('retreat_context_contract', {}).get('mode') != ARGS.context_mode:
        raise ValueError('経路入力の契約が一致しません')
    retreat_means = retreat_norm['features']['observation.state']['mean'][16:30]
    if len(retreat_means) != 14:
        raise ValueError('経路入力の統計が不足しています')
    hold_training=json.loads((HOLD_TRAIN/'result.json').read_text())
    if not hold_training['completed'] or hold_training['updates'] != 480 or hold_training['model_sha256'] != HOLD_SHA or file_hash(HOLD_MODEL/'model.safetensors') != HOLD_SHA or file_hash(HOLD_MODEL/'normalization.json') != NORM_SHA:
        raise ValueError('保持用モデルまたは正規化の同一性を確認できません')
    norm=json.loads(normpath.read_text())
    if norm.get('joint_action_encoding') != 'joint-relative-observation-v1':
        raise ValueError('腕の指令形式が不一致です')
    protocol=json.loads((DATA/'protocol.json').read_text())
    case=next(row for row in protocol['train' if ARGS.retreat_case == 'e-train04' else 'validation'] if row['id']==CASE)
    case={**case,'person':ENVIRONMENT_PERSON[PERSON]}
    source=Path(json.loads((DATA/CASE/'result.json').read_text())['original_source'])
    meta=json.loads((source.parent/'protocol.json').read_text())
    if meta['oracle_diagnostic'] or meta['drop_model'] != MOVING_NOZZLE:
        raise ValueError('画像推論・液滴モデルの契約が不一致です')
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT/'protocol.json',dict(case=CASE,environment_person=case['person'],seed=SEED,max_steps=MAX_STEPS,
        task='fixed RGB coarse approach; learned RGB final approach and learned near-target hold; fixed press/release and retreat with delayed full finger opening',
        fine_switch_distance_m=FINE_SWITCH_DISTANCE_M,
        reference_source=str(source),model_sha256=training['model_sha256'],normalization_sha256=NORM_SHA,
        inference_inputs=['left_rgb','right_rgb','robot_state'],alignment_internal_m=ALIGN_M,
        ground_truth='scoring only',initial_joints=meta['initial_joints'],training_updates=0,hold_model_sha256=HOLD_SHA,
        retreat_mode=ARGS.retreat_mode,retreat_context_mode=ARGS.context_mode,retreat_model_sha256=retreat_training['model_sha256'],
        retreat_normalization_sha256=retreat_training['normalization_sha256'],
        retreat_inputs=['left_rgb','right_rgb','robot_state'],retreat_gripper='fixed',retreat_home='own_initial_joints',
        static_gate_passed=None, diagnostic=ARGS.diagnostic,
        alignment_profile='exploratory_2_5mm', readiness='mixed-two-heads', position_diagnostic='four-people-'+POSITION_MODEL, prior_stricter_gate_passed=False))
    observer=PandaRgb(Path('outputs/learning'),position_mode=meta['position_mode'],
        all_position_head=meta['all_position_head'],diagnostic_ready_head=meta['diagnostic_ready_head'])
    mixture=install(observer,BASE/'blink-head-mix-handoff-20261031-01',BASE/'dual-host-blink-head-mix-extra-20261031-01.json')
    atomic_json(OUT/'readiness-mixture.json',mixture)
    folder=BASE/'four-people-position-fit-20261031-01'
    card=json.loads((folder/'result.json').read_text())
    candidate_sha='a24b3ed33f965935081b94533a1d1a30e08eb1ce47e37b9bfd8e6ae6c5cf68c1'
    if not card['completed'] or card['encoder_sha256']!=candidate_sha or POSITION_MODEL not in card['model_sha256']:
        raise ValueError('眼の位置認識器の重みが不一致です')
    position_face=FaceFeatures(BASE/'seven-stereo-face-candidate-20261031-01','mps',expected_sha256=candidate_sha)
    install_eye_crop_override(observer, position_face, ready_from_image)
    regressor=restore_sklearn_model(verified_model_bytes(folder/(POSITION_MODEL+'.pkl'),card['model_sha256'][POSITION_MODEL]))
    atomic_json(OUT/'position-model.json',{'name':POSITION_MODEL,'model_sha256':card['model_sha256'][POSITION_MODEL],
                                          'encoder_sha256':candidate_sha,'comparison_result_sha256':file_hash(folder/'result.json')})
    guard=ObservedReferencePosition(lambda images:predict_rgb_reference(position_face,regressor,images),
                                    max_age_seconds=meta['diagnostic_reference_horizon_s'])
    control=FixedEyeDrop(rest_gap=meta['rest_gap'],press_steps_limit=meta['press_steps_limit'],
        standby_gap=meta['standby_gap'],continue_approach_after_first_open=True,guard_prepare_contact=True)
    worker=hold_worker=retreat_worker=world=browser=None
    rows=[]; hits=[]; contacts=[]; samples=[]; motion=[]; domain=[]; error=None
    try:
        observer_pythonpath=os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH']='outputs/runtime/guarded-eval-person-20261029-01/src:outputs/runtime/guarded-eval-person-20261029-01/scripts:scripts:src'
        try:
            worker=PolicyWorker(RUNTIME.resolve(),normpath,CHECKPOINT,OUT/'worker.log',native_full=True,
                                prediction_chunk_size=1,expected_normalization_sha256=NORM_SHA)
            hold_worker=PolicyWorker(RUNTIME.resolve(),HOLD_MODEL/'normalization.json',HOLD_MODEL,OUT/'hold-worker.log',native_full=True,
                                     prediction_chunk_size=1,expected_normalization_sha256=NORM_SHA)
            retreat_worker=PolicyWorker(RUNTIME.resolve(),retreat_checkpoint/'normalization.json',retreat_checkpoint,OUT/'retreat-worker.log',native_full=True,
                                        prediction_chunk_size=1,expected_normalization_sha256=retreat_training['normalization_sha256'])
        finally:
            if observer_pythonpath is None:
                del os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH']=observer_pythonpath
        worker.call(operation='reset',seed=SEED)
        hold_worker.call(operation='reset',seed=SEED)
        retreat_worker.call(operation='reset',seed=SEED)
        world=PandaWorld(OUT,rest_gap=meta['rest_gap'],bottle_input=meta['bottle_input'],
            finger_release_speed_m_s=meta['finger_release_speed_m_s'],finger_press_speed_m_s=meta['finger_press_speed_m_s'],
            joint_velocity_cap=meta['joint_velocity_cap'],drop_model=MOVING_NOZZLE)
        browser=PandaBrowser(OUT/'browser.log',person=case['person'],hybrid_rgb=True)
        browser.call('setup',shapes=world.drawing())
        browser.call('reset',depth=case['depth_m'],phase=case['blink_phase_s'])
        world.reset(meta['initial_joints'])
        world.add_affine_geometry(browser.call('geometry'),browser.call('blinkGeometry'))
        history=history_from_normalization(norm,meta['rest_gap'])
        history.reset(world.own_state(),meta['rest_gap'])
        home_joints=world.own_state()[:7]
        path_context=OnlinePathContext(home_joints)
        release_gate=RetreatHomeGate(home_joints)
        start_nozzle=np.asarray(world.nozzle()); path=[home_joints]; held=None; fine_started=False
        for step in range(MAX_STEPS):
            frame=browser.call('render',state={'time':world.time,'links':world.links(),
                                              'eye_condition':'natural','capture_robot_mask':True},capture=True)
            blobs=[base64.b64decode(frame[k]) for k in ('left','right')]
            pred=observer.predict(*blobs,native_face=base64.b64decode(frame['face']))
            images=[np.array(Image.open(BytesIO(blob)).convert('RGB')) for blob in blobs]
            masks=[np.array(Image.open(BytesIO(base64.b64decode(frame['robot_mask_'+v]))).convert('L'))>0 for v in ('left','right')]
            reference=guard.update(images,masks,pred['openness'],step/20)
            valid=reference['valid']
            target=np.asarray(to_physics(reference['point_scene']))+np.array([0,0,TARGET_HEIGHT_M]) if valid else None
            distance=float(np.linalg.norm(np.asarray(world.nozzle())-target)) if valid else None
            aligned=bool(valid and distance<=ALIGN_M)
            decision=control.decide(openness=pred['openness'],gap=world.own_state()[7],aligned=aligned,position_valid=valid)
            state=history.observation()
            fine_started, action_source = select_source(fine_started, decision['arm'], valid, distance, FINE_SWITCH_DISTANCE_M)
            own_joints = world.own_state()[:7]
            decision=release_gate.adjust(decision,world.own_state()[7],own_joints)
            anchors = path_context.observe(decision['arm'],own_joints)
            retreat_input_state = None
            stabilize = should_stabilize(action_source, valid, distance, ALIGN_M)
            active_worker = hold_worker if stabilize else worker
            raw = active_worker.call(operation='act',state=state,left=frame['left'],right=frame['right'])['action'] if action_source=='learned_fine' else None
            if stabilize:
                action_source = 'learned_stabilize'
                target_joints = raw[:7]
            elif action_source == 'fixed_coarse':
                target_joints = synchronized_joint_target(own_joints, world.ik(target),
                    meta['joint_velocity_cap']/20, [(world.joint_info[i][8],world.joint_info[i][9]) for i in range(7)])
            else:
                target_joints = None if raw is None else raw[:7]
            if decision['arm']=='retreat':
                if release_gate.finished:
                    raw = None
                else:
                    if max(abs(a-b) for a,b in zip(state[:7],own_joints,strict=True)) > 1e-9:
                        raise ValueError('自己状態と実測関節が一致しません')
                    retreat_input_state = encode_runtime_state(state,anchors,retreat_norm['retreat_context_contract'])
                    raw = retreat_worker.call(operation='act',state=retreat_input_state,left=frame['left'],right=frame['right'])['action']
                limits = [(world.joint_info[i][8],world.joint_info[i][9]) for i in range(7)]
                def clamp(own, target):
                    return synchronized_joint_target(own,target,meta['joint_velocity_cap']/20,limits)
                action,action_source=retreat_action(release_gate,own_joints,raw,decision['gap'],clamp)
            else:
                action,held=combine_action(decision,own_joints,target_joints,path,held)
            outcome=world.step(action,audit_pressure=True,audit_motion=True)
            motion.extend(outcome['motion_samples']);domain.extend(outcome['drop_domain_samples'])
            points=browser.call('pressureSamples',samples=outcome['pressure_samples'])
            samples.extend(dict(p,command_gap_m=action[7]) for p in points)
            history.append(step+1,world.own_state(),action[7])
            hits.extend(browser.call('scoreSegments',segments=outcome.get('segments',[])))
            contacts.extend(outcome['contacts'])
            rows.append(dict(step=step,state=state,retreat_input_state=retreat_input_state,raw_action=raw,action=action,decision=decision,action_source=action_source,
                prediction=pred,reference=reference,distance_to_estimated_target_m=distance,audit_open=frame['open'],
                own=world.own_state(),nozzle=world.nozzle(),contacts=outcome['contacts']))
            for side,blob in zip(('left','right'),blobs):
                (OUT/f'{side}-{step:03}.png').write_bytes(blob)
            (OUT/f'face-{step:03}.png').write_bytes(base64.b64decode(frame['face']))
            atomic_json(OUT/'progress.json',{'steps':len(rows),'phase':decision['phase'],'drops':len(world.drops)})
            if contacts:
                break
    except Exception as exc:
        error=repr(exc)
    finally:
        if browser is not None: browser.close()
        if retreat_worker is not None: retreat_worker.close()
        if hold_worker is not None: hold_worker.close()
        if worker is not None: worker.close()
        if world is not None:
            final_gap=world.own_state()[7]
            retreat_error=float(np.linalg.norm(np.asarray(world.nozzle())-start_nozzle))
            drop_count=len(world.drops)
            world.close()
        else:
            final_gap=retreat_error=drop_count=None
    expected=len(rows)*SUBSTEPS
    fluid=audit_fluid(samples,motion,expected_intervals=expected) if rows else None
    physical=domain_summary(domain,expected_intervals=expected) if rows else None
    atomic_json(OUT/'trace.json',rows)
    atomic_json(OUT/'pressure-samples.json',samples)
    result=dict(completed=error is None,steps=len(rows),error=error,contacts=contacts,drops=drop_count,hits=hits,
        phase=control.phase,aborted=control.aborted,fluid=fluid,domain=physical,final_gap_m=final_gap,
        retreat_error_m=retreat_error,limit_steps=MAX_STEPS,
        action_source_counts={name:sum(r['action_source']==name for r in rows) for name in ('learned_fine','fixed_coarse','learned_stabilize','hold_or_retreat','learned_retreat','retreat_home_hold')},
        min_distance_to_estimated_target_m=min((r['distance_to_estimated_target_m'] for r in rows if r['distance_to_estimated_target_m'] is not None),default=None),
        scope='development condition; learned approach and retreat, fixed press, release, gripper, home stop and safety')
    atomic_json(OUT/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if error is not None: raise RuntimeError(error)

if __name__=='__main__':
    main()
