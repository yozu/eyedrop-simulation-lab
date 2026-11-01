"""保存出力だけから、尺度の効果・平均誤差・乱数差・動作段階を分ける。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS=(714,715,716,717,718)
JOINTS=7
ACTION_DIM=8
MM_PER_M=1000
SPLITS=('train','validation')
SOURCES=('micro','sequence')


def matching_rows(left,right):
    if not left or len(left)!=len(right):raise ValueError('比較行の件数が異なります')
    if any({k:v for k,v in a.items() if k!='prediction'}!={k:v for k,v in b.items() if k!='prediction'}
           for a,b in zip(left,right)):
        raise ValueError('教師、条件、行順が変わっています')
    return True


def error_components(error):
    error=np.asarray(error,dtype=np.float64)
    if error.ndim!=3 or error.shape[0]!=len(SEEDS) or error.shape[1]<1 or error.shape[2]!=ACTION_DIM or not np.isfinite(error).all():
        raise ValueError('同じ5乱数・非空行・8成分の有限値が必要です')
    average=error.mean(axis=0);residual=error-average
    total=float(np.square(error[...,:JOINTS]).mean())
    mean=float(np.square(average[:,:JOINTS]).mean());variance=float(np.square(residual[...,:JOINTS]).mean())
    if not np.isclose(total,mean+variance,rtol=1e-10,atol=1e-15):raise ValueError('二乗誤差の分解が一致しません')
    return dict(count=error.shape[1],joint_rmse_rad=total**.5,
        mean_prediction_joint_rmse_rad=mean**.5,seed_residual_joint_rms_rad=variance**.5,
        mean_error_fraction=mean/total if total else 0.,
        joint_rmse_per_axis_rad=np.sqrt(np.square(error[...,:JOINTS]).mean(axis=(0,1))).tolist(),
        mean_prediction_gap_max_mm=float(np.abs(average[:,JOINTS]).max()*MM_PER_M),
        gap_seed_range_max_mm=float(np.ptp(error[...,JOINTS],axis=0).max()*MM_PER_M))


def main():
    from relative_sequence_data import command_metrics
    from relative_sequence_training import compatible_norms
    from run_relative_sequence_cycle import compare_contracts
    from resumable_job import atomic_json
    from train_smolvla_micro_conditioning import check_metrics,OPEN_MINIMUM,PRESS_GAP_M
    parser=argparse.ArgumentParser();parser.add_argument('cycle');parser.add_argument('output');args=parser.parse_args()
    cycle,out=Path(args.cycle),Path(args.output)
    if out.exists():raise ValueError('完了出力は上書きしません')
    hashes={}
    def read(path):
        path=Path(path);raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    plan=read(cycle/'plan.json');finished=read(cycle/'comparison.json')
    if not finished['completed'] or not finished['one_factor_contract']:raise ValueError('固定比較が未完了です')
    train=[Path(p['output']) for p in plan if p['training']]
    evaluations=[Path(p['output']) for p in plan if not p['training']]
    if len(train)!=2 or len(evaluations)!=2:raise ValueError('事前登録した2候補ではありません')
    protocols=[read(p/'protocol.json') for p in train];compare_contracts(*protocols)
    norms=[read(p/'policy-final/normalization.json') for p in train]
    if not compatible_norms(*norms):raise ValueError('尺度以外の設定が異なります')
    summaries={};canonical={};seed714={}
    for candidate,evaluation,protocol,norm in zip(train,evaluations,protocols,norms):
        scale=protocol['scale'];trained=read(candidate/'result.json');ev=read(evaluation/'result.json')
        if (not trained['fixed_updates_completed'] or not ev['completed'] or
            trained['model_sha256']!=ev['model_sha256'] or trained['weights_fixed_at']!=ev['frozen_before_validation']):
            raise ValueError('重み固定・更新数・評価結果の対応が不一致です')
        summaries[scale]={};seed714[scale]={}
        for split in SPLITS:
            before=read(Path(protocol['connection'])/f'before-{scale}-{split}.json')
            outputs=[read(evaluation/f'noise-{seed}-{split}.json') for seed in SEEDS]
            summaries[scale][split]={};seed714[scale][split]={}
            for source in SOURCES:
                rows=[o[source]['rows'] for o in outputs];key=(split,source)
                matching_rows(before[source]['rows'],rows[0])
                for rs in rows:
                    matching_rows(rows[0],rs)
                    if key in canonical:matching_rows(canonical[key],rs)
                canonical[key]=rows[0]
                error=np.asarray([[np.asarray(r['prediction'])-r['reference'] for r in rs] for rs in rows])
                summary=error_components(error)
                summary['normalized_joint_rmse']=float(np.sqrt(np.square(error[...,:JOINTS]/np.asarray(norm['features']['action']['std'][:JOINTS])).mean()))
                if source=='sequence':
                    summary['phases']={phase:error_components(error[:,[i for i,r in enumerate(rows[0]) if r['phase']==phase]])
                        for phase in ('prepare','press','release','retreat')}
                # 平均出力は誤差の診断に限り、新しい制御方式へ採用しない。
                mean_prediction=np.asarray([[r['prediction'] for r in rs] for rs in rows]).mean(axis=0)
                mean_rows=[dict(row,prediction=mean_prediction[i].tolist()) for i,row in enumerate(rows[0])]
                summary['mean_prediction_diagnostic_metrics']=check_metrics(mean_rows) if source=='micro' else command_metrics(mean_rows)
                if source=='micro':summary['closed_half_false_press_each_seed']=[sum(r['actual_open']<OPEN_MINIMUM and r['prediction'][7]<PRESS_GAP_M for r in rs) for rs in rows]
                summary['all_fixed_seeds_passed']=all(o[source]['metrics']['passed'] for o in outputs)
                summaries[scale][split][source]=summary
                seed714[scale][split][source]={'before':before[source]['metrics'],'after':outputs[0][source]['metrics']}
    atomic_json(out,dict(completed=True,updates=0,new_inferences=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files_sha256=hashes,summaries=summaries,paired_before_after_seed714=seed714,adopted=False,full_task_success=False,
        limit='同じ画像の生成乱数差を分解する診断。平均を新しい方策へ採用しない。閉ループ動作・人物汎化を証明しない。初期尺度だけで変わる復元量と学習改善を区別する。'))


if __name__=='__main__':main()
