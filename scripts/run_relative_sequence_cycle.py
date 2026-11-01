"""許可済み2候補を直列実行し、結果評価と再開記録を残す。"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from budget_cycle import experiment_timeout, collect_review
from resumable_job import atomic_json, run_job, status

PYTHON = '.venv-lerobot-smolvla/bin/python'
JOBS = Path('outputs/autonomous-jobs')
TRAIN_TIMEOUT = 2400
EVALUATION_TIMEOUT = 1800


def completed_stage(receipt, result, *, training):
    if receipt.get('returncode') != 0 or receipt.get('live_descendants'):
        raise ValueError('終了失敗または子プロセス残存。自動再実行しません')
    if not result or result.get('completed') is not True:
        raise ValueError('結果が未完了です')
    if training and result.get('fixed_updates_completed') is not True:
        raise ValueError('固定更新数に届いていないため尺度比較へ進みません')
    return True


def compare_contracts(left, right):
    if (left['scale'],right['scale']) != ('old','relative'):
        raise ValueError('旧尺度と相対尺度の組合せが不正です')
    for key in ('initial_model_sha256','initial_rng_sha256','samples_sha256','labels_sha256',
                'batches_sha256','source_sha256','helper_sha256','official_update_source_sha256',
                'updates','batch_size','learning_rate','warmup','grad_clip','padding_loss','time_sampling'):
        if left[key] != right[key]:raise ValueError('固定条件が変わっています: '+key)
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('policy');parser.add_argument('connection');parser.add_argument('output')
    parser.add_argument('--attempt', type=int, default=1)
    args=parser.parse_args();policy_path,connection,out=map(Path,(args.policy,args.connection,args.output))
    if args.attempt < 1:raise ValueError('実行番号は正の整数です')
    suffix=f'{args.attempt:02d}'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'cycle.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=status(JOBS)
        if current['locked'] or (current['current'] and current['current']['live_descendants']):
            raise RuntimeError('別の実験が実行中です。重複起動しません')
        qa=json.loads((connection/'result.json').read_bytes())
        if not qa['ready_for_training']:raise ValueError('接続確認が不合格です')
        for key,value in dict(PYTHONPATH='scripts:outputs/runtime/panda-hybrid-v1/src',
                              HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1').items():os.environ[key]=value
        atomic_json(out/'runner.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
            command=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        plans=[]
        for scale in ('old','relative'):
            candidate=Path(f'outputs/learning/smolvla-sequence-{scale}-{suffix}')
            evaluation=Path(f'outputs/diagnostics/smolvla-sequence-{scale}-eval-{suffix}')
            plans.extend([
                dict(id=f'smolvla-sequence-{scale}-train-{suffix}',training=True,output=str(candidate),timeout=TRAIN_TIMEOUT,
                    command=[PYTHON,'scripts/train_smolvla_relative_sequence.py',str(connection),scale,str(candidate)]),
                dict(id=f'smolvla-sequence-{scale}-eval-{suffix}',training=False,output=str(evaluation),timeout=EVALUATION_TIMEOUT,
                    command=[PYTHON,'scripts/evaluate_smolvla_relative_sequence.py',str(candidate),str(evaluation)])])
        atomic_json(out/'plan.json',plans)
        decisions=[];ending='failed_review_due'
        try:
            for plan in plans:
                policy=json.loads(policy_path.read_bytes())
                receipt_path=JOBS/plan['id']/'exit.json';result_path=Path(plan['output'])/'result.json'
                if receipt_path.exists():
                    receipt=json.loads(receipt_path.read_bytes())
                else:
                    if (JOBS/plan['id']).exists() or Path(plan['output']).exists():
                        raise RuntimeError('中断出力が存在します。整合性を手動照合してから再開してください')
                    timeout=experiment_timeout(policy,datetime.now(timezone.utc),plan['timeout'])
                    if plan['id'] not in policy['job_ids']:policy['job_ids'].append(plan['id'])
                    if str(result_path) not in policy['evidence']:policy['evidence'].append(str(result_path))
                    atomic_json(policy_path,policy)
                    atomic_json(out/'progress.json',dict(status='running',current=plan,decisions=decisions))
                    receipt=run_job(JOBS,plan['id'],plan['command'],timeout)
                result=json.loads(result_path.read_bytes()) if result_path.exists() else None
                completed_stage(receipt,result,training=plan['training'])
                decision=dict(id=plan['id'],result=str(result_path),returncode=receipt['returncode'],
                    elapsed_seconds=receipt.get('elapsed_seconds'),
                    gate_passed=result.get('train_BC_passed') if plan['training'] else result['robust_BC_passed'],
                    next_reason='事前登録した同一条件の2尺度比較を続行。性能未達でも別の学習候補は追加しない',
                    full_task_success=False)
                decisions.append(decision)
                with (out/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(decision,ensure_ascii=False)+'\n')
            protocols=[json.loads((Path(plans[i]['output'])/'protocol.json').read_bytes()) for i in (0,2)]
            compare_contracts(*protocols)
            comparisons={}
            for i in (1,3):
                result=json.loads((Path(plans[i]['output'])/'result.json').read_bytes())
                comparisons[result['scale']]=result
            atomic_json(out/'comparison.json',dict(completed=True,one_factor_contract=True,
                comparisons=comparisons,adopted=False,full_task_success=False,
                review_required=True,scope='同じ連続300時点と開閉9対照、各split・5乱数の単時点比較。全動作ではない'))
            ending='comparison_completed_review_due'
        except Exception as error:
            ending='failed_review_due'
            atomic_json(out/'failure.json',dict(type=type(error).__name__,message=str(error),decisions=decisions))
            raise
        finally:
            policy=json.loads(policy_path.read_bytes());policy['status']=ending;policy['review_required']=True
            atomic_json(policy_path,policy)
            atomic_json(out/'progress.json',dict(status=ending,decisions=decisions,agent_review_required=True))
            atomic_json(out/'review-input.json',collect_review(policy,Path.cwd()))


if __name__=='__main__':main()
