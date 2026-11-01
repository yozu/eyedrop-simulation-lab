"""E・D退避モデルを開発人物2条件で順に評価する。"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from inspect_autonomous_job import inspect
from resumable_job import atomic_json
from run_learned_retreat_comparison_20261101 import check_fixed, FIXED, MAIN_SHA

BASE = Path('outputs/diagnostics/curriculum-redesign-20261030-01')
ROOT = Path('outputs/autonomous-jobs')
STATUS = Path('outputs/autonomous-cycles/retreat-ed-eval-20261101-01.json')
STATIC = BASE / 'retreat-ed-static-20261101-01.json'
PACKAGE_SHA = '3aba1c938a6ceb8c6a331aa28cf0e5586b21bb786ff6da5ecb12a5bf855ce040'
CASES = ('e-train04', 'd-val00')
AUDIT_JOB = 'dev-retreat-ed-comparison-audit-20261101-01'
AUDIT_OUT = BASE / 'retreat-ed-comparison-audit-20261101-01.json'


def main():
    with STATUS.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATUS.exists() or AUDIT_OUT.exists() or (ROOT / AUDIT_JOB).exists():
            raise FileExistsError('既存の比較を重複実行しません')
        static = json.loads(STATIC.read_text(encoding='utf8'))
        trained = json.loads((BASE / 'dual-host-retreat-ed-candidate-20261101-01/result.json').read_text(encoding='utf8'))
        if (not static['completed'] or static['package_sha256'] != PACKAGE_SHA
                or not trained['completed'] or trained['updates'] != 320 or trained['mode'] != 'candidate'
                or trained['smoke'] or static['models']['candidate']['model_sha256'] != trained['model_sha256']
                or static['models']['candidate']['normalization_sha256'] != trained['normalization_sha256']):
            raise ValueError('固定画像比較と学習モデルが違います')
        for case, name in FIXED.items():
            if not check_fixed(BASE / name):
                raise ValueError('固定退避の比較元が成立していません: ' + case)
        env = dict(os.environ)
        env['PYTHONPATH'] = ('outputs/diagnostics/curriculum-redesign-20261030-01:'
                             'outputs/runtime/review-sklearn-v1/src:'
                             'outputs/runtime/guarded-eval-person-20261029-01/scripts:scripts:src')
        completed = []
        for case in CASES:
            job = f'dev-retreat-ed-candidate-{case}-20261101-01'
            folder = BASE / job
            if (ROOT / job).exists() or folder.exists():
                raise FileExistsError('既存試行を重複実行しません: ' + job)
            command = ['.venv-learning/bin/python', str(BASE / 'evaluate_retreat_ed_20261101.py'),
                       '--case', case]
            digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
            runner = ['.venv-learning/bin/python', 'scripts/resumable_job.py', 'run', '--root', str(ROOT),
                      '--id', job, '--timeout', '1800', '--', *command]
            atomic_json(STATUS, {'status': 'running', 'pid': os.getpid(), 'job': job,
                                 'started_at': datetime.now(timezone.utc).isoformat(),
                                 'command_sha256': digest, 'completed': completed})
            with (BASE / (job + '.runner.log')).open('x') as log:
                child = subprocess.run(runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            state = inspect(job)
            receipt = state.get('exit')
            if (child.returncode or state.get('command_sha256') != digest or receipt is None
                    or receipt.get('returncode') != 0 or receipt.get('reason') != 'exited'
                    or receipt.get('live_descendants') or state.get('live_descendants')):
                raise ValueError('試行の終了を確認できません: ' + job)
            result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
            protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf8'))
            trace = json.loads((folder / 'trace.json').read_text(encoding='utf8'))
            person = 'vroid-beta-e' if case == 'e-train04' else 'vroid-beta-d'
            if (not result['completed'] or len(trace) != result['steps'] or protocol['seed'] != 714
                    or protocol['environment_person'] != person or protocol['retreat_context_mode'] != 'candidate'
                    or protocol['model_sha256'] != MAIN_SHA or protocol['retreat_model_sha256'] != trained['model_sha256']
                    or protocol['retreat_normalization_sha256'] != trained['normalization_sha256']
                    or protocol.get('retreat_orientation') != 'fixed_ik_separate_world'):
                raise ValueError('試行結果と固定条件が違います: ' + job)
            completed.append({'job': job, 'steps': result['steps'], 'contacts': result['contacts'],
                              'drops': result['drops'], 'domain': result['domain'],
                              'retreat_error_m': result['retreat_error_m']})
            atomic_json(STATUS, {'status': 'running', 'pid': os.getpid(), 'job': job, 'completed': completed})
        command = ['.venv-learning/bin/python', str(BASE / 'audit_retreat_ed_eval_20261101.py')]
        digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        runner = ['.venv-learning/bin/python', 'scripts/resumable_job.py', 'run', '--root', str(ROOT),
                  '--id', AUDIT_JOB, '--timeout', '300', '--', *command]
        atomic_json(STATUS, {'status': 'running_audit', 'pid': os.getpid(), 'job': AUDIT_JOB,
                             'command_sha256': digest, 'completed': completed})
        with (BASE / (AUDIT_JOB + '.runner.log')).open('x') as log:
            child = subprocess.run(runner, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        state = inspect(AUDIT_JOB)
        receipt = state.get('exit')
        if (child.returncode or state.get('command_sha256') != digest or receipt is None
                or receipt.get('returncode') != 0 or receipt.get('reason') != 'exited'
                or receipt.get('live_descendants') or state.get('live_descendants')):
            raise ValueError('比較監査の終了を確認できません')
        audit = json.loads(AUDIT_OUT.read_text(encoding='utf8'))
        if not audit.get('completed') or set(audit['cases']) != set(CASES):
            raise ValueError('比較監査が不完全です')
        atomic_json(STATUS, {'status': 'finished', 'pid': os.getpid(), 'completed': completed,
                             'candidate_both_passed': audit['candidate_both_passed'],
                             'audit_sha256': hashlib.sha256(AUDIT_OUT.read_bytes()).hexdigest(),
                             'finished_at': datetime.now(timezone.utc).isoformat(),
                             'next': '結果と限界を評価し、閉眼残圧と未使用人物を別に扱う。'})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if STATUS.exists():
            current = json.loads(STATUS.read_text(encoding='utf8'))
            if current.get('pid') == os.getpid() and current.get('status') != 'finished':
                current.update(status='failed', traceback=traceback.format_exc())
                atomic_json(STATUS, current)
        raise
