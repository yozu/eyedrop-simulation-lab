"""固定退避を確認後、退避モデルの追加前後を開発2人物で順に比較する。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from inspect_autonomous_job import inspect
from resumable_job import atomic_json

BASE = Path('outputs/diagnostics/curriculum-redesign-20261030-01')
ROOT = Path('outputs/autonomous-jobs')
STATUS = Path('outputs/autonomous-cycles/learned-retreat-comparison-20261101-01.json')
STATIC = BASE / 'cuda-retreat-current-static-20261101-01.json'
FIXED = {'e-train04': 'dev-e-retreat-current-fixed-e-20261101-01',
         'd-val00': 'dev-e-recovery-half-gap-dev-d-candidate-20261101-01'}
CASES = ('e-train04', 'd-val00')
MODES = ('baseline', 'candidate')
MAIN_SHA = 'b69054ec7138e65dabb6bb9e07cfdbd735b6fe7410a6eae7f877a647031d7043'
FULL_STEPS = 1200
REST_GAP_M = .0343
GAP_TOLERANCE_M = .0001
RETREAT_TOLERANCE_M = .001


def check_fixed(folder):
    result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf8'))
    hits = result['hits']
    return (result['completed'] and result['steps'] == FULL_STEPS and not result['contacts']
            and not result['aborted'] and result['drops'] == 1 and len(hits) == 1
            and hits[0]['result'] == 'target' and hits[0]['open'] == 1
            and result['fluid']['passed'] and result['domain']['valid']
            and abs(result['final_gap_m'] - REST_GAP_M) < GAP_TOLERANCE_M
            and result['phase'] == 'retreat' and result['retreat_error_m'] < RETREAT_TOLERANCE_M
            and protocol['model_sha256'] == MAIN_SHA)


def main():
    with STATUS.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATUS.exists():
            raise FileExistsError(STATUS)
        static = json.loads(STATIC.read_text(encoding='utf8'))
        if not static['completed'] or not static['passed'] or static['baseline_sha256'] != MAIN_SHA:
            raise ValueError('退避の固定画像の基準を確認できません')
        for case, name in FIXED.items():
            if not check_fixed(BASE / name):
                raise ValueError('固定退避の比較元が成立していません: ' + case)
        env = dict(os.environ)
        env['PYTHONPATH'] = ('outputs/diagnostics/curriculum-redesign-20261030-01:'
                             'outputs/runtime/review-sklearn-v1/src:'
                             'outputs/runtime/guarded-eval-person-20261029-01/scripts:scripts:src')
        completed = []
        for case in CASES:
            for mode in MODES:
                job = f'dev-retreat-learned-{case}-{mode}-20261101-01'
                folder = BASE / job
                if (ROOT / job).exists() or folder.exists():
                    raise FileExistsError('既存試行を重複実行しません: ' + job)
                command = ['.venv-learning/bin/python', str(BASE / 'evaluate_learned_retreat_20261101.py'),
                           '--case', case, '--retreat-mode', mode]
                digest = hashlib.sha256(json.dumps(command).encode()).hexdigest()
                runner = ['.venv-learning/bin/python', 'scripts/resumable_job.py', 'run', '--root', str(ROOT),
                          '--id', job, '--timeout', '1800', '--', *command]
                atomic_json(STATUS, {'status': 'running', 'pid': os.getpid(), 'job': job,
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
                        or protocol['environment_person'] != person or protocol['retreat_mode'] != mode
                        or protocol['model_sha256'] != MAIN_SHA):
                    raise ValueError('試行結果と事前条件が一致しません: ' + job)
                completed.append({'job': job, 'steps': result['steps'], 'contacts': result['contacts'],
                                  'drops': result['drops'], 'domain_valid': result['domain']['valid'],
                                  'retreat_error_m': result['retreat_error_m']})
                atomic_json(STATUS, {'status': 'running', 'pid': os.getpid(), 'job': job, 'completed': completed})
        atomic_json(STATUS, {'status': 'finished', 'pid': os.getpid(), 'completed': completed,
                             'next': '開始区間と各段階を固定退避・追加前と比較し採否を判断する。'})


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
