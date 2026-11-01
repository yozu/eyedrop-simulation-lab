"""Macの2試行と監査を待ち、照合した結果だけを回収する。"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from run_retreat_replay_20261101 import atomic

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / 'eyedrop_sim'
STATUS = SIM / 'outputs/dual-host-20261030/retreat-ed-collect-follow-20261101-02.json'
LOCK = STATUS.with_suffix('.lock')
REMOTE_ROOT = os.environ['EYEDROP_REMOTE_ROOT']
REMOTE_BASE = REMOTE_ROOT + 'outputs/diagnostics/curriculum-redesign-20261030-01/'
REMOTE_STATUS = REMOTE_ROOT + 'outputs/autonomous-cycles/retreat-ed-eval-20261101-01.json'
REMOTE_JOBS = REMOTE_ROOT + 'outputs/autonomous-jobs/'
AUDIT = HERE / 'retreat-ed-comparison-audit-20261101-01.json'
CASES = ('e-train04', 'd-val00')
TIMEOUT = 9000


def action_for(state):
    status = state.get('status')
    if status in ('running', 'running_audit'):
        return 'wait'
    if status == 'finished':
        return 'collect'
    if status == 'failed':
        return 'fail'
    raise ValueError('Mac評価の状態が不明です')


def run(command, timeout=60):
    return subprocess.run(command, text=True, encoding='utf8', capture_output=True, timeout=timeout,
                          check=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()


def ssh(command):
    return run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', os.environ['EYEDROP_SSH_HOST'], command])


def read_remote_status():
    for attempt in range(3):
        try:
            return json.loads(ssh('cat ' + REMOTE_STATUS))
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
            time.sleep(5)


def copy_verified(remote, local):
    if local.exists():
        raise FileExistsError('既存結果を上書きしません: ' + str(local))
    temporary = local.with_name('.' + local.name + '.part')
    if temporary.exists():
        raise FileExistsError('未完了の転送が残っています: ' + str(temporary))
    digest = ssh('shasum -a 256 ' + remote).split()[0]
    run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', os.environ['EYEDROP_SSH_HOST'] + ':' + remote, str(temporary)], 300)
    with temporary.open('rb') as file:
        actual = hashlib.file_digest(file, 'sha256').hexdigest()
    if actual != digest:
        raise ValueError('転送結果の識別情報が違います: ' + str(local))
    os.replace(temporary, local)
    return digest


def main():
    if STATUS.exists() or AUDIT.exists():
        raise FileExistsError('既存の回収または監査があります')
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(str(os.getpid()))
    state = {'status': 'waiting', 'pid': os.getpid(), 'started_at': datetime.now().astimezone().isoformat()}
    atomic(STATUS, state)
    try:
        deadline = time.monotonic() + TIMEOUT
        identity = None
        while time.monotonic() < deadline:
            remote = read_remote_status()
            action = action_for(remote)
            if action == 'fail':
                raise RuntimeError('Macの試行または監査が失敗しました')
            if action == 'collect':
                break
            pid = int(remote['pid'])
            try:
                observed = ssh('ps -p ' + str(pid) + ' -o lstart= -o command=')
            except subprocess.CalledProcessError:
                latest = read_remote_status()
                if action_for(latest) == 'wait':
                    raise RuntimeError('Mac評価の実プロセスを確認できません')
                continue
            if 'run_retreat_ed_eval_20261101.py' not in observed:
                raise RuntimeError('Mac評価プロセスのコマンドが違います')
            if identity is not None and observed != identity:
                raise RuntimeError('Mac評価プロセスの開始時刻またはコマンドが変わりました')
            identity = observed
            time.sleep(15)
        else:
            raise TimeoutError('Mac評価の終了を確認できません')
        if len(remote.get('completed', [])) != 2:
            raise ValueError('2条件の終了記録が揃っていません')
        state.update(status='collecting', mac_runner_identity=identity, mac_completed=remote['completed'])
        atomic(STATUS, state)
        digest = copy_verified(REMOTE_BASE + AUDIT.name, AUDIT)
        if digest != remote['audit_sha256']:
            raise ValueError('Macの完了記録と監査の識別情報が違います')
        audit = json.loads(AUDIT.read_text(encoding='utf8'))
        if not audit['completed'] or set(audit['cases']) != set(CASES):
            raise ValueError('監査結果が不完全です')
        for case in CASES:
            job = f'dev-retreat-ed-candidate-{case}-20261101-01'
            target = HERE / job
            target.mkdir(exist_ok=False)
            for name in ('protocol.json', 'result.json', 'trace.json'):
                value = copy_verified(REMOTE_BASE + job + '/' + name, target / name)
                if name == 'result.json' and value != audit['cases'][case]['candidate']['result_sha256']:
                    raise ValueError('試行結果と監査の識別情報が違います')
            for name in ('started.json', 'exit.json'):
                copy_verified(REMOTE_JOBS + job + '/' + name, target / name)
            exit_record = json.loads((target / 'exit.json').read_text(encoding='utf8'))
            if exit_record['returncode'] != 0 or exit_record['reason'] != 'exited' or exit_record['live_descendants']:
                raise ValueError('試行の終了記録が不完全です')
        state.update(status='finished', candidate_both_passed=audit['candidate_both_passed'],
                     audit_sha256=digest, finished_at=datetime.now().astimezone().isoformat(),
                     next='2人物の実動作を検討し、保持・閉眼残圧・未使用人物を別に報告する。')
        atomic(STATUS, state)
    except Exception:
        state.update(status='failed', traceback=traceback.format_exc())
        atomic(STATUS, state)
        raise
    finally:
        LOCK.unlink()


if __name__ == '__main__':
    main()
