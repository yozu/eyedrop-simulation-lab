"""退避の追加学習を重複させず、開始・終了と成果を記録する。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / 'eyedrop_sim'
BASE = SIM / 'outputs/dual-host-20261030'
PYTHON = SIM / '.venv-smolvla-cuda-20261030/Scripts/python.exe'
SCRIPT = HERE / 'train_retreat_replay_20261101.py'
LOCK = BASE / 'retreat-replay-20261101.lock'
TIMEOUT = 7200


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'candidate'), required=True)
    mode = parser.parse_args().mode
    status = BASE / f'retreat-replay-{mode}-runner-20261101-01.json'
    log = BASE / f'retreat-replay-{mode}-runner-20261101-01.log'
    result_path = BASE / f'cuda-retreat-replay-{mode}-20261101-01/result.json'
    if status.exists() or log.exists() or result_path.parent.exists():
        raise FileExistsError('既存の学習を重複実行しません')
    if mode == 'candidate':
        smoke = json.loads((BASE / 'retreat-replay-smoke-runner-20261101-01.json').read_text(encoding='utf8'))
        if smoke['status'] != 'finished' or smoke['returncode'] != 0:
            raise ValueError('1更新の接続確認が未完了です')
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(str(os.getpid()))
    state = {'status': 'starting', 'runner_pid': os.getpid(), 'mode': mode}
    atomic(status, state)
    try:
        env = dict(os.environ)
        env['PYTHONPATH'] = 'outputs/dual-host-20261030/runtime-v2;src;scripts;../evidence/development/curriculum-redesign-20261030-01'
        env['HF_HUB_OFFLINE'] = '1'
        command = [str(PYTHON), '-u', str(SCRIPT), '--mode', mode]
        script_sha = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        with log.open('x', encoding='utf8') as file:
            child = subprocess.Popen(command, cwd=SIM, env=env, stdin=subprocess.DEVNULL,
                                     stdout=file, stderr=subprocess.STDOUT)
            state.update(status='running', child_pid=child.pid, command=command,
                         started_at=datetime.now().astimezone().isoformat(),
                         script_sha256=script_sha, timeout_seconds=TIMEOUT)
            atomic(status, state)
            try:
                code = child.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                raise TimeoutError('学習の実行上限に達しました')
        state.update(returncode=code, finished_at=datetime.now().astimezone().isoformat())
        if code:
            raise RuntimeError('学習が正常終了しませんでした')
        result = json.loads(result_path.read_text(encoding='utf8'))
        if not result['completed'] or result['updates'] != (1 if mode == 'smoke' else 320):
            raise ValueError('学習結果が不完全です')
        state.update(status='finished', result=str(result_path),
                     result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
        atomic(status, state)
    except Exception:
        state.update(status='failed', traceback=traceback.format_exc())
        atomic(status, state)
        raise
    finally:
        LOCK.unlink()


if __name__ == '__main__':
    main()
