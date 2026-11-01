"""連続教材の終了を照合し、記録と画像アーカイブを回収・監査する。"""
from datetime import datetime
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import time
import traceback
from zipfile import ZipFile

from retreat_ed_collect_20261101 import copy_verified, ssh
from run_retreat_replay_20261101 import atomic

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / 'eyedrop_sim'
STATUS = SIM / 'outputs/dual-host-20261030/retreat-diverse-home-dataset-collect-20261101-01.json'
LOCK = STATUS.with_suffix('.lock')
JOB = 'retreat-diverse-home-data-20261101-01'
COMMAND_SHA = 'b1ee37ce6bd53793af7b8f3c5bc43bbf8520708542462ca681eb1137b7b20640'
DEST = HERE / JOB
ARCHIVE = SIM / 'outputs/dual-host-20261030/retreat-diverse-home-data-20261101-01.zip'
REMOTE = os.environ['EYEDROP_REMOTE_ROOT']
TIMEOUT = 2100
WAIT_SECONDS = 20
SSH_ATTEMPTS = 3
TOLERANCE = 1e-8


def verify_archive(path, result, protocol_bytes=None, result_bytes=None):
    with ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None:
            raise ValueError('アーカイブが破損または重複しています')
        if protocol_bytes is not None and z.read('protocol.json') != protocol_bytes:
            raise ValueError('収録された条件が違います')
        if result_bytes is not None and z.read('result.json') != result_bytes:
            raise ValueError('収録された結果が違います')
        expected = {'protocol.json', 'result.json'}
        images = 0
        for episode in result['episodes']:
            record_name = f"episode-{episode['index']:02}/records.json"
            raw = z.read(record_name)
            if hashlib.sha256(raw).hexdigest() != episode['records_sha256']:
                raise ValueError('系列の記録が違います')
            records = json.loads(raw)
            if len(records) != episode['records']:
                raise ValueError('系列の操作数が違います')
            expected.add(record_name)
            for row in records:
                for image in row['images'].values():
                    name = image['file']
                    if name in expected or hashlib.sha256(z.read(name)).hexdigest() != image['sha256']:
                        raise ValueError('画像が重複または一致しません')
                    expected.add(name)
                    images += 1
        if set(names) != expected or images != result['images']:
            raise ValueError('収録ファイルまたは画像数が違います')
        return images


def remote_action(state):
    current = state.get('current') or {}
    if current.get('id') != JOB or current.get('command_sha256') != COMMAND_SHA:
        raise ValueError('Macの実験またはコマンドが違います')
    receipt = current.get('exit')
    if receipt is None:
        if not current.get('identity_matches'):
            raise ValueError('Macの実プロセスを確認できません')
        return 'wait'
    if receipt.get('live_descendants') or current.get('live_descendants'):
        raise ValueError('Macの子プロセスが残っています')
    return 'collect' if receipt.get('returncode') == 0 and receipt.get('reason') == 'exited' else 'failed'


def get_remote_status():
    command = ('cd ' + shlex.quote(REMOTE) + ' && '
               'PYTHONPATH=outputs/diagnostics/curriculum-redesign-20261030-01:scripts:src '
               '.venv-learning/bin/python outputs/diagnostics/curriculum-redesign-20261030-01/'
               'inspect_autonomous_job.py ' + JOB)
    for attempt in range(SSH_ATTEMPTS):
        try:
            return json.loads(ssh(command))
        except subprocess.TimeoutExpired:
            if attempt == SSH_ATTEMPTS - 1:
                raise
            time.sleep(5)


def main():
    if STATUS.exists() or DEST.exists() or ARCHIVE.exists():
        raise FileExistsError('既存の回収記録を上書きしません')
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    state = {'status': 'waiting', 'pid': os.getpid(), 'started_at': datetime.now().astimezone().isoformat(), 'job': JOB}
    atomic(STATUS, state)
    try:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            current = get_remote_status()
            action = remote_action(current)
            if action != 'wait':
                break
            time.sleep(WAIT_SECONDS)
        else:
            raise TimeoutError('Macジョブの終了を確認できません')
        state.update(status='collecting', mac_started_at=current['current']['started_at'], mac_command_sha256=COMMAND_SHA)
        atomic(STATUS, state)
        DEST.mkdir()
        job = REMOTE + 'outputs/autonomous-jobs/' + JOB + '/'
        for name in ('started.json', 'exit.json', 'stdout.log'):
            copy_verified(job + name, DEST / name)
        if action == 'failed':
            raise RuntimeError('Macの収集が正常終了しなかったため終了記録とログを保持しました')
        root = REMOTE + 'outputs/diagnostics/curriculum-redesign-20261030-01/' + JOB + '/'
        for name in ('protocol.json', 'result.json'):
            copy_verified(root + name, DEST / name)
        result = json.loads((DEST / 'result.json').read_text(encoding='utf8'))
        expected_ready = (result['completed'] and len(result['episodes']) == 36 and len(result['groups']) == 12
                          and all(row['passed'] for row in result['groups'])
                          and result['train_validation_image_overlap'] == 0 and result['fixed_evaluation_image_overlap'] == 0
                          and result['max_context_error'] <= TOLERANCE and result['max_action_error'] <= TOLERANCE)
        if result['dataset_ready'] != expected_ready or result['smoke']:
            raise ValueError('教材の品質判定が一致しません')
        state.update(status='finished_not_ready', dataset_ready=False, episodes=len(result['episodes']),
                     images=result['images'], result_sha256=hashlib.sha256((DEST / 'result.json').read_bytes()).hexdigest())
        if result['dataset_ready']:
            copy_verified(root + 'dataset-receipt.json', DEST / 'dataset-receipt.json')
            receipt = json.loads((DEST / 'dataset-receipt.json').read_text(encoding='utf8'))
            digest = copy_verified(root + 'dataset.zip', ARCHIVE)
            if digest != receipt['sha256'] or ARCHIVE.stat().st_size != receipt['bytes']:
                raise ValueError('教材アーカイブの識別情報が違います')
            images = verify_archive(ARCHIVE, result, (DEST / 'protocol.json').read_bytes(), (DEST / 'result.json').read_bytes())
            state.update(status='finished', dataset_ready=True, archive_sha256=digest,
                         archive_bytes=receipt['bytes'], verified_images=images)
        state['finished_at'] = datetime.now().astimezone().isoformat()
        state['next'] = '教材品質と分割を評価し、成立した場合は固定比較に基づき学習を準備する。'
        atomic(STATUS, state)
    except Exception:
        state.update(status='failed', traceback=traceback.format_exc())
        atomic(STATUS, state)
        raise
    finally:
        LOCK.unlink()


if __name__ == '__main__':
    main()
