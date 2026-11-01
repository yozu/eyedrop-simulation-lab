"""現在の実験が切り替わっても、指定した実験の終了記録を照合する。"""
import argparse
import json
from pathlib import Path

from resumable_job import live_descendants, process_info

ROOT = Path('outputs/autonomous-jobs')


def inspect(job):
    if not job or Path(job).name != job or job in ('.', '..'):
        raise ValueError('実験IDが不正です')
    folder = ROOT / job
    current = json.loads((folder / 'started.json').read_text(encoding='utf8'))
    if current['id'] != job:
        raise ValueError('実験IDが一致しません')
    current['live_process'] = process_info(current['pid'])
    current['identity_matches'] = bool(current['live_process']) and current['live_process'] == current['process_identity']
    receipt = folder / 'exit.json'
    current['exit'] = json.loads(receipt.read_text(encoding='utf8')) if receipt.exists() else None
    current['live_descendants'] = live_descendants(folder)
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    print(json.dumps({'current': inspect(args.job)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
