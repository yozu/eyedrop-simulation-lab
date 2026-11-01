"""訓練の姿勢範囲と開発Eの退避中の実姿勢を比べる。"""
import hashlib
import json
from pathlib import Path
import statistics

from retreat_diverse_coverage_20261101 import summarize

HERE = Path(__file__).resolve().parent
PARTITION = HERE / 'retreat-diverse-home-training-partition-20261101-01.json'
TRACE = HERE / 'retreat-diverse-training-receipts-20261101-01/first-diagnostic/trace.json'
RESULT = HERE / 'retreat-diverse-coverage-e-20261101-01.json'


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    if RESULT.exists():
        raise FileExistsError('既存の診断を上書きしません')
    data = json.loads(PARTITION.read_text(encoding='utf8'))
    trace = json.loads(TRACE.read_text(encoding='utf8'))
    training = data['items']['train']
    rows = [r for r in trace if r['action_source'] == 'learned_retreat']
    if not data['passed'] or len(training) != 350 or len(trace) != 1200 or len(rows) != 1010:
        raise ValueError('比較対象または操作数が一致しません')
    observed = [row['state'][:7] for row in rows]
    groups = {'all': training, 'old': [r for r in training if r['archive'] == 'old'],
              'new': [r for r in training if r['archive'] == 'new']}
    reports = {}
    for name, group in groups.items():
        result = summarize([r['state'][:7] for r in group], observed)
        values = [r['nearest_linf_rad'] for r in result['records']]
        result.update(training_points=len(group), median_nearest_linf_rad=statistics.median(values),
                      first_nearest_linf_rad=values[0], last_nearest_linf_rad=values[-1])
        reports[name] = result
    payload = {'completed': True, 'partition_sha256': digest(PARTITION), 'trace_sha256': digest(TRACE),
               'steps': [r['step'] for r in rows], 'groups': reports,
               'scope': '開発Eの1連続系列。訓練例の姿勢範囲診断であり、失敗原因や一般化を証明しない。'}
    with RESULT.open('x', encoding='utf8', newline='\n') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({name: {k: v for k, v in group.items() if k != 'records'}
                      for name, group in reports.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
