"""旧教材と新系列を、姿勢・方向と展開画像で分割監査する。"""
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
from zipfile import ZipFile

from PIL import Image

from retreat_coarse_context_20261101 import coarse_context
from retreat_diverse_home_partition_20261101 import partition_records
from retreat_path_context_20261101 import anchors_from_trace, encode_context

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[2] / 'eyedrop_sim'
OLD = SIM / 'outputs/dual-host-20261030/retreat-ed-package-20261101-01.zip'
NEW = SIM / 'outputs/dual-host-20261030/retreat-diverse-home-data-20261101-01.zip'
RESULT = HERE / 'retreat-diverse-home-data-20261101-01/result.json'
OUT = HERE / 'retreat-diverse-home-training-partition-20261101-01.json'
AUDIT = HERE / 'retreat-diverse-home-training-partition-audit-20261101-01.json'
OLD_SHA = '3aba1c938a6ceb8c6a331aa28cf0e5586b21bb786ff6da5ecb12a5bf855ce040'
NEW_SHA = 'f4fcdf9a58acd98ef2ce610f98ec15e8a9e0b1903b65b0f037e43c81b6a15415'
RESULT_SHA = 'e8541ec8d99614e6bbf1ee6280c8ee04ee92019927860ce10f76d70020433d4c'
SPLITS = ('train', 'development', 'heldout_home')
CONTEXTS = ('context_coarse', 'context_exact')
MIN_STD = 1e-6
TOLERANCE = 1e-9
IMAGE_SIZE = (512, 512)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pixel_sha(raw):
    with Image.open(BytesIO(raw)) as image:
        image = image.convert('RGB')
        if image.size != IMAGE_SIZE:
            raise ValueError('画像サイズが違います')
        return hashlib.sha256(image.tobytes()).hexdigest()


def conflicts(rows, context_key):
    groups = {}
    for row in rows:
        key = (tuple(row['pixel_sha256'][side] for side in ('left', 'right')), tuple(row['state'][:16]),
               tuple(row[context_key]), tuple(row['state'][30:]))
        groups.setdefault(key, []).append(row['action'][:7])
    return sum(any(max(abs(a-b) for a, b in zip(actions[0], other, strict=True)) > TOLERANCE
                   for other in actions[1:]) for actions in groups.values())


def main():
    if OUT.exists() or AUDIT.exists():
        raise FileExistsError('既存の分割を上書きしません')
    if sha(OLD) != OLD_SHA or sha(NEW) != NEW_SHA or sha(RESULT) != RESULT_SHA:
        raise ValueError('旧教材・新教材・収集結果の識別情報が違います')
    collected = json.loads(RESULT.read_text(encoding='utf8'))
    if not collected['dataset_ready'] or len(collected['episodes']) != 36 or not all(x['accepted'] for x in collected['episodes']):
        raise ValueError('新系列の品質が不成立です')
    rows = {split: [] for split in SPLITS}
    source_hashes = {}
    with ZipFile(OLD) as old, ZipFile(NEW) as new:
        manifest = json.loads(old.read('manifest.json'))
        if not manifest['completed'] or [len(manifest['items'][k]) for k in ('train', 'validation')] != [158, 40]:
            raise ValueError('旧教材の分割が違います')
        anchors = {}
        for old_split, target in (('train', 'train'), ('validation', 'development')):
            for original in manifest['items'][old_split]:
                source = original['source']
                if source not in anchors:
                    path = HERE / source / 'trace.json'
                    source_hashes[source] = sha(path)
                    anchors[source] = anchors_from_trace(json.loads(path.read_text(encoding='utf8')))
                precise = encode_context(original['state'], anchors[source])[16:30]
                rounded = coarse_context(original['state'], anchors[source])[16:30]
                if max(abs(a-b) for a, b in zip(rounded, original['context'], strict=True)) > TOLERANCE:
                    raise ValueError('旧教材と元の経路が違います')
                rows[target].append({**original, 'split': target, 'archive': 'old',
                                     'context_exact': precise, 'context_coarse': rounded})
        for episode in collected['episodes']:
            raw = new.read(f"episode-{episode['index']:02}/records.json")
            if hashlib.sha256(raw).hexdigest() != episode['records_sha256']:
                raise ValueError('新系列の記録が違います')
            for row in partition_records(episode, json.loads(raw)):
                rows[row['split']].append(row)
        archives = {'old': old, 'new': new}
        pixels = {split: set() for split in SPLITS}
        cache = {}
        for split in SPLITS:
            for row in rows[split]:
                fingerprints = {}
                for side in ('left', 'right'):
                    image = row['images'][side]
                    key = (row['archive'], image['file'])
                    if key not in cache:
                        raw = archives[key[0]].read(key[1])
                        cache[key] = (hashlib.sha256(raw).hexdigest(), pixel_sha(raw))
                    encoded, pixels_sha = cache[key]
                    if encoded != image['sha256']:
                        raise ValueError('画像ハッシュが違います')
                    fingerprints[side] = pixels_sha
                    pixels[split].add(pixels_sha)
                row['pixel_sha256'] = fingerprints
    overlaps = {a + '_' + b: len(pixels[a] & pixels[b]) for a, b in
                (('train', 'development'), ('train', 'heldout_home'), ('development', 'heldout_home'))}
    conflict_counts = {key: conflicts(rows['train'], key) for key in CONTEXTS}
    statistics_by_context = {}
    for key in CONTEXTS:
        stats = []
        for index in range(14):
            values = [row[key][index] for row in rows['train']]
            mean, std = statistics.mean(values), statistics.pstdev(values)
            if not math.isfinite(mean) or not math.isfinite(std) or std <= MIN_STD:
                raise ValueError('訓練側の正規化統計が不正です')
            stats.append({'mean': mean, 'std': std})
        statistics_by_context[key] = stats
    counts = {split: dict(Counter((row['archive'] + '_' + row['kind']) for row in rows[split])) for split in SPLITS}
    passed = all(rows[x] for x in SPLITS) and not any(overlaps.values()) and not any(conflict_counts.values())
    audit = {'completed': True, 'passed': passed, 'counts': counts, 'image_overlap': overlaps,
             'training_conflicts': conflict_counts, 'old_sha256': OLD_SHA, 'new_sha256': NEW_SHA,
             'source_trace_sha256': source_hashes, 'image_comparison': 'decoded_rgb_512x512',
             'scope': '開始姿勢・方向による分割。未使用姿勢は候補選択に使用しない。人物の独立性は別評価。'}
    output = {**audit, 'items': rows, 'training_context_stats': statistics_by_context}
    with OUT.open('x', encoding='utf8', newline='\n') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with AUDIT.open('x', encoding='utf8', newline='\n') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(audit, ensure_ascii=True))


if __name__ == '__main__':
    main()
