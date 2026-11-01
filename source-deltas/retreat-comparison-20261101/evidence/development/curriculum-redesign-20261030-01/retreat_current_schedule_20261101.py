"""移動と停止を4対1にし、完全一致する画像組を均等に巡回する。"""
from collections import defaultdict
import random

UPDATES = 240
MOVING_PER_CYCLE = 4


def make_schedule(rows, seed):
    groups = {'moving': defaultdict(list), 'idle': defaultdict(list)}
    for index, row in enumerate(rows):
        kind = row['kind']
        if kind not in groups:
            raise ValueError('未定義の退避区分です')
        key = tuple(row['images'][side]['sha256'] for side in ('left', 'right'))
        groups[kind][key].append(index)
    if not all(groups.values()):
        raise ValueError('移動と停止の両方が必要です')
    rng = random.Random(seed)
    keys = {kind: sorted(bucket) for kind, bucket in groups.items()}
    for group in keys.values():
        rng.shuffle(group)
    seen = defaultdict(int)
    counts = {'moving': 0, 'idle': 0}
    schedule = []
    for update in range(UPDATES):
        kind = 'idle' if update % (MOVING_PER_CYCLE + 1) == MOVING_PER_CYCLE else 'moving'
        key = keys[kind][counts[kind] % len(keys[kind])]
        options = groups[kind][key]
        index = options[seen[(kind, key)] % len(options)]
        seen[(kind, key)] += 1
        counts[kind] += 1
        schedule.append((kind, index))
    return schedule
