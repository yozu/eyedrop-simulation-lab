"""新旧・移動停止を均等にし、各系列を順に使う。"""
from collections import defaultdict
import random

KEYS = tuple((archive, kind) for archive in ('old', 'new') for kind in ('moving', 'idle'))


def make_schedule(rows, updates, seed):
    if updates <= 0:
        raise ValueError('更新数が不正です')
    groups = {key: defaultdict(list) for key in KEYS}
    for index, row in enumerate(rows):
        key = row['archive'], row['kind']
        if key not in groups:
            raise ValueError('教材の区分が不正です')
        episode = row['route_group'] if row['archive'] == 'old' else row['episode_index']
        groups[key][episode].append(index)
    if not all(groups.values()):
        raise ValueError('新旧・移動停止の教材が揃っていません')
    rng = random.Random(seed)
    order = {key: sorted(value) for key, value in groups.items()}
    for value in order.values():
        rng.shuffle(value)
    counts = defaultdict(int)
    schedule = []
    for update in range(updates):
        batch = []
        for key in KEYS:
            episode = order[key][update % len(order[key])]
            indices = groups[key][episode]
            batch.append(indices[counts[(key, episode)] % len(indices)])
            counts[(key, episode)] += 1
        schedule.append(tuple(batch))
    return schedule
