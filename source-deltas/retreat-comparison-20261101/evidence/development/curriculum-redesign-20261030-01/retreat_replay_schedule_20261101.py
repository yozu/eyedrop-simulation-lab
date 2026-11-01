"""補正2例と既存の移動・停止各1例を同じ更新で巡回する。"""
import random


def schedule(rows, updates, seed):
    if len(rows) != 88 or updates <= 0 or any(row['kind'] != 'moving' for row in rows[80:]):
        raise ValueError('教材の構成が想定と異なります')
    moving = [i for i, row in enumerate(rows[:80]) if row['kind'] == 'moving']
    idle = [i for i, row in enumerate(rows[:80]) if row['kind'] == 'idle']
    if len(moving) != 64 or len(idle) != 16:
        raise ValueError('元教材の移動・停止の件数が異なります')
    rng = random.Random(seed)
    rng.shuffle(moving)
    rng.shuffle(idle)
    return [(80+i%4, 87-i%4, moving[i%len(moving)], idle[i%len(idle)])
            for i in range(updates)]
