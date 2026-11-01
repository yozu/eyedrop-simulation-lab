"""事前固定した4区分の開発誤差から候補を選ぶ。"""
import math

GROUPS = ('old_moving', 'old_idle', 'new_moving', 'new_idle')


def choose(means):
    if set(means) != {'initial', 'coarse', 'exact'}:
        raise ValueError('比較モデルが不正です')
    for values in means.values():
        if set(values) != set(GROUPS) or not all(math.isfinite(x) and x >= 0 for x in values.values()):
            raise ValueError('比較区分または誤差が不正です')
    candidates = [name for name in ('coarse', 'exact')
                  if all(means[name][group] <= means['initial'][group] for group in GROUPS)]
    return min(candidates, key=lambda name: means[name]['new_moving']) if candidates else None
