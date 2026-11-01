"""実姿勢が教材の各関節範囲と姿勢の組合せに含まれるかを診断する。"""
import math

JOINTS = 7


def summarize(samples, observed):
    if (not samples or not observed or any(len(row) != JOINTS or not all(math.isfinite(x) for x in row)
                                           for row in [*samples, *observed])):
        raise ValueError('姿勢の次元または値が不正です')
    low = [min(row[j] for row in samples) for j in range(JOINTS)]
    high = [max(row[j] for row in samples) for j in range(JOINTS)]
    records = []
    for state in observed:
        outside = [j for j in range(JOINTS) if state[j] < low[j] or state[j] > high[j]]
        nearest = min(max(abs(a-b) for a, b in zip(state, sample, strict=True)) for sample in samples)
        records.append({'out_of_range_joints': outside, 'nearest_linf_rad': nearest})
    return {'steps': len(records), 'out_of_range_steps': sum(bool(r['out_of_range_joints']) for r in records),
            'minimum_nearest_linf_rad': min(r['nearest_linf_rad'] for r in records),
            'maximum_nearest_linf_rad': max(r['nearest_linf_rad'] for r in records),
            'records': records}
