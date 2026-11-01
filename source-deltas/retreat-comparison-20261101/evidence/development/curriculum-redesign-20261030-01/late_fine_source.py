"""画像目標への距離で固定接近から学習動作へ切り替える。"""
import math


def select_source(started,arm,position_valid,distance_m,switch_distance_m):
    if arm!='target':
        return started,'hold_or_retreat'
    if (not position_valid or distance_m is None or not math.isfinite(distance_m) or distance_m<0
        or not math.isfinite(switch_distance_m) or switch_distance_m<=0):
        raise ValueError('有効な画像目標または切替距離がありません')
    started=started or distance_m<=switch_distance_m
    return started, 'learned_fine' if started else 'fixed_coarse'
