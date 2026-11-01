"""過去の自己経路を粗い関節角に丸め、退避用の状態へ組み込む。"""
import math

from retreat_path_context_20261101 import ARM_DIM, encode_context

ANCHOR_QUANTIZATION_RAD = .02


def coarse_anchors(anchors):
    result = {}
    for key in ('home', 'middle'):
        values = anchors[key]
        if len(values) != ARM_DIM or not all(math.isfinite(v) for v in values):
            raise ValueError('経路の要約が不正です')
        result[key] = [round(v / ANCHOR_QUANTIZATION_RAD) * ANCHOR_QUANTIZATION_RAD for v in values]
    return result


def coarse_context(state, anchors):
    return encode_context(state, coarse_anchors(anchors))
