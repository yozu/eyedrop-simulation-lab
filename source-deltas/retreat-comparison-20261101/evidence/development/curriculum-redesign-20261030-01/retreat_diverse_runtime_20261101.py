"""観測済みの自己経路を学習時と同じ方法で状態へ組み込む。"""
from retreat_coarse_context_20261101 import coarse_context
from retreat_path_context_20261101 import encode_context


def encode_runtime_state(state, anchors, contract):
    mode = contract.get('mode')
    expected = {'version': 2, 'mode': mode, 'source': 'own_observed_approach',
                'anchor_quantization_rad': .02 if mode == 'coarse' else None, 'slots': [16, 30]}
    if mode not in ('coarse', 'exact') or contract != expected:
        raise ValueError('退避モデルの入力契約が違います')
    return coarse_context(state, anchors) if mode == 'coarse' else encode_context(state, anchors)
