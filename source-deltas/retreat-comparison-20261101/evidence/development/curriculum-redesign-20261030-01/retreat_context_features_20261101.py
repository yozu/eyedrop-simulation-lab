"""退避モデル用の状態欄を、経路要約または訓練平均で置き換える。"""
import math

STATE_DIM = 32
CONTEXT_DIM = 14
CONTEXT_START = 16
CONTEXT_END = CONTEXT_START + CONTEXT_DIM


def model_state(state, context, means, *, use_context):
    if (len(state) != STATE_DIM or len(context) != CONTEXT_DIM or len(means) != CONTEXT_DIM
            or not all(math.isfinite(v) for v in [*state, *context, *means])):
        raise ValueError('退避モデルの入力次元または値が不正です')
    values = context if use_context else means
    return list(state[:CONTEXT_START]) + list(values) + list(state[CONTEXT_END:])
