"""開始姿勢と方向で訓練・開発・未使用姿勢を分割する。"""
from retreat_diverse_home_protocol_20261101 import PROBE_OFFSETS, TRAIN_OFFSETS, VALIDATION_OFFSETS


def split_episode(episode):
    home = tuple(episode['home_offset'])
    probe = tuple(episode['probe_offset'])
    if probe not in PROBE_OFFSETS:
        raise ValueError('未登録の移動方向です')
    if episode['split'] == 'validation' and home in VALIDATION_OFFSETS:
        return 'heldout_home'
    if episode['split'] == 'train' and home in TRAIN_OFFSETS:
        return 'development' if probe == PROBE_OFFSETS[2] else 'train'
    raise ValueError('開始姿勢と分割が一致しません')
