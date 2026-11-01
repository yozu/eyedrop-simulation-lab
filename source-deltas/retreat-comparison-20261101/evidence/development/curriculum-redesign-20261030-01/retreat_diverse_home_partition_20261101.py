"""連続系列の役割と、自己経路の2種類の表現を決める。"""
from retreat_coarse_context_20261101 import coarse_context
from retreat_diverse_home_split_20261101 import split_episode
from retreat_path_context_20261101 import encode_context

HOLD_STEPS = 3


def partition_records(episode, records):
    if not episode['accepted'] or len(records) != episode['records'] or len(records) <= HOLD_STEPS:
        raise ValueError('成立した連続系列ではありません')
    split = split_episode(episode)
    anchors = episode['anchors']
    rows = []
    for index, record in enumerate(records):
        exact = encode_context(record['state'], anchors)[16:30]
        if exact != record['context']:
            raise ValueError('保存された精密経路が一致しません')
        rows.append({**record, 'split': split, 'kind': 'idle' if index >= len(records)-HOLD_STEPS else 'moving',
                     'context_exact': exact, 'context_coarse': coarse_context(record['state'], anchors)[16:30],
                     'archive': 'new', 'episode_index': episode['index'], 'person': episode['person'],
                     'home_offset': episode['home_offset'], 'probe_offset': episode['probe_offset']})
    return rows
