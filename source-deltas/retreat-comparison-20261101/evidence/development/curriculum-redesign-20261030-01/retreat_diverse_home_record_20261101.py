"""実行前の自己状態と教師指令を、同時刻の画像と組み合わせる。"""
import math

from retreat_path_context_20261101 import encode_context

OWN_JOINTS = 7
STATE_SIZE = 32
ACTION_SIZE = 8
CONTEXT_START = 16
CONTEXT_END = 30
STATE_TOLERANCE = 1e-9


def make_record(step, time_s, state, observed_joints, anchors, action, images):
    if (type(step) is not int or step < 0 or not math.isfinite(time_s) or time_s < 0
            or len(state) != STATE_SIZE or len(observed_joints) != OWN_JOINTS or len(action) != ACTION_SIZE
            or not all(math.isfinite(x) for x in [*state, *observed_joints, *action])):
        raise ValueError('記録する状態または指令が不正です')
    if max(abs(a-b) for a, b in zip(state[:OWN_JOINTS], observed_joints, strict=True)) > STATE_TOLERANCE:
        raise ValueError('操作前の自己状態と実測姿勢が違います')
    if set(images) != {'left', 'right'} or any(set(image) != {'file', 'sha256'} for image in images.values()):
        raise ValueError('左右画像の識別情報が不正です')
    context = encode_context(state, anchors)[CONTEXT_START:CONTEXT_END]
    return {'step': step, 'time_s': time_s, 'state': list(state), 'context': context,
            'action': list(action), 'images': images}
