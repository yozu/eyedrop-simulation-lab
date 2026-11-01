"""退避モデルの腕指令を制限し、指幅は固定制御から受け取る。"""
import math

ARM_JOINTS = 7


def retreat_action(gate, own_joints, model_action, fixed_gap, clamp):
    if len(own_joints) != ARM_JOINTS or not all(math.isfinite(x) for x in own_joints):
        raise ValueError('現在の関節角が不正です')
    if not math.isfinite(fixed_gap):
        raise ValueError('固定指幅が不正です')
    if gate.finished:
        target = gate.home
        source = 'retreat_home_hold'
    else:
        if model_action is None or len(model_action) < ARM_JOINTS or not all(
                math.isfinite(x) for x in model_action[:ARM_JOINTS]):
            raise ValueError('退避モデルの指令が不正です')
        target = model_action[:ARM_JOINTS]
        source = 'learned_retreat'
    joints = list(clamp(own_joints, target))
    if len(joints) != ARM_JOINTS or not all(math.isfinite(x) for x in joints):
        raise ValueError('制限後の関節指令が不正です')
    return [*joints, fixed_gap], source
