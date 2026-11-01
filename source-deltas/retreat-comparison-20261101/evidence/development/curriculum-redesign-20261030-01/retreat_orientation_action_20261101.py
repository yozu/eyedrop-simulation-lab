"""別worldで退避指令の先端移動を保ち、容器の向きを補正する。"""
from learned_retreat_action_20261101 import retreat_action


def projected_retreat_action(gate, own_joints, model_action, fixed_gap, clamp, projection_world):
    action, source = retreat_action(gate, own_joints, model_action, fixed_gap, clamp)
    if gate.finished:
        return action, source
    projection_world.reset(action[:7], gap=fixed_gap)
    destination = projection_world.nozzle()
    projection_world.reset(own_joints, gap=fixed_gap)
    target = projection_world.ik(destination)
    return [*clamp(own_joints, target), fixed_gap], source
