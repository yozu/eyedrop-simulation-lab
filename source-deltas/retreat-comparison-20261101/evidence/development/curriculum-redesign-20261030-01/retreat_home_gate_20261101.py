"""退避先を観測した自分の開始関節角で判定し、帰還後に指を全開にする。"""
import math

from eyedrop.panda_fixed_control import RETREAT_GAP_M
from hybrid_action import WAYPOINT_TOLERANCE_RAD

ARM_JOINTS = 7


class RetreatHomeGate:
    def __init__(self, home_joints):
        if len(home_joints) != ARM_JOINTS or not all(math.isfinite(x) for x in home_joints):
            raise ValueError('開始時の関節角が不正です')
        self.home = tuple(home_joints)
        self.retreat_gap = None
        self.finished = False

    def adjust(self, decision, observed_gap, own_joints):
        if decision['phase'] != 'retreat' or self.finished:
            return decision
        if not math.isfinite(observed_gap) or observed_gap < RETREAT_GAP_M:
            raise ValueError('解除済みの指幅を確認できません')
        if len(own_joints) != ARM_JOINTS or not all(math.isfinite(x) for x in own_joints):
            raise ValueError('現在の関節角が不正です')
        if max(abs(a - b) for a, b in zip(own_joints, self.home, strict=True)) <= WAYPOINT_TOLERANCE_RAD:
            self.finished = True
            return decision
        self.retreat_gap = max(self.retreat_gap or observed_gap, observed_gap)
        return {**decision, 'gap': min(decision['gap'], self.retreat_gap)}
