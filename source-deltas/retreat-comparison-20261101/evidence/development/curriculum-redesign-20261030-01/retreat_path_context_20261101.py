"""退避開始までの自己姿勢を要約し、退避モデルの状態へ組み込む。"""
import math

from hybrid_action import PATH_SPACING_RAD

ARM_DIM = 7
STATE_DIM = 32
CONTEXT_START = 16
MIDDLE_START = CONTEXT_START + ARM_DIM
CONTEXT_END = MIDDLE_START + ARM_DIM


class OnlinePathContext:
    def __init__(self, initial):
        if len(initial) != ARM_DIM or not all(math.isfinite(v) for v in initial):
            raise ValueError('開始姿勢が不正です')
        self.path = [list(initial)]
        self.anchors = None

    def observe(self, arm, own):
        if len(own) != ARM_DIM or not all(math.isfinite(v) for v in own):
            raise ValueError('腕の姿勢が不正です')
        if arm == 'target':
            if self.anchors is not None:
                raise ValueError('退避開始後に接近経路を変更できません')
            if max(abs(a-b) for a, b in zip(own, self.path[-1], strict=True)) > PATH_SPACING_RAD:
                self.path.append(list(own))
        elif arm == 'retreat' and self.anchors is None:
            if len(self.path) < 3:
                raise ValueError('接近経路が不足しています')
            self.anchors = {'home': self.path[0], 'middle': self.path[len(self.path)//2],
                            'approach_points': len(self.path)}
        elif arm not in ('target', 'hold', 'retreat'):
            raise ValueError('動作段階が不正です')
        return self.anchors


def anchors_from_trace(trace):
    if not trace or len(trace[0]['state']) < ARM_DIM:
        raise ValueError('開始姿勢がありません')
    path = [list(trace[0]['state'][:ARM_DIM])]
    for index, row in enumerate(trace):
        arm = row['decision']['arm']
        own = list(row['state'][:ARM_DIM])
        if len(own) != ARM_DIM or not all(math.isfinite(v) for v in own):
            raise ValueError('腕の姿勢が不正です')
        if arm == 'retreat':
            if len(path) < 3:
                raise ValueError('接近経路が不足しています')
            return {'home': path[0], 'middle': path[len(path)//2],
                    'approach_points': len(path), 'retreat_start_step': index}
        if arm == 'target' and max(abs(a-b) for a, b in zip(own, path[-1], strict=True)) > PATH_SPACING_RAD:
            path.append(own)
    raise ValueError('退避開始が記録されていません')


def encode_context(state, anchors):
    if len(state) != STATE_DIM or not all(math.isfinite(v) for v in state):
        raise ValueError('自己状態が不正です')
    home, middle = anchors['home'], anchors['middle']
    if any(len(values) != ARM_DIM or not all(math.isfinite(v) for v in values) for values in (home, middle)):
        raise ValueError('経路の要約が不正です')
    own = state[:ARM_DIM]
    return (list(state[:CONTEXT_START]) + [a-b for a, b in zip(home, own, strict=True)]
            + [a-b for a, b in zip(middle, own, strict=True)] + list(state[CONTEXT_END:]))
