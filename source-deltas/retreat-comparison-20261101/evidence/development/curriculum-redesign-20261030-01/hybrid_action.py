"""学習した接近指令と固定の押圧・退避を合成する。"""
import numpy as np

PATH_SPACING_RAD = .001
WAYPOINT_TOLERANCE_RAD = .001

def combine_action(decision, own_joints, learned_joints, path, held):
    own = np.asarray(own_joints, dtype=float)
    if own.shape != (7,) or not np.isfinite(own).all():
        raise ValueError('実測関節が不正です')
    if decision['arm'] == 'target':
        learned = np.asarray(learned_joints, dtype=float)
        if learned.shape != (7,) or not np.isfinite(learned).all():
            raise ValueError('学習した関節指令が不正です')
        if np.max(np.abs(own-np.asarray(path[-1]))) > PATH_SPACING_RAD:
            path.append(own.tolist())
        joints = learned.tolist()
        held = None
    elif decision['arm'] == 'hold':
        if held is None:
            held = own.tolist()
        joints = held
    elif decision['arm'] == 'retreat':
        if len(path)>1 and np.max(np.abs(own-np.asarray(path[-1]))) <= WAYPOINT_TOLERANCE_RAD:
            path.pop()
        joints = path[-1]
    else:
        raise ValueError('未知の腕の指令です')
    return [*joints, decision['gap']], held
