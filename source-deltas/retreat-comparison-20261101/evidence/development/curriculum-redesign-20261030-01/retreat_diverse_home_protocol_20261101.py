"""開始姿勢を分けた退避教師の成立確認条件。"""
import math

TRAIN_OFFSETS = ((-.016, 0.), (-.008, .008), (.008, -.008), (.016, 0.))
VALIDATION_OFFSETS = ((-.004, -.004), (.004, .004))
PROBE_OFFSETS = ((.03, 0.), (0., .03), (-.06, .06))
PERSONS = ('vroid-beta-e', 'vroid-beta-d')
JOINT_INDICES = (0, 2)
HOME_TOLERANCE_RAD = .001
RELEASED_GAP_M = .0187
MAX_OPERATIONS = 100
ACTION_HZ = 20


def offset_joints(joints, pair):
    if len(joints) != 7 or len(pair) != 2 or not all(math.isfinite(v) for v in [*joints, *pair]):
        raise ValueError('関節姿勢または変位が不正です')
    values = list(joints)
    for index, offset in zip(JOINT_INDICES, pair, strict=True):
        values[index] += offset
    return values


def cases():
    if set(TRAIN_OFFSETS) & set(VALIDATION_OFFSETS):
        raise ValueError('開始姿勢の訓練・検査が重複します')
    return [{'person': person, 'split': split, 'home_offset': list(home), 'probe_offset': list(probe)}
            for person in PERSONS
            for split, homes in (('train', TRAIN_OFFSETS), ('validation', VALIDATION_OFFSETS))
            for home in homes for probe in PROBE_OFFSETS]


def reached(current, target):
    if len(current) != 7 or len(target) != 7 or not all(math.isfinite(v) for v in [*current, *target]):
        raise ValueError('現在または開始姿勢が不正です')
    return max(abs(a-b) for a, b in zip(current, target, strict=True)) <= HOME_TOLERANCE_RAD
