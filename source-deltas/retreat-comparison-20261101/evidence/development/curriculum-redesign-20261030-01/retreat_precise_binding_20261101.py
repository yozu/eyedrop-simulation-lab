"""開始姿勢の表現に対応する正規化の変更範囲を検査する。"""
import copy
import math

START, END = 16, 30


def validate_context_change(original, changed, *, mode):
    if mode not in ('coarse', 'exact'):
        raise ValueError('比較モードが不正です')
    contract = {'version': 2, 'mode': mode, 'source': 'own_observed_approach',
                'anchor_quantization_rad': .02 if mode == 'coarse' else None, 'slots': [START, END]}
    if changed.get('retreat_context_contract') != contract:
        raise ValueError('退避入力の契約が不一致です')
    expected = copy.deepcopy(original)
    for key in ('mean', 'std'):
        before = expected['features']['observation.state'][key]
        after = changed['features']['observation.state'][key]
        if len(before) != 32 or len(after) != 32:
            raise ValueError('状態統計の次元が違います')
        values = after[START:END]
        if not all(math.isfinite(v) for v in values) or (key == 'std' and not all(v > 1e-6 for v in values)):
            raise ValueError('経路入力の統計が不正です')
        before[START:END] = values
    expected['retreat_context_contract'] = contract
    if changed != expected:
        raise ValueError('経路入力以外の設定が変更されています')


def rebind(policy, original, changed, *, mode):
    from relative_joint_action import bind_policy, validate_policy_binding
    validate_policy_binding(policy, original)
    validate_context_change(original, changed, mode=mode)
    bind_policy(policy, changed)
    validate_policy_binding(policy, changed)
