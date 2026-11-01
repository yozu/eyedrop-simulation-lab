"""固定退避と経路要約A/Bの全動作・前半・実入力を監査する。"""
import json

from audit_learned_retreat_comparison_20261101 import BASE, FIXED, load, prefix
from retreat_coarse_context_20261101 import coarse_context
from retreat_context_features_20261101 import model_state
from retreat_path_context_20261101 import OnlinePathContext

OUT = BASE / 'retreat-context-comparison-audit-20261101-01.json'
TOLERANCE = 1e-9


def verify_trace_context(trace, mode, means):
    if not trace or mode not in ('control', 'candidate') or len(means) != 14:
        raise ValueError('監査対象または入力統計が不正です')
    history = OnlinePathContext(trace[0]['state'][:7])
    count, mismatches, maximum = 0, 0, 0.
    for row in trace:
        state = row['state']
        anchors = history.observe(row['decision']['arm'], state[:7])
        observed = row.get('retreat_input_state')
        if row['action_source'] == 'learned_retreat':
            count += 1
            if observed is None or len(observed) != 32 or anchors is None:
                mismatches += 1
                continue
            context = coarse_context(state, anchors)[16:30]
            expected = model_state(state, context, means, use_context=mode == 'candidate')
            difference = max(abs(a-b) for a, b in zip(observed, expected, strict=True))
            maximum = max(maximum, difference)
            mismatches += int(difference > TOLERANCE)
        elif observed is not None:
            mismatches += 1
    return {'learned_retreat_steps': count, 'mismatch_count': mismatches,
            'max_difference': maximum, 'passed': count > 0 and mismatches == 0}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    norms = {}
    for mode in ('control', 'candidate'):
        folder = BASE / f'dual-host-retreat-context-v2-{mode}-20261101-01/policy-final'
        norms[mode] = json.loads((folder/'normalization.json').read_text(encoding='utf8'))
    cases = {}
    for case, fixed_name in FIXED.items():
        folders = {'fixed': BASE / fixed_name,
                   'control': BASE / f'dev-retreat-context-control-{case}-20261101-01',
                   'candidate': BASE / f'dev-retreat-context-candidate-{case}-20261101-01'}
        results = {key: load(folder, key != 'fixed') for key, folder in folders.items()}
        fixed = results['fixed'][1]
        if any(results[key][1]['seed'] != fixed['seed'] or results[key][1]['person'] != fixed['person']
               or results[key][1]['main_model_sha256'] != fixed['main_model_sha256']
               for key in ('control', 'candidate')):
            raise ValueError('人物・乱数・主モデルが違います: ' + case)
        prefixes = {key: prefix((folders['fixed'], folders[key]), (results['fixed'][0], results[key][0]))
                    for key in ('control', 'candidate')}
        inputs = {key: verify_trace_context(results[key][0], key,
                  norms[key]['features']['observation.state']['mean'][16:30]) for key in ('control', 'candidate')}
        cases[case] = {'fixed': fixed, 'control': results['control'][1], 'candidate': results['candidate'][1],
                       'prefixes': prefixes, 'input_audit': inputs}
    passed = (set(cases) == {'e-train04', 'd-val00'} and all(
        item['fixed']['all_checks_passed'] and item['candidate']['all_checks_passed']
        and all(item['prefixes'][mode]['exact'] and item['input_audit'][mode]['passed']
                for mode in ('control', 'candidate')) for item in cases.values()))
    output = {'completed': True, 'cases': cases, 'candidate_both_passed': passed,
              'scope': '開発2人物各1条件。過去の自己経路要約は固定処理、腕は学習。未知人物・突然閉眼は未評価。'}
    with OUT.open('x', encoding='utf8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'completed': True, 'candidate_both_passed': passed}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
