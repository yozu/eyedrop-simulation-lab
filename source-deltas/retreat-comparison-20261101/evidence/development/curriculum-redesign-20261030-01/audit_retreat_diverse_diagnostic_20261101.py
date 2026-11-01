"""E・D教材の退避動作を固定・従来候補と同じ条件で照合する。"""
import json

from audit_learned_retreat_comparison_20261101 import BASE, FIXED, load, prefix
from audit_retreat_context_eval_20261101 import verify_trace_context

OUT = BASE / 'retreat-diverse-diagnostic-audit-20261101-01.json'


def main():
    if OUT.exists():
        raise FileExistsError('既存の監査を上書きしません')
    norm = json.loads((BASE / 'dual-host-retreat-diverse-coarse-20261101-01/policy-final/normalization.json').read_text(encoding='utf8'))
    means = norm['features']['observation.state']['mean'][16:30]
    cases = {}
    for case, fixed_name in FIXED.items():
        folders = {'fixed': BASE / fixed_name,
                   'previous': BASE / f'dev-retreat-context-candidate-{case}-20261101-01',
                   'candidate': BASE / f'dev-retreat-diverse-coarse-diagnostic-{case}-20261101-01'}
        results = {key: load(folder, key != 'fixed') for key, folder in folders.items()}
        fixed = results['fixed'][1]
        if any(results[key][1]['seed'] != fixed['seed'] or results[key][1]['person'] != fixed['person']
               or results[key][1]['main_model_sha256'] != fixed['main_model_sha256']
               for key in ('previous', 'candidate')):
            raise ValueError('人物・乱数・主モデルが違います: ' + case)
        prefixes = {key: prefix((folders['fixed'], folders[key]), (results['fixed'][0], results[key][0]))
                    for key in ('previous', 'candidate')}
        inputs = verify_trace_context(results['candidate'][0], 'candidate', means)
        cases[case] = {'fixed': fixed, 'previous': results['previous'][1], 'candidate': results['candidate'][1],
                       'prefixes': prefixes, 'input_audit': inputs}
    passed = (set(cases) == {'e-train04', 'd-val00'} and all(
        item['fixed']['all_checks_passed'] and item['candidate']['all_checks_passed']
        and all(item['prefixes'][mode]['exact'] for mode in ('previous', 'candidate'))
        and item['input_audit']['passed'] for item in cases.values()))
    result = {'completed': True, 'cases': cases, 'candidate_both_passed': passed, 'adopted': False, 'static_gate_passed': False,
              'scope': '開発2人物各1条件。突然閉眼と学習未使用人物は未評価。'}
    with OUT.open('x', encoding='utf8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps({'completed': True, 'candidate_both_passed': passed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
