"""固定退避と学習前後の退避を、開始区間と全動作に分けて照合する。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path('outputs/diagnostics/curriculum-redesign-20261030-01')
OUT = BASE / 'learned-retreat-comparison-audit-20261101-01.json'
FIXED = {'e-train04': 'dev-e-retreat-current-fixed-e-20261101-01',
         'd-val00': 'dev-e-recovery-half-gap-dev-d-candidate-20261101-01'}
REST_GAP_M = .0343
GAP_TOLERANCE_M = .0001
RETREAT_TOLERANCE_M = .001
FULL_STEPS = 1200


def load(folder, learned):
    result = json.loads((folder / 'result.json').read_text(encoding='utf8'))
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf8'))
    trace = json.loads((folder / 'trace.json').read_text(encoding='utf8'))
    if not result['completed'] or len(trace) != result['steps']:
        raise ValueError('操作数または終了状態が違います: ' + str(folder))
    hits = result['hits']
    checks = {'full_trial': result['steps'] == FULL_STEPS,
              'target_open_one_drop': result['drops'] == 1 and len(hits) == 1
              and hits[0]['result'] == 'target' and hits[0]['open'] == 1 and result['domain']['valid'],
              'noncontact_and_margin': not result['contacts'], 'not_aborted': not result['aborted'],
              'closed_safety': result['fluid']['passed'],
              'release': abs(result['final_gap_m'] - REST_GAP_M) < GAP_TOLERANCE_M,
              'retreat': result['phase'] == 'retreat' and result['retreat_error_m'] < RETREAT_TOLERANCE_M}
    if learned:
        checks['learned_retreat_used'] = result['action_source_counts']['learned_retreat'] > 0
        checks['home_hold_reached'] = result['action_source_counts']['retreat_home_hold'] > 0
    summary = {'checks': checks, 'all_checks_passed': all(checks.values()), 'steps': result['steps'],
               'drops': result['drops'], 'hits': hits, 'contacts': result['contacts'],
               'fluid': result['fluid'], 'domain': result['domain'], 'final_gap_m': result['final_gap_m'],
               'retreat_error_m': result['retreat_error_m'], 'action_source_counts': result['action_source_counts'],
               'seed': protocol['seed'], 'person': protocol['environment_person'],
               'main_model_sha256': protocol['model_sha256'], 'retreat_model_sha256': protocol.get('retreat_model_sha256'),
               'result_sha256': hashlib.sha256((folder / 'result.json').read_bytes()).hexdigest()}
    return trace, summary


def prefix(folders, traces):
    aligned = []
    for left, right in zip(*traces):
        if left['decision']['phase'] == 'retreat' or right['decision']['phase'] == 'retreat':
            break
        aligned.append((left, right))
    if not aligned:
        raise ValueError('退避前の共通区間がありません')
    image_differences = 0
    for step in range(len(aligned)):
        for side in ('left', 'right'):
            name = f'{side}-{step:03}.png'
            with Image.open(folders[0] / name) as a, Image.open(folders[1] / name) as b:
                if not np.array_equal(np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))):
                    image_differences += 1
    max_state = max(max(abs(a - b) for a, b in zip(x['state'], y['state'])) for x, y in aligned)
    max_action = max(max(abs(a - b) for a, b in zip(x['action'], y['action'])) for x, y in aligned)
    return {'steps': len(aligned), 'different_images': image_differences, 'max_state_difference': max_state,
            'max_action_difference': max_action,
            'exact': image_differences == 0 and max_state == 0 and max_action == 0}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    cases = {}
    for case, fixed_name in FIXED.items():
        folders = {'fixed': BASE / fixed_name,
                   'baseline': BASE / f'dev-retreat-learned-{case}-baseline-20261101-01',
                   'candidate': BASE / f'dev-retreat-learned-{case}-candidate-20261101-01'}
        results = {key: load(folder, key != 'fixed') for key, folder in folders.items()}
        fixed_protocol = results['fixed'][1]
        if any(results[key][1]['seed'] != fixed_protocol['seed'] or results[key][1]['person'] != fixed_protocol['person']
               or results[key][1]['main_model_sha256'] != fixed_protocol['main_model_sha256'] for key in ('baseline', 'candidate')):
            raise ValueError('人物・乱数・主モデルが一致しません: ' + case)
        prefixes = {key: prefix((folders['fixed'], folders[key]), (results['fixed'][0], results[key][0]))
                    for key in ('baseline', 'candidate')}
        cases[case] = {'fixed': results['fixed'][1], 'baseline': results['baseline'][1],
                       'candidate': results['candidate'][1], 'prefixes': prefixes}
    output = {'completed': True, 'cases': cases,
              'candidate_both_passed': all(value['fixed']['all_checks_passed'] and
                                           value['candidate']['all_checks_passed'] and
                                           value['prefixes']['baseline']['exact'] and
                                           value['prefixes']['candidate']['exact'] for value in cases.values()),
              'scope': '2開発人物の各1条件。指幅・移行・安全監視は固定、未知人物と突然閉眼は未評価。'}
    with OUT.open('x', encoding='utf8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
