"""開始姿勢を分割し、連続移動の画像と退避教師指令を保存する。"""
import argparse
import base64
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from PIL import Image

from retreat_diverse_home_protocol_20261101 import (ACTION_HZ, MAX_OPERATIONS, PERSONS, RELEASED_GAP_M,
                                                     cases, offset_joints, reached)
from retreat_diverse_home_record_20261101 import make_record
from retreat_path_context_20261101 import OnlinePathContext, encode_context

BASE = Path('outputs/diagnostics/curriculum-redesign-20261030-01')
DATA = Path('outputs/learning/panda-synchronized-data-01')
PREFLIGHT = BASE / 'retreat-diverse-home-preflight-20261101-01/result.json'
PREFLIGHT_SHA = '04076f9d15bc3ac493d0a218568a9aa14df7c3bb2a2e1418de664844e12667e7'
NORMALIZATION = BASE / 'dual-host-retreat-onpolicy-candidate-20261101-01/policy-final/normalization.json'
NORMALIZATION_SHA = '80f84cecc060a4c186c5b5835bf453282f01c166f7226abfb75cd7f4bfc2dc28'
FIXED = ('dev-e-retreat-current-fixed-e-20261101-01', 'dev-e-recovery-half-gap-dev-d-candidate-20261101-01')
OUT = BASE / 'retreat-diverse-home-data-20261101-01'
CASE = 'original-train-00'
HOLD_STEPS = 3
MIN_ACCEPTED_PER_HOME = 2
FIXED_STEPS = 1200
AUDIT_TOLERANCE = 1e-8
IMAGE_SIZE = (512, 512)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def pixel_sha(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        if image.size != IMAGE_SIZE:
            raise ValueError('画像サイズが違います: ' + str(path))
        return hashlib.sha256(image.tobytes()).hexdigest()


def main():
    global OUT
    from eyedrop.drop_kinematics import MOVING_NOZZLE
    from eyedrop.panda_world import PandaWorld, synchronized_joint_target
    from eyedrop.smolvla_observation import history_from_normalization
    from panda_rpc import PandaBrowser
    from resumable_job import atomic_json

    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        OUT = BASE / 'retreat-diverse-home-data-smoke-20261101-01'
    if OUT.exists():
        raise FileExistsError('既存教材を上書きしません')
    if sha(PREFLIGHT) != PREFLIGHT_SHA or sha(NORMALIZATION) != NORMALIZATION_SHA:
        raise ValueError('成立確認または正規化が違います')
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf8'))
    specification = cases()
    if (not preflight['completed'] or not preflight['preflight_passed'] or len(preflight['rows']) != len(specification)
            or not all(row['accepted'] for row in preflight['rows'])):
        raise ValueError('成立確認が完了していません')
    for item, checked in zip(specification, preflight['rows'], strict=True):
        if any(item[key] != checked[key] for key in ('person', 'split', 'home_offset', 'probe_offset')):
            raise ValueError('成立確認と記録条件が違います')
    planned = specification[:1] if args.smoke else specification
    protocol = json.loads((DATA / 'protocol.json').read_text())
    case = next(x for x in protocol['train'] if x['id'] == CASE)
    source = Path(json.loads((DATA / CASE / 'result.json').read_text())['original_source'])
    meta = json.loads((source.parent / 'protocol.json').read_text())
    if meta['oracle_diagnostic'] or meta['drop_model'] != MOVING_NOZZLE:
        raise ValueError('元の環境条件が違います')
    norm = json.loads(NORMALIZATION.read_text())
    OUT.mkdir(parents=True)
    atomic_json(OUT / 'protocol.json', {'case': CASE, 'conditions': planned, 'preflight_sha256': PREFLIGHT_SHA,
        'smoke': args.smoke,
        'normalization_sha256': NORMALIZATION_SHA, 'gap_m': RELEASED_GAP_M, 'max_operations': MAX_OPERATIONS,
        'hold_steps': HOLD_STEPS, 'context': 'unrounded_observed_home_and_observed_approach_midpoint',
        'scope': '別worldでの人工的な接近と復帰。開始時に容器を初期化し、各系列の動きは連続。点眼全体ではない。'})
    episodes, image_paths, limits_by_person = [], [], {}
    for person in PERSONS:
        if not any(item['person'] == person for item in planned):
            continue
        folder = OUT / person
        folder.mkdir()
        world = PandaWorld(folder, rest_gap=meta['rest_gap'], bottle_input=meta['bottle_input'],
                           finger_release_speed_m_s=meta['finger_release_speed_m_s'],
                           finger_press_speed_m_s=meta['finger_press_speed_m_s'],
                           joint_velocity_cap=meta['joint_velocity_cap'], drop_model=MOVING_NOZZLE)
        browser = None
        try:
            browser = PandaBrowser(folder / 'browser.log', person=person, hybrid_rgb=True)
            browser.call('setup', shapes=world.drawing())
            browser.call('reset', depth=case['depth_m'], phase=case['blink_phase_s'])
            world.reset(meta['initial_joints'])
            world.add_affine_geometry(browser.call('geometry'), browser.call('blinkGeometry'))
            limits = [(world.joint_info[i][8], world.joint_info[i][9]) for i in range(7)]
            limits_by_person[person] = limits
            for item in (row for row in planned if row['person'] == person):
                index = len(episodes)
                episode_dir = OUT / f'episode-{index:02}'
                episode_dir.mkdir()
                atomic_json(OUT / 'progress.json', {'completed_episodes': index, 'total_episodes': len(planned),
                                                   'current': item, 'phase': 'approach'})
                planned_home = offset_joints(meta['initial_joints'], item['home_offset'])
                target = offset_joints(planned_home, item['probe_offset'])
                if any(not lower <= q <= upper for values in (planned_home, target)
                       for q, (lower, upper) in zip(values, limits, strict=True)):
                    raise ValueError('姿勢が関節範囲外です')
                world.reset(planned_home, gap=RELEASED_GAP_M)
                initial = world.own_state()
                home = initial[:7]
                if not reached(home, planned_home):
                    raise ValueError('開始姿勢が一致しません')
                history = history_from_normalization(norm, meta['rest_gap'])
                history.reset(initial, RELEASED_GAP_M)
                path = OnlinePathContext(home)
                contacts = world.contacts(swept=False)
                action_count = 0
                while not contacts and not reached(world.own_state()[:7], target) and action_count < MAX_OPERATIONS:
                    own = world.own_state()
                    path.observe('target', own[:7])
                    joints = synchronized_joint_target(own[:7], target, meta['joint_velocity_cap'] / ACTION_HZ, limits)
                    action = [*joints, RELEASED_GAP_M]
                    outcome = world.step(action)
                    contacts = outcome['contacts']
                    action_count += 1
                    history.append(action_count, world.own_state(), action[7])
                approached = reached(world.own_state()[:7], target)
                records = []
                anchors = None
                if approached and not contacts:
                    path.observe('target', world.own_state()[:7])
                    anchors = path.observe('retreat', world.own_state()[:7])

                    def save_step(action):
                        frame = browser.call('render', state={'time': world.time, 'links': world.links(),
                                             'eye_condition': 'natural'}, capture=True)
                        images = {}
                        for side in ('left', 'right'):
                            raw = base64.b64decode(frame[side], validate=True)
                            path_name = f'{side}-{len(records):03}.png'
                            file = episode_dir / path_name
                            with file.open('xb') as f:
                                f.write(raw)
                            images[side] = {'file': f'{episode_dir.name}/{path_name}',
                                            'sha256': hashlib.sha256(raw).hexdigest()}
                            image_paths.append((item['split'], file, images[side]['sha256']))
                        records.append(make_record(action_count, world.time, history.observation(),
                                                   world.own_state()[:7], anchors, action, images))

                    returns = 0
                    while not contacts and not reached(world.own_state()[:7], home) and returns < MAX_OPERATIONS:
                        own = world.own_state()
                        joints = synchronized_joint_target(own[:7], home, meta['joint_velocity_cap'] / ACTION_HZ, limits)
                        action = [*joints, RELEASED_GAP_M]
                        save_step(action)
                        outcome = world.step(action)
                        contacts = outcome['contacts']
                        action_count += 1
                        returns += 1
                        history.append(action_count, world.own_state(), action[7])
                    returned = reached(world.own_state()[:7], home)
                    for _ in range(HOLD_STEPS if returned and not contacts else 0):
                        own = world.own_state()
                        joints = synchronized_joint_target(own[:7], home, meta['joint_velocity_cap'] / ACTION_HZ, limits)
                        action = [*joints, RELEASED_GAP_M]
                        save_step(action)
                        outcome = world.step(action)
                        contacts = outcome['contacts']
                        action_count += 1
                        history.append(action_count, world.own_state(), action[7])
                        if contacts:
                            break
                else:
                    returned = False
                returned = bool(returned and reached(world.own_state()[:7], home))
                accepted = bool(approached and returned and not contacts and not world.drops and len(records) >= HOLD_STEPS + 1)
                atomic_json(episode_dir / 'records.json', records)
                episodes.append({**item, 'index': index, 'approached': approached, 'returned': returned,
                                 'contacts': contacts, 'drops': len(world.drops), 'records': len(records),
                                 'observed_home': home, 'observed_approach': path.path, 'anchors': anchors,
                                 'accepted': accepted, 'records_sha256': sha(episode_dir / 'records.json')})
                atomic_json(OUT / 'progress.json', {'completed_episodes': len(episodes), 'total_episodes': len(planned),
                                                   'saved_images': len(image_paths), 'last_accepted': accepted})
        finally:
            if browser is not None:
                browser.close()
            world.close()
    groups = defaultdict(lambda: {'count': 0, 'accepted': 0})
    for row in episodes:
        key = (row['person'], row['split'], tuple(row['home_offset']))
        groups[key]['count'] += 1
        groups[key]['accepted'] += int(row['accepted'])
    summary = [{'person': person, 'split': split, 'home_offset': list(home), **value,
                'passed': value['count'] == 3 and value['accepted'] >= MIN_ACCEPTED_PER_HOME}
               for (person, split, home), value in groups.items()]
    hashes = {'train': set(), 'validation': set()}
    for split, path, expected in image_paths:
        if sha(path) != expected:
            raise ValueError('保存画像が違います')
        hashes[split].add(pixel_sha(path))
    max_context_error = max_action_error = 0.
    for episode in episodes:
        records_path = OUT / f"episode-{episode['index']:02}" / 'records.json'
        if sha(records_path) != episode['records_sha256']:
            raise ValueError('記録が変更されました')
        records = json.loads(records_path.read_text(encoding='utf8'))
        if len(records) != episode['records']:
            raise ValueError('記録数が一致しません')
        anchors = episode['anchors']
        if records:
            approach = episode['observed_approach']
            if (anchors is None or len(approach) < 3 or anchors['home'] != episode['observed_home']
                    or anchors['home'] != approach[0] or anchors['middle'] != approach[len(approach)//2]):
                raise ValueError('過去の自己経路と退避入力が一致しません')
        previous = None
        for row in records:
            expected_context = encode_context(row['state'], anchors)[16:30]
            max_context_error = max(max_context_error, *(abs(a-b) for a, b in zip(row['context'], expected_context, strict=True)))
            target = synchronized_joint_target(row['state'][:7], episode['observed_home'],
                                               meta['joint_velocity_cap'] / ACTION_HZ, limits_by_person[episode['person']])
            max_action_error = max(max_action_error, *(abs(a-b) for a, b in zip(row['action'][:7], target, strict=True)),
                                   abs(row['action'][7] - RELEASED_GAP_M))
            if previous is not None and (row['step'] != previous['step'] + 1
                    or not math.isclose(row['time_s'] - previous['time_s'], 1/ACTION_HZ, abs_tol=AUDIT_TOLERANCE)):
                raise ValueError('状態と指令の時刻が連続していません')
            previous = row
    if max_context_error > AUDIT_TOLERANCE or max_action_error > AUDIT_TOLERANCE:
        raise ValueError('自己経路または教師指令が一致しません')
    fixed_hashes = set()
    for folder in FIXED:
        for index in range(FIXED_STEPS):
            for side in ('left', 'right'):
                fixed_hashes.add(pixel_sha(BASE / folder / f'{side}-{index:03}.png'))
    split_overlap = len(hashes['train'] & hashes['validation'])
    fixed_overlap = len((hashes['train'] | hashes['validation']) & fixed_hashes)
    group_passed = ((len(episodes) == 1 and episodes[0]['accepted']) if args.smoke
                    else (len(summary) == 12 and all(row['passed'] for row in summary)))
    passed = len(episodes) == len(planned) and group_passed and split_overlap == 0 and fixed_overlap == 0
    result = {'completed': len(episodes) == len(planned), 'smoke': args.smoke,
              'smoke_passed': passed if args.smoke else None, 'dataset_ready': passed if not args.smoke else False,
              'episodes': episodes,
              'groups': summary, 'images': len(image_paths), 'train_validation_image_overlap': split_overlap,
              'fixed_evaluation_image_overlap': fixed_overlap, 'preflight_sha256': PREFLIGHT_SHA,
              'max_context_error': max_context_error, 'max_action_error': max_action_error,
              'image_overlap_method': 'decoded_rgb_pixels_512x512',
              'scope': '開発2人物の人工的な接近・復帰。系列開始時に容器初期化。独立人物・点眼全体ではない。'}
    atomic_json(OUT / 'result.json', result)
    if passed and not args.smoke:
        temporary = OUT / 'dataset.tmp'
        final = OUT / 'dataset.zip'
        with ZipFile(temporary, 'x', compression=ZIP_STORED) as archive:
            archive.write(OUT / 'protocol.json', 'protocol.json')
            archive.write(OUT / 'result.json', 'result.json')
            for episode in episodes:
                directory = OUT / f"episode-{episode['index']:02}"
                archive.write(directory / 'records.json', f"{directory.name}/records.json")
            for _, path, _ in image_paths:
                archive.write(path, f'{path.parent.name}/{path.name}')
        os.replace(temporary, final)
        atomic_json(OUT / 'dataset-receipt.json', {'sha256': sha(final), 'bytes': final.stat().st_size,
                                                  'images': len(image_paths), 'episodes': len(episodes)})
    print(json.dumps({'completed': result['completed'], 'smoke': args.smoke, 'smoke_passed': result['smoke_passed'],
                      'dataset_ready': result['dataset_ready'], 'episodes': len(episodes),
                      'images': len(image_paths), 'train_validation_image_overlap': split_overlap,
                      'fixed_evaluation_image_overlap': fixed_overlap}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
