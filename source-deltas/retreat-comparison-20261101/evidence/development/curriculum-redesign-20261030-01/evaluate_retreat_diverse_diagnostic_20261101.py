"""静的には不採用の退避候補を、固定した開発条件で診断する。"""
import argparse
import hashlib
import importlib
import json
import sys

import torch
import eye_crop_override
from install_tree_eye_override_20261101 import install_tree_eye_override
from live_tree_eye_runtime_20261101 import BASE, MODEL_SHA, TreeEyeRuntime
from retreat_diverse_selection_20261101 import choose
from retreat_orientation_action_20261101 import projected_retreat_action
from eyedrop.panda_world import PandaWorld

FOLDER = BASE / 'dev-e-tree-diagnostic-20261101-01'
EVALUATOR = BASE / 'evaluate_retreat_diverse_impl_20261101.py'
EVALUATOR_SHA = 'd534ea059cf0069c6be1314ae612fff3f2897a7a19e580efe0a4ec22b7203664'
FIXED_STATIC = BASE / 'cuda-e-recovery-half-static-20261101-01.json'
FIXED_STATIC_SHA = 'e9e7b22564dfca4b05e3b2356d7bf957884254bd3b0187b0be7c62b3ff9d8407'
RETREAT_STATIC = BASE / 'retreat-diverse-development-comparison-20261101-01.json'
MAIN_MODEL_SHA = 'b69054ec7138e65dabb6bb9e07cfdbd735b6fe7410a6eae7f877a647031d7043'
INITIAL_SHA = 'c91c454364498536a0330e8588036dedab1f0bc5bfbdb374b4a473a7df7df5c6'
TRAIN = BASE / 'dual-host-retreat-diverse-coarse-20261101-01'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('e-train04', 'd-val00'), required=True)
    args = parser.parse_args()
    out = BASE / f'dev-retreat-diverse-coarse-diagnostic-{args.case}-20261101-01'
    if out.exists():
        raise FileExistsError('既存の評価を上書きしません')
    preflight = json.loads((FOLDER / 'mac-preflight.json').read_text(encoding='utf8'))
    fixed = json.loads(FIXED_STATIC.read_text(encoding='utf8'))
    static = json.loads(RETREAT_STATIC.read_text(encoding='utf8'))
    trained = json.loads((TRAIN / 'result.json').read_text(encoding='utf8'))
    protocol = json.loads((TRAIN / 'protocol.json').read_text(encoding='utf8'))
    policy = TRAIN / 'policy-final'
    if (not preflight['completed'] or not preflight['passed'] or preflight['count'] != 16
            or preflight['model_sha256'] != MODEL_SHA or sha(EVALUATOR) != EVALUATOR_SHA
            or sha(FIXED_STATIC) != FIXED_STATIC_SHA or not fixed['completed'] or not fixed['passed']
            or fixed['candidate_sha256'] != MAIN_MODEL_SHA or not static['completed'] or static['selected'] is not None
            or choose(static['means']) is not None or not trained['completed'] or trained['updates'] != 320
            or trained['mode'] != 'coarse' or trained['smoke'] or trained['reload_max_abs_difference'] >= 1e-5
            or protocol['initial_model_sha256'] != INITIAL_SHA or protocol['partition_sha256'] != static['partition_sha256']
            or trained['schedule_sha256'] != static['schedule_sha256']
            or sha(policy / 'model.safetensors') != trained['model_sha256']
            or sha(policy / 'normalization.json') != trained['normalization_sha256']):
        raise ValueError('固定条件・不採用の診断対象・重みを確認できません')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = TreeEyeRuntime()

    def install(observer, _encoder, _ready_from_image):
        install_tree_eye_override(observer, runtime)

    eye_crop_override.install_eye_crop_override = install
    person = 'dev-e' if args.case == 'e-train04' else 'dev-d'
    sys.argv = [str(EVALUATOR), '--mode', 'candidate', '--person', person, '--position-model', 'candidate',
                '--diagnostic', '--alignment-profile', 'exploratory', '--retreat-mode', 'candidate',
                '--retreat-case', args.case, '--context-mode', 'coarse']
    evaluator = importlib.import_module('evaluate_retreat_diverse_impl_20261101')
    evaluator.OUT = out
    evaluator.TARGET_HEIGHT_M = .055
    original_atomic = evaluator.atomic_json

    def atomic(path, data):
        if path == out / 'protocol.json':
            data = {**data, 'readiness': 'dual-tree-two-frame-diagnostic', 'eye_model_sha256': MODEL_SHA,
                    'static_gate_passed': False, 'diagnostic_only': True, 'eye_inputs': ['native_face_rgb'],
                    'confirmation': 'current_and_previous', 'target_height_m': .055,
                    'retreat_static_sha256': sha(RETREAT_STATIC), 'preflight_sha256': sha(FOLDER / 'mac-preflight.json'),
                    'retreat_orientation': 'fixed_ik_separate_world',
                    'orientation_action_sha256': sha(BASE / 'retreat_orientation_action_20261101.py')}
        return original_atomic(path, data)

    evaluator.atomic_json = atomic
    projection_dir = BASE / f'retreat-diverse-diagnostic-orientation-world-{args.case}-20261101-01'
    projection_dir.mkdir(exist_ok=False)
    projection_world = PandaWorld(projection_dir)

    def projected(gate, own_joints, model_action, fixed_gap, clamp):
        return projected_retreat_action(gate, own_joints, model_action, fixed_gap, clamp, projection_world)

    evaluator.retreat_action = projected
    try:
        evaluator.main()
    finally:
        projection_world.close()


if __name__ == '__main__':
    main()
