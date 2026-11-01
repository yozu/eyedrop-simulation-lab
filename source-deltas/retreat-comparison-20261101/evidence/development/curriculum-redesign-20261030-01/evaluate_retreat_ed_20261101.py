"""E・D追加後の退避モデルを固定した開発条件で評価する。"""
import argparse
import hashlib
import importlib
import json
import sys

import torch
import eye_crop_override
from install_tree_eye_override_20261101 import install_tree_eye_override
from live_tree_eye_runtime_20261101 import BASE, MODEL_SHA, TreeEyeRuntime
from retreat_orientation_action_20261101 import projected_retreat_action
from eyedrop.panda_world import PandaWorld

FOLDER = BASE / 'dev-e-tree-diagnostic-20261101-01'
EVALUATOR = BASE / 'evaluate_retreat_ed_impl_20261101.py'
EVALUATOR_SHA = 'c71001eb26bdf76ed672c39b1ee7d656becdaabc832edc080903f66479d49ead'
FIXED_STATIC = BASE / 'cuda-e-recovery-half-static-20261101-01.json'
FIXED_STATIC_SHA = 'e9e7b22564dfca4b05e3b2356d7bf957884254bd3b0187b0be7c62b3ff9d8407'
RETREAT_STATIC = BASE / 'retreat-ed-static-20261101-01.json'
MAIN_MODEL_SHA = 'b69054ec7138e65dabb6bb9e07cfdbd735b6fe7410a6eae7f877a647031d7043'
PACKAGE_SHA = '3aba1c938a6ceb8c6a331aa28cf0e5586b21bb786ff6da5ecb12a5bf855ce040'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('e-train04', 'd-val00'), required=True)
    args = parser.parse_args()
    out = BASE / f'dev-retreat-ed-candidate-{args.case}-20261101-01'
    if out.exists():
        raise FileExistsError('既存の評価を上書きしません')
    preflight = json.loads((FOLDER / 'mac-preflight.json').read_text(encoding='utf8'))
    fixed = json.loads(FIXED_STATIC.read_text(encoding='utf8'))
    static = json.loads(RETREAT_STATIC.read_text(encoding='utf8'))
    trained = json.loads((BASE / 'dual-host-retreat-ed-candidate-20261101-01/result.json').read_text(encoding='utf8'))
    info = static['models']['candidate']
    if (not preflight['completed'] or not preflight['passed'] or preflight['count'] != 16
            or preflight['model_sha256'] != MODEL_SHA or sha(EVALUATOR) != EVALUATOR_SHA
            or sha(FIXED_STATIC) != FIXED_STATIC_SHA or not fixed['completed'] or not fixed['passed']
            or fixed['candidate_sha256'] != MAIN_MODEL_SHA or not static['completed']
            or static['package_sha256'] != PACKAGE_SHA or not trained['completed'] or trained['updates'] != 320
            or trained['mode'] != 'candidate' or trained['smoke']
            or info['model_sha256'] != trained['model_sha256']
            or info['normalization_sha256'] != trained['normalization_sha256']):
        raise ValueError('眼判定・主モデル・退避比較を確認できません')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = TreeEyeRuntime()

    def install(observer, _encoder, _ready_from_image):
        install_tree_eye_override(observer, runtime)

    eye_crop_override.install_eye_crop_override = install
    person = 'dev-e' if args.case == 'e-train04' else 'dev-d'
    sys.argv = [str(EVALUATOR), '--mode', 'candidate', '--person', person,
                '--position-model', 'candidate', '--diagnostic', '--alignment-profile', 'exploratory',
                '--retreat-mode', 'candidate', '--retreat-case', args.case, '--context-mode', 'candidate']
    evaluator = importlib.import_module('evaluate_retreat_ed_impl_20261101')
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
    projection_dir = BASE / f'retreat-ed-orientation-world-{args.case}-20261101-01'
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
