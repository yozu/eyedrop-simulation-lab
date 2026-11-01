"""固定した接近・押圧の後に、退避モデルだけを切り替えて比較する。"""
import argparse
import hashlib
import importlib
import json
import sys

import torch
import eye_crop_override
from install_tree_eye_override_20261101 import install_tree_eye_override
from live_tree_eye_runtime_20261101 import BASE, MODEL_SHA, TreeEyeRuntime

FOLDER = BASE / 'dev-e-tree-diagnostic-20261101-01'
EVALUATOR = BASE / 'evaluate_learned_retreat_impl_20261101.py'
EVALUATOR_SHA = '5700442d1275211aed619ecee2845fdd7248629310a22294aeab0bd4a6bf239b'
FIXED_STATIC = BASE / 'cuda-e-recovery-half-static-20261101-01.json'
FIXED_STATIC_SHA = 'e9e7b22564dfca4b05e3b2356d7bf957884254bd3b0187b0be7c62b3ff9d8407'
RETREAT_STATIC = BASE / 'cuda-retreat-current-static-20261101-01.json'
MAIN_MODEL_SHA = 'b69054ec7138e65dabb6bb9e07cfdbd735b6fe7410a6eae7f877a647031d7043'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('e-train04', 'd-val00'), required=True)
    parser.add_argument('--retreat-mode', choices=('baseline', 'candidate'), required=True)
    args = parser.parse_args()
    out = BASE / f'dev-retreat-learned-{args.case}-{args.retreat_mode}-20261101-01'
    if out.exists():
        raise FileExistsError(out)
    preflight = json.loads((FOLDER / 'mac-preflight.json').read_text(encoding='utf8'))
    fixed = json.loads(FIXED_STATIC.read_text(encoding='utf8'))
    retreat = json.loads(RETREAT_STATIC.read_text(encoding='utf8'))
    trained = json.loads((BASE / 'dual-host-retreat-current-candidate-20261101-01/result.json').read_text(encoding='utf8'))
    if (not preflight['completed'] or not preflight['passed'] or preflight['count'] != 16
            or preflight['model_sha256'] != MODEL_SHA or sha(EVALUATOR) != EVALUATOR_SHA
            or sha(FIXED_STATIC) != FIXED_STATIC_SHA or not fixed['completed'] or not fixed['passed']
            or fixed['candidate_sha256'] != MAIN_MODEL_SHA or not retreat['completed'] or not retreat['passed']
            or retreat['baseline_sha256'] != MAIN_MODEL_SHA or not trained['completed']
            or retreat['candidate_sha256'] != trained['model_sha256']):
        raise ValueError('眼判定・主モデル・退避の固定画像比較を確認できません')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = TreeEyeRuntime()

    def install(observer, _encoder, _ready_from_image):
        install_tree_eye_override(observer, runtime)

    eye_crop_override.install_eye_crop_override = install
    person = 'dev-e' if args.case == 'e-train04' else 'dev-d'
    sys.argv = [str(EVALUATOR), '--mode', 'candidate', '--person', person,
                '--position-model', 'candidate', '--diagnostic', '--alignment-profile', 'exploratory',
                '--retreat-mode', args.retreat_mode, '--retreat-case', args.case]
    evaluator = importlib.import_module('evaluate_learned_retreat_impl_20261101')
    evaluator.OUT = out
    evaluator.TARGET_HEIGHT_M = .055
    original_atomic = evaluator.atomic_json

    def atomic(path, data):
        if path == out / 'protocol.json':
            data = {**data, 'readiness': 'dual-tree-two-frame-diagnostic', 'eye_model_sha256': MODEL_SHA,
                    'static_gate_passed': False, 'diagnostic_only': True, 'eye_inputs': ['native_face_rgb'],
                    'confirmation': 'current_and_previous', 'target_height_m': .055,
                    'retreat_static_sha256': sha(RETREAT_STATIC), 'preflight_sha256': sha(FOLDER / 'mac-preflight.json')}
        return original_atomic(path, data)

    evaluator.atomic_json = atomic
    evaluator.main()


if __name__ == '__main__':
    main()
