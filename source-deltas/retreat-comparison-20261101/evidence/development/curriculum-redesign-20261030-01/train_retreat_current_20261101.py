"""現行指幅の教師から退避専用SmolVLAの7関節を小規模に学習する。"""
import argparse
import gc
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from joint_only_loss import sequence_loss
from relative_joint_action import save_policy_binding
from retreat_current_schedule_20261101 import make_schedule
from smolvla_runtime import ModelInput, load_policy

BASE = Path('outputs/dual-host-20261030')
CHECKPOINT = BASE / 'cuda-e-recovery-half-20261101-01/policy'
PACKAGE = BASE / 'retreat-current-package-20261101-01.zip'
EXPECTED_MODEL_SHA = 'b69054ec7138e65dabb6bb9e07cfdbd735b6fe7410a6eae7f877a647031d7043'
EXPECTED_PACKAGE_SHA = '460596d01615731da2ee243025e7cfa6fec919adf147355bf9211c8220db8b59'
PREFIXES = ('model.vlm_with_expert.lm_expert.', 'model.state_proj.', 'model.action_in_proj.',
            'model.action_out_proj.', 'model.action_time_mlp_in.', 'model.action_time_mlp_out.')
SEED = 20261101
LR = 1e-5
GRADIENT_LIMIT = 10.


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def make_sample(archive, row):
    sample = {'observation.state': torch.tensor(row['state'], dtype=torch.float32),
              'action': torch.tensor(row['action'], dtype=torch.float32).reshape(1, 8),
              'action_is_pad': torch.zeros(1, dtype=torch.bool)}
    for camera, side in (('camera1', 'left'), ('camera2', 'right')):
        data = archive.read(row['images'][side]['file'])
        if hashlib.sha256(data).hexdigest() != row['images'][side]['sha256']:
            raise ValueError('教材画像のハッシュが違います')
        with Image.open(BytesIO(data)) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        sample['observation.images.' + camera] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'candidate'), required=True)
    args = parser.parse_args()
    out = BASE / f'cuda-retreat-current-{args.mode}-20261101-01'
    if out.exists():
        raise FileExistsError(out)
    receipt = json.loads(PACKAGE.with_suffix('.json').read_text(encoding='utf8'))
    if (not receipt['completed'] or sha(PACKAGE) != EXPECTED_PACKAGE_SHA or receipt['sha256'] != EXPECTED_PACKAGE_SHA
            or sha(CHECKPOINT / 'model.safetensors') != EXPECTED_MODEL_SHA):
        raise ValueError('教材または初期重みが違います')
    with ZipFile(PACKAGE) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if (not manifest['completed'] or not manifest['passed']
                or [len(manifest['items'][key]) for key in ('train', 'validation')] != [80, 40]):
            raise ValueError('訓練・検査分割が不正です')
        rows = manifest['items']['train']
        schedule = make_schedule(rows, SEED)
        if args.mode == 'smoke':
            schedule = schedule[:1]
        out.mkdir()
        protocol = {'base_model_sha256': EXPECTED_MODEL_SHA, 'package_sha256': EXPECTED_PACKAGE_SHA,
                    'updates': len(schedule), 'seed': SEED, 'learning_rate': LR, 'gradient_limit': GRADIENT_LIMIT,
                    'schedule': schedule, 'inputs': ['left_rgb', 'right_rgb', 'robot_state'],
                    'scope': '開発Eの退避専用モデル。指幅・押圧・移行・安全監視は学習しない。'}
        (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        policy = load_policy(full_policy=CHECKPOINT, train=True, native_full=True, chunk_size=1, action_dim=8)
        inputs = ModelInput(policy, CHECKPOINT / 'normalization.json')
        parameters = [parameter for name, parameter in policy.named_parameters()
                      if name.startswith(PREFIXES) and parameter.requires_grad]
        if not parameters:
            raise ValueError('更新対象がありません')
        optimizer = torch.optim.AdamW(parameters, lr=LR)
        losses = []
        for i, (_, index) in enumerate(schedule, 1):
            batch = inputs.batch(make_sample(archive, rows[index]))
            torch.manual_seed(SEED + i)
            torch.cuda.manual_seed(SEED + i)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(policy, batch, first_weight=1.)
            if not torch.isfinite(loss):
                raise ValueError('損失が非有限です')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_LIMIT)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise ValueError('勾配が不正です')
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if i % 20 == 0:
                print(json.dumps({'updates': i, 'loss': losses[-1]}), flush=True)
        policy.eval()
        observation = inputs.batch({key: value for key, value in make_sample(archive, rows[0]).items()
                                    if key not in ('action', 'action_is_pad')})
        noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(714)).to('cuda')
        with torch.inference_mode():
            reference = policy.predict_action_chunk(observation, noise=noise.clone()).cpu()
        target = out / 'policy'
        policy.save_pretrained(target, safe_serialization=True)
        save_policy_binding(policy, target)
        (target / 'normalization.json').write_bytes((CHECKPOINT / 'normalization.json').read_bytes())
        del optimizer, parameters, policy, inputs
        gc.collect()
        torch.cuda.empty_cache()
        restored = load_policy(full_policy=target, chunk_size=1, action_dim=8)
        with torch.inference_mode():
            actual = restored.predict_action_chunk(observation, noise=noise.clone()).cpu()
        difference = float((reference - actual).abs().max())
        result = {'completed': difference < 1e-5, 'mode': args.mode, 'updates': len(schedule),
                  'losses': losses, 'model_sha256': sha(target / 'model.safetensors'),
                  'reload_max_abs_difference': difference,
                  'scope': '開発Eの教師から退避中の腕のみ学習。連続動作や汎化は未確認。'}
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        if not result['completed']:
            raise ValueError('保存後の出力が違います')
        print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
