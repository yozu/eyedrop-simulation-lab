"""固定した新旧教材で、開始姿勢の表現を比較する。"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from joint_only_loss import sequence_loss
from relative_joint_action import save_policy_binding
from retreat_context_features_20261101 import model_state
from retreat_diverse_schedule_20261101 import make_schedule
from retreat_precise_binding_20261101 import rebind
from retreat_single_fit_probe_20261101 import atomic, sha
from smolvla_runtime import ModelInput, load_policy
from train_retreat_current_20261101 import make_sample, PREFIXES, SEED, LR, GRADIENT_LIMIT

HERE = Path(__file__).resolve().parent
BASE = Path('outputs/dual-host-20261030')
PARTITION = HERE / 'retreat-diverse-home-training-partition-20261101-01.json'
OLD = BASE / 'retreat-ed-package-20261101-01.zip'
NEW = BASE / 'retreat-diverse-home-data-20261101-01.zip'
CHECKPOINT = BASE / 'cuda-retreat-onpolicy-candidate-20261101-01/policy'
MODEL_SHA = 'c91c454364498536a0330e8588036dedab1f0bc5bfbdb374b4a473a7df7df5c6'
UPDATES = 320
START, END = 16, 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('coarse', 'exact'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    suffix = '-smoke' if args.smoke else ''
    out = BASE / f'cuda-retreat-diverse-{args.mode}{suffix}-20261101-01'
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(PARTITION.read_text(encoding='utf8'))
    if (not manifest['completed'] or not manifest['passed'] or sha(OLD) != manifest['old_sha256']
            or sha(NEW) != manifest['new_sha256'] or sha(CHECKPOINT / 'model.safetensors') != MODEL_SHA):
        raise ValueError('教材・分割監査・出発重みが一致しません')
    rows = manifest['items']['train']
    if len(rows) != 350 or any(row['split'] != 'train' for row in rows):
        raise ValueError('訓練の分割が不正です')
    schedule = make_schedule(rows, UPDATES, SEED)
    if args.smoke:
        schedule = schedule[:1]
    stats = manifest['training_context_stats']['context_' + args.mode]
    means, stds = [x['mean'] for x in stats], [x['std'] for x in stats]
    if len(means) != 14 or len(stds) != 14:
        raise ValueError('訓練統計の次元が不正です')
    original = json.loads((CHECKPOINT / 'normalization.json').read_text(encoding='utf8'))
    norm = copy.deepcopy(original)
    norm['features']['observation.state']['mean'][START:END] = means
    norm['features']['observation.state']['std'][START:END] = stds
    norm['retreat_context_contract'] = {'version': 2, 'mode': args.mode, 'source': 'own_observed_approach',
                                        'anchor_quantization_rad': .02 if args.mode == 'coarse' else None,
                                        'slots': [START, END]}
    out.mkdir()
    normpath = out / 'normalization.json'
    atomic(normpath, norm)
    schedule_sha = hashlib.sha256(json.dumps(schedule, separators=(',', ':')).encode()).hexdigest()
    atomic(out / 'protocol.json', {'partition_sha256': sha(PARTITION), 'initial_model_sha256': MODEL_SHA,
           'old_sha256': manifest['old_sha256'], 'new_sha256': manifest['new_sha256'], 'mode': args.mode,
           'updates': len(schedule), 'schedule': schedule, 'schedule_sha256': schedule_sha,
           'batch_size': 4, 'learning_rate': LR, 'gradient_limit': GRADIENT_LIMIT, 'seed': SEED,
           'normalization_sha256': sha(normpath), 'inputs': ['left_rgb', 'right_rgb', 'robot_state'],
           'scope': '退避専用。本人の腕の観測履歴を使用。人物ID、眼の正解や未来情報は入力しない。'})
    with ZipFile(OLD) as old, ZipFile(NEW) as new:
        archives = {'old': old, 'new': new}
        policy = load_policy(full_policy=CHECKPOINT, train=True, native_full=True, chunk_size=1, action_dim=8)
        rebind(policy, original, norm, mode=args.mode)
        inputs = ModelInput(policy, normpath)
        parameters = [p for name, p in policy.named_parameters() if name.startswith(PREFIXES) and p.requires_grad]
        if not parameters:
            raise ValueError('更新対象がありません')
        optimizer = torch.optim.AdamW(parameters, lr=LR)

        def sample_for(index):
            row = rows[index]
            sample = make_sample(archives[row['archive']], row)
            sample['observation.state'] = torch.tensor(model_state(sample['observation.state'].tolist(),
                row['context_' + args.mode], means, use_context=True), dtype=torch.float32)
            return sample

        losses = []
        atomic(out / 'progress.json', {'status': 'running', 'updates': 0})
        for update, indices in enumerate(schedule, 1):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            values = []
            for part, pair in enumerate((indices[:2], indices[2:])):
                examples = [inputs.batch(sample_for(index)) for index in pair]
                if set(examples[0]) != set(examples[1]):
                    raise ValueError('教材の入力構成が異なります')
                batch = {key: torch.cat([x[key] for x in examples], dim=0) for key in examples[0]}
                torch.manual_seed(SEED + update * 2 + part)
                torch.cuda.manual_seed(SEED + update * 2 + part)
                loss = sequence_loss(policy, batch, first_weight=1.)
                if not torch.isfinite(loss):
                    raise ValueError('損失が非有限です')
                (loss * .5).backward()
                values.append(float(loss.detach().cpu()))
            gradient = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_LIMIT)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise ValueError('勾配が不正です')
            optimizer.step()
            losses.append(values)
            if update % 20 == 0:
                atomic(out / 'progress.json', {'status': 'running', 'updates': update, 'losses': values})
                print(json.dumps({'updates': update, 'losses': values}), flush=True)
        policy.eval()
        sample = sample_for(schedule[0][0])
        observation = inputs.batch({k: v for k, v in sample.items() if k not in ('action', 'action_is_pad')})
        noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(714)).to('cuda')
        with torch.inference_mode():
            expected = policy.predict_action_chunk(observation, noise=noise.clone()).cpu()
        target = out / 'policy'
        policy.save_pretrained(target, safe_serialization=True)
        save_policy_binding(policy, target)
        (target / 'normalization.json').write_bytes(normpath.read_bytes())
        del optimizer, parameters, policy, inputs
        gc.collect()
        torch.cuda.empty_cache()
        restored = load_policy(full_policy=target, chunk_size=1, action_dim=8)
        restored_inputs = ModelInput(restored, target / 'normalization.json')
        restored_observation = restored_inputs.batch({k: v for k, v in sample.items() if k not in ('action', 'action_is_pad')})
        if any(not torch.equal(observation[k], restored_observation[k]) for k in observation):
            raise ValueError('保存後の入力変換が一致しません')
        with torch.inference_mode():
            actual = restored.predict_action_chunk(observation, noise=noise.clone()).cpu()
        difference = float((expected - actual).abs().max())
        if difference >= 1e-5:
            raise ValueError('保存した重みの再読込が一致しません')
        atomic(out / 'result.json', {'completed': True, 'mode': args.mode, 'smoke': args.smoke,
               'updates': len(schedule), 'schedule_sha256': schedule_sha, 'losses': losses,
               'model_sha256': sha(target / 'model.safetensors'), 'normalization_sha256': sha(normpath),
               'reload_max_abs_difference': difference})
        atomic(out / 'progress.json', {'status': 'finished', 'updates': len(schedule)})
        print(json.dumps({'completed': True, 'mode': args.mode, 'updates': len(schedule)}), flush=True)


if __name__ == '__main__':
    main()
