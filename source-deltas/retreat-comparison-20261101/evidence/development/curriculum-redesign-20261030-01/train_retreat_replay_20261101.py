"""退避の補正姿勢と従来の移動・停止を同時に追加学習する。"""
import argparse
import gc
import json
from pathlib import Path
from zipfile import ZipFile

import torch

from joint_only_loss import sequence_loss
from relative_joint_action import save_policy_binding
from retreat_replay_schedule_20261101 import schedule
from retreat_single_fit_probe_20261101 import atomic, metrics, passes, sha
from smolvla_runtime import ModelInput, load_policy
from train_retreat_current_20261101 import make_sample, PREFIXES, SEED, LR, GRADIENT_LIMIT

BASE = Path('outputs/dual-host-20261030')
PACKAGE = BASE / 'retreat-onpolicy-package-20261101-01.zip'
CHECKPOINT = BASE / 'cuda-retreat-onpolicy-candidate-20261101-01/policy'
PACKAGE_SHA = 'a9ac4bc0548fb8ac7076e8d1c021040a3ba07af375184d68b67760cef86bd955'
MODEL_SHA = 'c91c454364498536a0330e8588036dedab1f0bc5bfbdb374b4a473a7df7df5c6'
TOTAL_UPDATES = 320
SEEDS = (714, 715)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'candidate'), required=True)
    mode = parser.parse_args().mode
    out = BASE / f'cuda-retreat-replay-{mode}-20261101-01'
    if out.exists():
        raise FileExistsError(out)
    if sha(PACKAGE) != PACKAGE_SHA or sha(CHECKPOINT / 'model.safetensors') != MODEL_SHA:
        raise ValueError('教材または出発重みが違います')
    with ZipFile(PACKAGE) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        rows = manifest['items']['train']
        if (not manifest['passed'] or len(rows) != 88 or len(manifest['items']['validation']) != 40
                or any(row['source'] != 'onpolicy-train00-s715' for row in rows[80:])):
            raise ValueError('訓練・検査の固定条件が不成立です')
        allocation = schedule(rows, TOTAL_UPDATES, SEED)
        if mode == 'smoke':
            allocation = allocation[:1]
        out.mkdir()
        protocol = {'package_sha256': PACKAGE_SHA, 'initial_model_sha256': MODEL_SHA,
                    'updates': len(allocation), 'schedule': allocation, 'batch_size': 4,
                    'gradient_accumulation': 'two batches of two, each weighted 0.5',
                    'learning_rate': LR, 'gradient_limit': GRADIENT_LIMIT, 'seed': SEED,
                    'inputs': ['left_rgb', 'right_rgb', 'robot_state'],
                    'scope': '固定した訓練系列の追加学習。静止画だけでは点眼・退避・汎化の証明ではない。'}
        atomic(out / 'protocol.json', protocol)
        policy = load_policy(full_policy=CHECKPOINT, train=True, native_full=True, chunk_size=1, action_dim=8)
        inputs = ModelInput(policy, CHECKPOINT / 'normalization.json')
        parameters = [p for name, p in policy.named_parameters() if name.startswith(PREFIXES) and p.requires_grad]
        if not parameters:
            raise ValueError('更新対象がありません')
        optimizer = torch.optim.AdamW(parameters, lr=LR)

        def evaluate(row):
            example = make_sample(archive, row)
            observation = inputs.batch({k: v for k, v in example.items() if k not in ('action', 'action_is_pad')})
            result = {}
            for seed in SEEDS:
                noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(seed)).to('cuda')
                with torch.inference_mode():
                    raw = policy.predict_action_chunk(observation, noise=noise)[0, 0]
                    action = inputs.denormalize_action(raw, reference_state=example['observation.state']).tolist()
                result[seed] = metrics(row['state'], action, row['action'])
            return result

        policy.eval()
        before = {row['step']: evaluate(row) for row in rows[80:]}
        losses = []
        atomic(out / 'progress.json', {'status': 'running', 'updates': 0})
        for update, indices in enumerate(allocation, 1):
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            losses_this = []
            for part, pair in enumerate((indices[:2], indices[2:])):
                examples = [inputs.batch(make_sample(archive, rows[index])) for index in pair]
                if set(examples[0]) != set(examples[1]):
                    raise ValueError('教材の入力構成が異なります')
                batch = {key: torch.cat([x[key] for x in examples], dim=0) for key in examples[0]}
                torch.manual_seed(SEED + update * 2 + part)
                torch.cuda.manual_seed(SEED + update * 2 + part)
                loss = sequence_loss(policy, batch, first_weight=1.)
                if not torch.isfinite(loss):
                    raise ValueError('損失が非有限です')
                (loss * .5).backward()
                losses_this.append(float(loss.detach().cpu()))
            gradient = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_LIMIT)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise ValueError('勾配が不正です')
            optimizer.step()
            losses.append(losses_this)
            if update % 20 == 0:
                atomic(out / 'progress.json', {'status': 'running', 'updates': update, 'losses': losses_this})
                print(json.dumps({'updates': update, 'losses': losses_this}), flush=True)
        policy.eval()
        after = {row['step']: evaluate(row) for row in rows[80:]}
        sample = make_sample(archive, rows[80])
        observation = inputs.batch({k: v for k, v in sample.items() if k not in ('action', 'action_is_pad')})
        noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(714)).to('cuda')
        with torch.inference_mode():
            expected = policy.predict_action_chunk(observation, noise=noise.clone()).cpu()
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
        difference = float((expected - actual).abs().max())
        if difference >= 1e-5:
            raise ValueError('保存した重みの再読込が一致しません')
        result = {'completed': True, 'mode': mode, 'updates': len(allocation),
                  'initial_all': before, 'final_all': after,
                  'target_pair_passed': all(passes(after[step], before[step]) for step in (399, 454)),
                  'losses': losses, 'model_sha256': sha(target / 'model.safetensors'),
                  'reload_max_abs_difference': difference, 'scope': protocol['scope']}
        atomic(out / 'result.json', result)
        atomic(out / 'progress.json', {'status': 'finished', 'updates': len(allocation)})
        print(json.dumps({'completed': True, 'updates': len(allocation),
                          'target_pair_passed': result['target_pair_passed']}), flush=True)


if __name__ == '__main__':
    main()
