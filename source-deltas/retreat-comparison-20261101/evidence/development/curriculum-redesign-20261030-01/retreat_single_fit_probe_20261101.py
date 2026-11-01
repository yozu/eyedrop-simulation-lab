"""訓練用の1姿勢に限定し、モデルが教師指令を再現できるかを診断する。"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

BASE = Path('outputs/dual-host-20261030')
PACKAGE = BASE / 'retreat-onpolicy-package-20261101-01.zip'
CHECKPOINT = BASE / 'cuda-retreat-onpolicy-candidate-20261101-01/policy'
OUT = BASE / 'cuda-retreat-single-fit-20261101-01'
PACKAGE_SHA = 'a9ac4bc0548fb8ac7076e8d1c021040a3ba07af375184d68b67760cef86bd955'
MODEL_SHA = 'c91c454364498536a0330e8588036dedab1f0bc5bfbdb374b4a473a7df7df5c6'
TARGET_STEP = 454
SEEDS = (714, 715)
UPDATES = 320
INTERVAL = 80


def metrics(state, action, teacher):
    if any(len(values) < 7 or not all(math.isfinite(x) for x in values[:7])
           for values in (state, action, teacher)):
        raise ValueError('腕の状態または指令が不正です')
    actual = [a-s for s, a in zip(state[:7], action[:7], strict=True)]
    target = [a-s for s, a in zip(state[:7], teacher[:7], strict=True)]
    actual_norm = math.sqrt(sum(x*x for x in actual))
    target_norm = math.sqrt(sum(x*x for x in target))
    if target_norm <= 1e-12:
        raise ValueError('教師の移動量がありません')
    cosine = sum(a*b for a, b in zip(actual, target, strict=True)) / (actual_norm * target_norm) if actual_norm > 1e-12 else None
    return {'cosine': cosine, 'size_ratio': actual_norm / target_norm,
            'mean_abs_error_rad': sum(abs(a-b) for a, b in zip(action[:7], teacher[:7], strict=True)) / 7}


def passes(current, initial):
    return set(current) == {714, 715} and all(
        current[seed]['cosine'] is not None and current[seed]['cosine'] >= .5
        and current[seed]['size_ratio'] >= .5
        and current[seed]['mean_abs_error_rad'] < initial[seed]['mean_abs_error_rad']
        for seed in (714, 715))


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def main():
    import torch
    from joint_only_loss import sequence_loss
    from relative_joint_action import save_policy_binding
    from smolvla_runtime import ModelInput, load_policy
    from train_retreat_current_20261101 import make_sample, PREFIXES, SEED, LR, GRADIENT_LIMIT

    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(PACKAGE) != PACKAGE_SHA or sha(CHECKPOINT / 'model.safetensors') != MODEL_SHA:
        raise ValueError('教材または出発重みが違います')
    with ZipFile(PACKAGE) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        rows = manifest['items']['train']
        recovery = [row for row in rows[80:] if row['source'] == 'onpolicy-train00-s715']
        selected = [row for row in recovery if row['step'] == TARGET_STEP]
        if not manifest['passed'] or len(recovery) != 8 or len(selected) != 1 or len(manifest['items']['validation']) != 40:
            raise ValueError('訓練・検査の固定条件が不成立です')
        target = selected[0]
        OUT.mkdir()
        protocol = {'package_sha256': PACKAGE_SHA, 'initial_model_sha256': MODEL_SHA,
                    'target_step': TARGET_STEP, 'updates': UPDATES, 'checkpoint_interval': INTERVAL,
                    'seeds': SEEDS, 'learning_rate': LR, 'gradient_limit': GRADIENT_LIMIT,
                    'acceptance': '両乱数で方向余弦0.5以上・教師移動量の半分以上・0更新より誤差減少',
                    'scope': '訓練1姿勢の能力診断。候補採用・汎化・連続安全ではない。'}
        atomic(OUT / 'protocol.json', protocol)
        policy = load_policy(full_policy=CHECKPOINT, train=True, native_full=True, chunk_size=1, action_dim=8)
        inputs = ModelInput(policy, CHECKPOINT / 'normalization.json')
        parameters = [parameter for name, parameter in policy.named_parameters()
                      if name.startswith(PREFIXES) and parameter.requires_grad]
        if not parameters:
            raise ValueError('更新対象がありません')
        optimizer = torch.optim.AdamW(parameters, lr=LR)
        sample = make_sample(archive, target)
        batch = inputs.batch(sample)

        def evaluate(row):
            example = make_sample(archive, row)
            observation = inputs.batch({key: value for key, value in example.items()
                                        if key not in ('action', 'action_is_pad')})
            output = {}
            for seed in SEEDS:
                noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(seed)).to('cuda')
                with torch.inference_mode():
                    raw = policy.predict_action_chunk(observation, noise=noise)[0, 0]
                    action = inputs.denormalize_action(raw, reference_state=example['observation.state']).tolist()
                output[seed] = metrics(row['state'], action, row['action'])
            return output

        policy.eval()
        initial = evaluate(target)
        initial_all = {row['step']: evaluate(row) for row in recovery}
        checkpoints = [{'updates': 0, 'target': initial, 'passed': False}]
        losses = []
        atomic(OUT / 'progress.json', {'status': 'running', 'updates': 0, 'checkpoints': checkpoints})
        for update in range(1, UPDATES + 1):
            policy.train()
            torch.manual_seed(SEED + update)
            torch.cuda.manual_seed(SEED + update)
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
            if update % INTERVAL == 0:
                policy.eval()
                current = evaluate(target)
                checkpoints.append({'updates': update, 'target': current, 'passed': passes(current, initial)})
                atomic(OUT / 'progress.json', {'status': 'running', 'updates': update, 'checkpoints': checkpoints})
                print(json.dumps({'updates': update, 'passed': checkpoints[-1]['passed']}, ensure_ascii=False), flush=True)
        policy.eval()
        final_all = {row['step']: evaluate(row) for row in recovery}
        sample_input = inputs.batch({key: value for key, value in sample.items() if key not in ('action', 'action_is_pad')})
        noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(714)).to('cuda')
        with torch.inference_mode():
            expected = policy.predict_action_chunk(sample_input, noise=noise.clone()).cpu()
        target_dir = OUT / 'policy'
        policy.save_pretrained(target_dir, safe_serialization=True)
        save_policy_binding(policy, target_dir)
        (target_dir / 'normalization.json').write_bytes((CHECKPOINT / 'normalization.json').read_bytes())
        del optimizer, parameters, policy, inputs
        gc.collect()
        torch.cuda.empty_cache()
        restored = load_policy(full_policy=target_dir, chunk_size=1, action_dim=8)
        with torch.inference_mode():
            actual = restored.predict_action_chunk(sample_input, noise=noise.clone()).cpu()
        difference = float((expected - actual).abs().max())
        if difference >= 1e-5:
            raise ValueError('保存した重みの再読込が一致しません')
        result = {'completed': True, 'target_step': TARGET_STEP, 'updates': UPDATES, 'checkpoints': checkpoints,
                  'any_checkpoint_passed': any(row['passed'] for row in checkpoints),
                  'initial_all': initial_all, 'final_all': final_all, 'losses': losses,
                  'reload_max_abs_difference': difference, 'model_sha256': sha(target_dir / 'model.safetensors'),
                  'scope': protocol['scope']}
        atomic(OUT / 'result.json', result)
        atomic(OUT / 'progress.json', {'status': 'finished', 'updates': UPDATES,
                                       'any_checkpoint_passed': result['any_checkpoint_passed']})
        print(json.dumps({'completed': True, 'passed': result['any_checkpoint_passed']}), flush=True)


if __name__ == '__main__':
    main()
