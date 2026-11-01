"""未使用開始姿勢に触れず、同じ開発画像で固定2乱数を比較する。"""
from collections import defaultdict
import gc
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from retreat_context_features_20261101 import model_state
from retreat_diverse_selection_20261101 import choose
from retreat_single_fit_probe_20261101 import atomic, sha
from smolvla_runtime import ModelInput, load_policy
from train_retreat_current_20261101 import make_sample

HERE = Path(__file__).resolve().parent
BASE = Path('outputs/dual-host-20261030')
PARTITION = HERE / 'retreat-diverse-home-training-partition-20261101-01.json'
OLD = BASE / 'retreat-ed-package-20261101-01.zip'
NEW = BASE / 'retreat-diverse-home-data-20261101-01.zip'
CHECKPOINT = BASE / 'cuda-retreat-onpolicy-candidate-20261101-01/policy'
INITIAL_SHA = 'c91c454364498536a0330e8588036dedab1f0bc5bfbdb374b4a473a7df7df5c6'
RESULT = BASE / 'retreat-diverse-development-comparison-20261101-01.json'
PROGRESS = BASE / 'retreat-diverse-development-progress-20261101-01.json'
SEEDS = (714, 715)


def main():
    if RESULT.exists() or PROGRESS.exists():
        raise FileExistsError('既存比較を上書きしません')
    manifest = json.loads(PARTITION.read_text(encoding='utf8'))
    if (not manifest['completed'] or not manifest['passed'] or sha(OLD) != manifest['old_sha256']
            or sha(NEW) != manifest['new_sha256'] or sha(CHECKPOINT / 'model.safetensors') != INITIAL_SHA):
        raise ValueError('データまたは出発重みの識別が不一致です')
    rows = manifest['items']['development']
    if len(rows) != 200 or any(row['split'] != 'development' for row in rows):
        raise ValueError('開発用の分割が不正です')
    checkpoints = {'initial': CHECKPOINT}
    schedules = []
    for mode in ('coarse', 'exact'):
        root = BASE / f'cuda-retreat-diverse-{mode}-20261101-01'
        result = json.loads((root / 'result.json').read_text(encoding='utf8'))
        protocol = json.loads((root / 'protocol.json').read_text(encoding='utf8'))
        checkpoint = root / 'policy'
        if (not result['completed'] or result['mode'] != mode or result['smoke'] or result['updates'] != 320
                or result['reload_max_abs_difference'] >= 1e-5 or protocol['partition_sha256'] != sha(PARTITION)
                or protocol['initial_model_sha256'] != INITIAL_SHA
                or result['model_sha256'] != sha(checkpoint / 'model.safetensors')
                or result['normalization_sha256'] != sha(checkpoint / 'normalization.json')):
            raise ValueError('学習結果または重みが不一致です: ' + mode)
        checkpoints[mode] = checkpoint
        schedules.append(result['schedule_sha256'])
    if schedules[0] != schedules[1]:
        raise ValueError('2方式の教材抽出順が一致しません')
    outputs = []
    with ZipFile(OLD) as old, ZipFile(NEW) as new:
        archives = {'old': old, 'new': new}
        for mode, checkpoint in checkpoints.items():
            policy = load_policy(full_policy=checkpoint, chunk_size=1, action_dim=8)
            inputs = ModelInput(policy, checkpoint / 'normalization.json')
            norm = json.loads((checkpoint / 'normalization.json').read_text(encoding='utf8'))
            means = norm['features']['observation.state']['mean'][16:30]
            for index, row in enumerate(rows):
                sample = make_sample(archives[row['archive']], row)
                if mode != 'initial':
                    sample['observation.state'] = torch.tensor(model_state(sample['observation.state'].tolist(),
                        row['context_' + mode], means, use_context=True), dtype=torch.float32)
                observation = inputs.batch({k: v for k, v in sample.items() if k not in ('action', 'action_is_pad')})
                teacher = np.asarray(row['action'][:7], dtype=np.float64)
                for seed in SEEDS:
                    noise = torch.randn(1, 1, 32, generator=torch.Generator().manual_seed(seed)).to('cuda')
                    with torch.inference_mode():
                        raw = policy.predict_action_chunk(observation, noise=noise.clone())[0, 0]
                        predicted = np.asarray(inputs.denormalize_action(raw,
                            reference_state=sample['observation.state'])[:7], dtype=np.float64)
                    error = np.abs(predicted - teacher)
                    if not np.isfinite(error).all():
                        raise ValueError('指令誤差が非有限です')
                    outputs.append({'model': mode, 'group': row['archive'] + '_' + row['kind'],
                                    'index': index, 'seed': seed, 'mean_abs_rad': float(error.mean())})
                if (index + 1) % 20 == 0:
                    atomic(PROGRESS, {'status': 'running', 'model': mode, 'samples': index + 1})
            del policy, inputs
            gc.collect()
            torch.cuda.empty_cache()
    groups = defaultdict(list)
    for row in outputs:
        groups[(row['model'], row['group'])].append(row['mean_abs_rad'])
    means = {mode: {group: float(np.mean(values)) for (m, group), values in groups.items() if m == mode}
             for mode in checkpoints}
    selected = choose(means)
    atomic(RESULT, {'completed': True, 'partition_sha256': sha(PARTITION), 'schedule_sha256': schedules[0],
                    'means': means, 'selected': selected, 'rows': outputs,
                    'scope': '開発用の静的比較。連続動作や未使用人物の成功ではない。'})
    atomic(PROGRESS, {'status': 'finished', 'samples': len(outputs)})
    print(json.dumps({'completed': True, 'means': means, 'selected': selected}, ensure_ascii=False))


if __name__ == '__main__':
    main()
