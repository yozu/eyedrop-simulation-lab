"""実用診断に限り、固定された２つの眼判定器を組み合わせる。"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from blink_head_mix_runtime import MixedHead

ALPHA = .5
FACE_SHA = '7e4c8716130ed4323ab4e6b7ca7e1841d0fb13c4e5f0e89e595b6fe1ac1121a2'
HEAD_SHA = 'ed64e9290fd0ead9570bfaad200df41d25b410bcb39f61e5f4aedcf81c9bae2f'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(observer, folder, extra):
    folder, extra = Path(folder), Path(extra)
    mix_path = folder/'mix.json'
    mix = json.loads(mix_path.read_text(encoding='utf-8'))
    quality = json.loads(extra.read_text(encoding='utf-8'))
    if (observer.face.sha256 != FACE_SHA or observer.ready_head is None
            or observer.hashes.get('diagnostic_ready/head.pt') != HEAD_SHA
            or not mix['completed'] or mix['selected_alpha'] != ALPHA or mix['static_gate']
            or not quality['completed'] or not quality['exploratory_gate']
            or quality['prior_stricter_gate_passed'] is not False or quality['alpha'] != ALPHA
            or quality['mix_sha256'] != sha(mix_path) or quality['models'] != mix['models']):
        raise ValueError('混合の診断条件が一致しません')
    heads = []
    for name in ('previous', 'boundary'):
        path = folder/f'{name}-head.pt'
        if sha(path) != mix['models'][name]:
            raise ValueError('混合する頭部のハッシュが違います: '+name)
        head = copy.deepcopy(observer.face.model.openness_head).to(observer.face.device)
        head.load_state_dict(torch.load(path, map_location=observer.face.device, weights_only=True))
        heads.append(head.eval())
    observer.ready_head = MixedHead(*heads, ALPHA).to(observer.face.device).eval()
    observer.hashes['diagnostic_ready/mix'] = sha(mix_path)
    return {'alpha': ALPHA, 'models': mix['models'], 'mix_sha256': sha(mix_path),
            'extra_sha256': sha(extra), 'prior_stricter_gate_passed': False}
