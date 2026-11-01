"""２種類の顔予測と固定眼モデルから診断用の開眼値を計算する。"""
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from blink_head_mix_runtime import MixedHead
from eyedrop.face_eye_patches import eye_patches,normalized_eye_patches
from eyedrop.face_features import FaceFeatures,face_image_tensor
from live_face_feature_contract_20261101 import combine_live_features

BASE=Path('outputs/diagnostics/curriculum-redesign-20261030-01')
INPUT=BASE/'face-context-coherent-input-20261101-01'
MODEL=BASE/'dev-e-tree-diagnostic-20261101-01/classifier.joblib'
MODEL_SHA='e4d633088a0902029d5b6567135b57221ecd88dbcac34c1a063f8ef94dcb8f31'
OLD_SHA='7e4c8716130ed4323ab4e6b7ca7e1841d0fb13c4e5f0e89e595b6fe1ac1121a2'
NEW_SHA='a24b3ed33f965935081b94533a1d1a30e08eb1ce47e37b9bfd8e6ae6c5cf68c1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class OpenConfirmation:
    def __init__(self):self.previous=False
    def update(self,opened):
        current=bool(opened);result=self.previous and current;self.previous=current;return result


class TreeEyeRuntime:
    def __init__(self):
        if sha(MODEL)!=MODEL_SHA:raise ValueError('診断判定器が違います')
        self.model=joblib.load(MODEL)
        self.old=FaceFeatures(Path('outputs/learning/separate-v2/openness'),'cpu',expected_sha256=OLD_SHA)
        self.new=FaceFeatures(BASE/'seven-stereo-face-candidate-20261031-01','cpu',expected_sha256=NEW_SHA)
        card=json.loads((INPUT/'mix.json').read_text(encoding='utf8'))
        if not card['completed'] or card['selected_alpha']!=.5:raise ValueError('混合条件が違います')
        expected={'previous-head.pt':card['models']['previous'],'boundary-head.pt':card['models']['boundary'],
                  'bce-head.pt':'76f21ccdc35957febe3675983ac0421aedf5f1503671290fda987f34d64d3ede',
                  'mixed-head.pt':'fb6fceeee6b2d3119df3902b2c659d708965d0d36ee2c858edca55b5eacc6fab'}
        if any(sha(INPUT/name)!=digest for name,digest in expected.items()):raise ValueError('固定眼判定器が違います')
        def base(name):
            head=copy.deepcopy(self.old.model.openness_head)
            head.load_state_dict(torch.load(INPUT/name,map_location='cpu',weights_only=True))
            return head
        def mixed():return MixedHead(base('previous-head.pt'),base('boundary-head.pt'),.5).eval()
        self.heads=[mixed(),mixed(),mixed()]
        for head,name in zip(self.heads[1:],('bce-head.pt','mixed-head.pt')):
            head.load_state_dict(torch.load(INPUT/name,map_location='cpu',weights_only=True));head.eval()

    def score(self,blob):
        image=face_image_tensor(blob).unsqueeze(0)
        with torch.inference_mode():
            old_pos=self.old.batch(image)[:,:8];new_pos=self.new.batch(image)[:,:8]
            old_patch=normalized_eye_patches(eye_patches(image,(old_pos.reshape(-1,4,2)+1)/2))
            new_patch=normalized_eye_patches(eye_patches(image,(new_pos.reshape(-1,4,2)+1)/2))
            old_logits=np.concatenate([h(old_patch.flatten(0,1)).reshape(1,2).numpy() for h in self.heads],axis=1)
            new_logits=np.concatenate([h(new_patch.flatten(0,1)).reshape(1,2).numpy() for h in self.heads[1:]],axis=1)
            dual,_=combine_live_features(old_pos.numpy(),new_pos.numpy(),old_logits,new_logits,np.zeros((1,256)))
        score=float(self.model.predict_proba(dual)[0,1])
        if not np.isfinite(score):raise ValueError('開眼判定が非有限です')
        return score
