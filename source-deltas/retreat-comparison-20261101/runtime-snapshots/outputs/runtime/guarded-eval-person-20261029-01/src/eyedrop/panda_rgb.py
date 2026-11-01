"""保存済みの認識モデルをPandaの左右RGBへ接続する。"""
import hashlib
import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .face_features import FaceFeatures
from .open_image_features import eye_image_features
from .stereo_control import StereoPosition

IMAGE_SIDE = 512
FACE_SIDE = 256


def face_rgb(content):
    with Image.open(io.BytesIO(content)) as image:
        if image.mode != 'RGB' or image.size != (IMAGE_SIDE, IMAGE_SIDE):
            raise ValueError('左右の画像は512角RGBである必要があります')
        output = io.BytesIO()
        image.resize((FACE_SIDE, FACE_SIDE), Image.Resampling.BILINEAR).save(output, format='PNG')
        return output.getvalue()


class PandaRgb:
    def __init__(self, root):
        root = Path(root)
        torch.set_num_threads(2)
        self.face = FaceFeatures(root/'separate-v2/openness', 'mps')
        self.position = StereoPosition(root/'target-readout-v1', root/'rgb-depth-v1/position.pkl')
        self.calibration = pickle.loads((root/'c-open-v7/calibration.pkl').read_bytes())
        files = ('separate-v2/openness/weights.pt', 'target-readout-v1/weights.pt',
                 'rgb-depth-v1/position.pkl', 'c-open-v7/calibration.pkl')
        self.hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}

    def predict(self, left, right):
        small = face_rgb(left)
        face_rgb(right)
        features = self.face.extract(small).cpu().numpy()
        opening = float(self.calibration.predict([eye_image_features(small, features)])[0])
        point, stereo = self.position.predict(features, small, left, right)
        if not np.isfinite([*point, opening]).all():
            raise ValueError('画像推定に非有限値があります')
        return {'eye_scene': point.tolist(), 'full_stereo_eye_scene': stereo['measured_point'],
                'openness': opening, 'stereo': stereo}
