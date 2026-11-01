"""画像と自己形状から、眼領域の変化と観測不能を記録する。"""
import base64
import binascii
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
from eyedrop.eye_image_change import EyeImageChange


class RGBGuardObserver:
    def __init__(self, browser, world, face_model, directory):
        self.browser = browser
        self.world = world
        self.face_model = face_model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.detector = EyeImageChange()
        self.records = []

    def __call__(self):
        frame = self.browser.call('render', state={'time': self.world.time, 'links': self.world.links(),
                                                  'eye_condition': 'natural', 'capture_face_robot_mask': True},
                                  capture=True)
        try:
            face = base64.b64decode(frame['face'], validate=True)
            mask = base64.b64decode(frame['robot_mask_face'], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as error:
            raise ValueError('眼画像または自己形状の取得に失敗しました') from error
        with Image.open(io.BytesIO(face)) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
        with Image.open(io.BytesIO(mask)) as image:
            robot_mask = np.asarray(image.convert('L')) > 0
        features = self.face_model.extract(face).cpu().numpy()
        centers = (features[:4].reshape(2, 2)+1)/2
        changed = self.detector.observe_visible(rgb, centers, robot_mask)
        number = len(self.records)
        face_name = f'face-{number:06}.png'
        mask_name = f'robot-mask-{number:06}.png'
        with (self.directory/face_name).open('xb') as f:
            f.write(face)
        with (self.directory/mask_name).open('xb') as f:
            f.write(mask)
        self.records.append({'time': self.world.time, 'face': face_name, 'mask': mask_name,
                             'face_sha256': hashlib.sha256(face).hexdigest(),
                             'mask_sha256': hashlib.sha256(mask).hexdigest(),
                             'changed_pixels': changed})
        return changed
