"""SmolVLAへ渡す、同一エピソード内の過去RGBビュー。"""
from collections import deque
from numbers import Integral


IMAGE_LAGS = tuple(range(11, -1, -1))
MAX_HISTORY = 12
CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
)
_CONTRACT = {
    "version": 1,
    "fps": 20,
    "max_history": MAX_HISTORY,
    "lags": list(IMAGE_LAGS),
    "padding": "repeat_first_observation_within_episode",
    "input": "past_synchronized_rgb_views_only",
    "order": "oldest_to_current_camera1_camera2",
}


def temporal_rgb_contract():
    return {**_CONTRACT, "lags": list(IMAGE_LAGS)}


def legacy_image_feature_keys():
    return CAMERA_KEYS


def temporal_image_feature_keys(contract):
    if contract != temporal_rgb_contract():
        raise ValueError("時系列RGB契約が未対応です")
    keys = []
    for lag in IMAGE_LAGS:
        suffix = "" if lag == 0 else f"_past_{lag}"
        keys.extend(f"{camera}{suffix}" for camera in CAMERA_KEYS)
    return tuple(keys)


class CausalRGBHistory:
    """各呼出し時点までに見た同期RGBだけを保持する。"""

    def __init__(self, contract):
        temporal_image_feature_keys(contract)
        self.frames = deque(maxlen=MAX_HISTORY)

    @property
    def observations_seen(self):
        return len(self.frames)

    def reset(self):
        self.frames.clear()

    def append(self, camera1, camera2):
        if camera1 is None or camera2 is None:
            raise ValueError("左右カメラ画像が必要です")
        self.frames.append((camera1, camera2))
        frames = list(self.frames)
        result = {}
        for lag in IMAGE_LAGS:
            source_index = max(0, len(frames) - 1 - lag)
            suffix = "" if lag == 0 else f"_past_{lag}"
            for camera_index, camera in enumerate(CAMERA_KEYS):
                key = f"{camera}{suffix}"
                result[key] = frames[source_index][camera_index]
                result[f"{key}_padding_mask"] = lag < len(frames)
        return result


class TemporalRGBDataset:
    """学習標本へ同じエピソードの過去RGBを加える読み取り専用view。"""

    def __init__(self, dataset, contract):
        temporal_image_feature_keys(contract)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _metadata(sample):
        if "episode_index" not in sample or "frame_index" not in sample:
            raise ValueError("時系列RGBにはepisode_indexとframe_indexが必要です")
        episode = sample["episode_index"]
        frame = sample["frame_index"]
        if isinstance(episode, bool) or isinstance(frame, bool):
            raise ValueError("エピソード・フレーム番号が不正です")
        try:
            episode, frame = int(episode), int(frame)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("エピソード・フレーム番号が不正です") from error
        if episode < 0 or frame < 0:
            raise ValueError("エピソード・フレーム番号が負です")
        return episode, frame

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("単一フレームの整数indexが必要です")
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)

        current = self.dataset[index]
        episode, frame = self._metadata(current)
        episode_start = index - frame
        if episode_start < 0:
            raise ValueError("エピソード先頭を特定できません")

        result = dict(current)
        for lag in IMAGE_LAGS:
            target_frame = max(0, frame - lag)
            source_index = episode_start + target_frame
            if source_index > index:
                raise ValueError("未来フレーム参照を拒否しました")
            source = current if source_index == index else self.dataset[source_index]
            if self._metadata(source) != (episode, target_frame):
                raise ValueError("過去RGBのエピソード境界または時刻が不一致です")
            suffix = "" if lag == 0 else f"_past_{lag}"
            for camera in CAMERA_KEYS:
                if camera not in source:
                    raise ValueError(f"過去RGBがありません: {camera}")
                key = f"{camera}{suffix}"
                result[key] = source[camera]
                result[f"{key}_padding_mask"] = lag <= frame
        return result
