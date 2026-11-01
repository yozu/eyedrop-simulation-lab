"""SmolVLAの時間履歴データと実行時正規化の結び付けを検査する。"""
import hashlib
import json
from pathlib import Path
from .guarded_action_contract import guarded_action_contract
from .smolvla_observation import command_observation_contract
from .smolvla_temporal import temporal_rgb_contract


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_temporal_dataset_contract(gate, normalization, normalization_path):
    normalization_sha256 = file_sha256(normalization_path)
    guard = normalization.get('guarded_action_contract')
    gate_guard = gate.get('guarded_action_contract')
    if guard is not None or gate_guard is not None:
        if guard != guarded_action_contract() or gate_guard != guard:
            raise ValueError('ガード付き行動のデータ契約が一致しません')
        if gate.get('accepted') is not True or gate.get('normalization_sha256') != normalization_sha256:
            raise ValueError('ガード付きデータの品質または正規化SHAが一致しません')
        if gate.get('language_instruction') != normalization.get('language_instruction'):
            raise ValueError('ガード付きデータの指示文が一致しません')
    temporal_contract = normalization.get("temporal_rgb_contract")
    gate_temporal_contract = gate.get("temporal_rgb_contract")
    if temporal_contract is None and gate_temporal_contract is None:
        return normalization_sha256
    if temporal_contract is None or gate_temporal_contract is None:
        raise ValueError("データ出力と正規化の時間履歴契約が揃っていません")
    if gate.get("accepted") is not True:
        raise ValueError("時間履歴データの品質ゲートが未合格です")
    if gate.get("normalization_sha256") != normalization_sha256:
        raise ValueError("時間履歴データの正規化SHAが一致しません")
    if gate.get("language_instruction") != normalization.get("language_instruction"):
        raise ValueError("データ出力と正規化の指示文が一致しません")
    if gate.get("temporal_rgb_contract") != temporal_contract:
        raise ValueError("データ出力と正規化の時間履歴契約が一致しません")
    return normalization_sha256


def compare_view_normalizations(current_gate, current, current_path, history_gate, history, history_path):
    """現在画像と履歴画像で共通の正規化・契約が一致するか確認する。"""
    for document, path in ((current, current_path), (history, history_path)):
        if json.loads(Path(path).read_text(encoding='utf-8')) != document:
            raise ValueError('正規化データと保存ファイルが一致しません')
    if current.get('guarded_action_contract') != guarded_action_contract():
        raise ValueError('現在画像側のガード契約がありません')
    if 'temporal_rgb_contract' in current or history.get('temporal_rgb_contract') != temporal_rgb_contract():
        raise ValueError('比較する画像履歴契約が不正です')
    current_sha = verify_temporal_dataset_contract(current_gate, current, current_path)
    history_sha = verify_temporal_dataset_contract(history_gate, history, history_path)
    common_history = {key: value for key, value in history.items() if key != 'temporal_rgb_contract'}
    if current != common_history:
        raise ValueError('共通の正規化値または契約が一致しません')
    return {'current_sha256': current_sha, 'history_sha256': history_sha, 'common_metadata_equal': True}


def verify_temporal_training_contract(training, gate, gate_path, normalization, normalization_path):
    normalization_sha256 = verify_temporal_dataset_contract(gate, normalization, normalization_path)
    if normalization.get("temporal_rgb_contract") is None and normalization.get('guarded_action_contract') is None:
        return normalization_sha256
    if training.get("training_completed") is not True:
        raise ValueError("時間履歴モデルの学習が完了していません")
    if training.get("normalization_sha256") != normalization_sha256:
        raise ValueError("学習記録と正規化ファイルのSHAが一致しません")
    if training.get("dataset_gate_sha256") != file_sha256(gate_path):
        raise ValueError("学習記録とデータ出力記録のSHAが一致しません")
    if training.get("language_instruction") != normalization.get("language_instruction"):
        raise ValueError("学習時と実行時の指示文が一致しません")
    if training.get("temporal_rgb_contract") != normalization.get("temporal_rgb_contract"):
        raise ValueError("学習時と実行時の時間履歴契約が一致しません")
    if training.get('guarded_action_contract') != normalization.get('guarded_action_contract'):
        raise ValueError('学習時と実行時のガード付き行動契約が一致しません')
    return normalization_sha256


def verify_worker_normalization(normalization, normalization_path, expected_sha256, *, guarded_requests_only=False):
    guard=normalization.get('guarded_action_contract')
    if guard is not None:
        if (not guarded_requests_only or guard!=guarded_action_contract()
                or normalization.get('observation_history_contract')!=command_observation_contract()
                or not expected_sha256):
            raise ValueError('ガード付き履歴はガードへ渡す要求専用の推論と正規化SHAが必要です')
    elif guarded_requests_only:
        raise ValueError('要求専用のガード契約がありません')
    actual_sha256 = file_sha256(normalization_path)
    if normalization.get("temporal_rgb_contract") is not None and not expected_sha256:
        raise ValueError("時間履歴推論には学習時の正規化SHAが必要です")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("推論workerと学習時の正規化SHAが一致しません")
    return actual_sha256
