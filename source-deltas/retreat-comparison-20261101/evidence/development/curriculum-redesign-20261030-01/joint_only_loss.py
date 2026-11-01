"""SmolVLAのflow損失から7関節成分だけを評価する。"""
import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

ARM_DIM = 7


def mean_joint_loss(losses, padding):
    if losses.ndim != 3 or losses.shape[-1] < ARM_DIM or padding.shape != losses.shape[:2]:
        raise ValueError('損失と有効時点の形が一致しません')
    valid = (~padding.bool()).to(losses.dtype)
    if not bool(valid.any()):
        raise ValueError('有効な教師指令がありません')
    return (losses[..., :ARM_DIM] * valid.unsqueeze(-1)).sum() / (valid.sum()*ARM_DIM)


def sequence_loss(policy, batch, first_weight):
    if first_weight != 1 or policy.config.chunk_size != 1:
        raise ValueError('1操作の腕損失だけを扱います')
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    action = policy.prepare_action(batch)
    losses = policy.model.forward(images, masks, batch[OBS_LANGUAGE_TOKENS],
                                  batch[OBS_LANGUAGE_ATTENTION_MASK], state, action)
    return mean_joint_loss(losses, batch['action_is_pad'])
