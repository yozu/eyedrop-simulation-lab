"""保存系列の画像から算出した特徴を候補ごとに揃える。"""
import numpy as np


def combine_live_features(old,new,old_logits,new_logits,embeddings):
    n=len(old)
    if old.shape!=(n,8) or new.shape!=(n,8) or old_logits.shape!=(n,6) or new_logits.shape!=(n,4) or embeddings.shape!=(n,256):
        raise ValueError('モデル入力の形が違います')
    probability=1/(1+np.exp(-np.clip(old_logits[:,:2].mean(axis=1,keepdims=True),-60,60)))
    dual=np.concatenate((old,probability,old_logits[:,2:],new,new_logits),axis=1)
    eye=np.concatenate((new,embeddings),axis=1)
    if not np.isfinite(dual).all() or not np.isfinite(eye).all():raise ValueError('モデル入力が非有限です')
    return dual,eye
