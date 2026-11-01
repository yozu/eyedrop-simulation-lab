"""診断時だけ開眼判定を固定候補と因果的確認に置き換える。"""
from live_tree_eye_runtime_20261101 import OpenConfirmation


def install_tree_eye_override(observer,runtime):
    original=observer.predict;confirmation=OpenConfirmation()
    def predict(left,right,*,native_face=None):
        if native_face is None:raise ValueError('眼判定には顔画像が必要です')
        result=original(left,right,native_face=native_face)
        score=runtime.score(native_face);raw=score>=.5;opened=confirmation.update(raw)
        result['openness']=1. if opened else 0.
        result['tree_eye_score']=score;result['tree_eye_raw']=raw;result['tree_eye_confirmed']=opened
        return result
    observer.predict=predict
