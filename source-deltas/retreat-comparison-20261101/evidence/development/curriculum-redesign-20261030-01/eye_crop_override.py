"""開眼判定の切り出し位置だけを指定した顔特徴へ切り替える。"""


def install_eye_crop_override(observer, encoder, ready_from_image):
    original = observer.predict

    def predict(left, right, *, native_face=None):
        if native_face is None:
            raise ValueError('眼の判定には顔画像が必要です')
        prediction = original(left, right, native_face=native_face)
        features = encoder.extract(native_face).cpu().numpy()
        prediction['openness'] = ready_from_image(
            observer.ready_head, native_face, features, observer.face.device
        )
        return prediction

    observer.predict = predict
