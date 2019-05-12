import cv2
import keras
import numpy as np


def predict(model, ifn, ofn):
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    im = cv2.imread(ifn)
    h, w, _ = im.shape
    inputs = cv2.resize(im, (512, 512))
    inputs = inputs.astype('float32')
    inputs.shape = (1,) + inputs.shape
    inputs = inputs / 127.5 - 1
    mask = model.predict(inputs)
    mask.shape = mask.shape[1:]
    mask = cv2.resize(mask, (w, h))
    mask.shape = h, w, 1
    color = np.array([0x80, 0x2b, 0xcb])
    color.shape = 1, 1, 3
    im = im * (1 - mask) + im.max(axis=2, keepdims=True) / 255 * mask * color
    cv2.imwrite(ofn, im)


if __name__ == '__main__':
    predict('weights.002.h5', 'a.jpg', 'a.a.jpg')
