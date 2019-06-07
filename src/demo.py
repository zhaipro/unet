import time

import cv2
import keras
import numpy as np
import scipy


def predict(model, im):
    h, w, _ = im.shape
    inputs = cv2.resize(im, (480, 480))
    inputs = inputs.astype('float32')
    inputs.shape = (1,) + inputs.shape
    inputs = inputs / 255
    mask = model.predict(inputs)
    mask.shape = mask.shape[1:]
    mask = cv2.resize(mask, (w, h))
    mask.shape = h, w, 1
    return mask


def recolor(im, mask, color):
    color = np.array(color)
    mean = (im.max(axis=2, keepdims=True) * mask).sum() / mask.sum()
    f = scipy.interpolate.interp1d([0, mean, 255], [0, color.max(), 255])
    f = f(np.arange(256)).astype('uint8')
    color = color / color.max()
    color.shape = 1, 1, 3
    im = im * (1 - mask) + f[im.max(axis=2, keepdims=True)] * color * mask
    return im


def main(model, ifn, ofn, color=(0x40, 0x16, 0x66)):
    if isinstance(model, str):
        model = keras.models.load_model(model, compile=False)
    im = cv2.imread(ifn)
    start = time.perf_counter()
    mask = predict(model, im)
    print(time.perf_counter() - start)
    start = time.perf_counter()
    im = recolor(im, mask, color)
    print(time.perf_counter() - start)
    cv2.imwrite(ofn, im)
    cv2.imwrite('mask.jpg', mask * 255)


if __name__ == '__main__':
    # data = np.load('../celeba.npz')
    # images, masks = data['images'], data['masks']
    # cv2.imwrite('celeba.image.123.jpg', images[123])
    # cv2.imwrite('celeba.mark.123.jpg', masks[123])
    main('weights.050.h5', './14.jpg', 'h.14.jpg')
