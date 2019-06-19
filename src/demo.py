import time

import cv2
import keras
import numpy as np


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


def recolor(im, mask, color=(0x40, 0x16, 0x66)):
    # 工程化
    color = np.array(color, dtype='float', ndmin=3)
    # 染发
    epsilon = 1e-7
    x = np.max(im, axis=2, keepdims=True)   # 获取亮度
    x_target = np.max(color)
    x = x / 255                             # 数学化
    x_target = x_target / 255
    x_target = -np.log(epsilon + 1 - x_target)
    x_mean = np.sum(-np.log(epsilon + 1 - x)  * mask) / np.sum(mask)
    alpha = x_target / x_mean
    x = 1 - (1 - x) ** alpha
    x = x * 255                             # 二进制化
    im = im * (1 - mask) + (x * mask) * (color / np.max(color))
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
    main('weights.005.h5', './t.jpg', 'i.t.jpg')
