import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, MaxPooling2D, concatenate
from keras.layers import Conv2D, Input, Activation, UpSampling2D, Dropout
from keras.models import Model
from keras.optimizers import RMSprop


def get_unet_512(input_shape=(None, None, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(16, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D()(down0)
    # 128

    down1 = Dropout(0.25)(down0_pool)
    down1 = Conv2D(32, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D()(down1)
    # 64

    down2 = Conv2D(32, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(32, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D()(down2)
    # 32

    down3 = Dropout(0.25)(down2_pool)
    down3 = Conv2D(64, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(64, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D()(down3)
    # 16

    down4 = Conv2D(64, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(64, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D()(down4)
    # 8

    center = Dropout(0.25)(down4_pool)
    center = Conv2D(128, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(128, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D()(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(64, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D()(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(64, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D()(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(32, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(32, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(32, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D()(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D()(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)
    return Model(inputs=inputs, outputs=classify)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    intersection = K.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return score


def psnr(hr, sr, max_val=1):
    mse = K.mean(K.square(hr - sr))
    return 10.0 / np.log(10) * K.log(max_val ** 2 / mse)


def data_generator(path, batch_size=2):
    '''data generator for fit_generator'''
    n = 3000
    i = 0
    idxs = np.arange(n)
    while True:
        x, y = [], []
        for b in range(batch_size):
            if i % n == 0:
                np.random.shuffle(idxs)
            fn = f'{path}/x/{idxs[i]}.jpg'
            im = cv2.imread(fn)
            x.append(im)
            fn = f'{path}/y/{idxs[i]}.jpg'
            im = cv2.imread(fn, 0)
            y.append(im)
            i = (i + 1) % n
        x = np.array(x, dtype='float32') / 255
        y = np.array(y, dtype='float32') / 255
        y.shape = y.shape + (1,)
        yield x, y


model = get_unet_512()
reduce_lr = ReduceLROnPlateau(verbose=1)
model.compile(optimizer=RMSprop(), loss='mean_absolute_error', metrics=[dice_coeff, psnr])
model.fit_generator(data_generator('../dataset', 2),
                    steps_per_epoch=1500,
                    epochs=10,
                    callbacks=[ModelCheckpoint('./weights.{epoch:03d}.h5'), reduce_lr])
model.save('model.h5', include_optimizer=False)
