#!/usr/bin/env python
'''
Blind sound source separation of multiple speakers on a single channel with GAN.
'''
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Multiply, Add, Lambda, Conv2D, Flatten
import librosa
from scipy import signal


class MelCNN(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.filter_size = 2
        self.img_rows = 8000
        self.img_columns = 1
        #self.a_channel = 256
        self.a_channel = 1
        self.r_channels = 64
        self.s_channels = 256
        self.d_channels = 128
        self.n_loop = 4
        self.n_layer = 10
        self.dilation = [2 ** i for i in range(10)] * 4

        self.model = self.wavenet()


    def ResidualBlock(self, block_in, dilation_index):
        res = block_in
        tanh_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
                          dilation_rate=(dilation_index, 1), activation='tanh')(block_in)
        sigm_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
                          dilation_rate=(dilation_index, 1), activation='sigmoid')(block_in)
        marged = Multiply()([tanh_out, sigm_out])
        res_out = Conv2D(self.r_channels, (1,1), padding='same')(marged)
        skip_out = Conv2D(self.s_channels, (1,1), padding='same')(marged)
        res_out = Add()([res_out,res])

        return res_out, skip_out


    def ResidualNet(self, block_in):
        skip_out_list = []
        for dilation_index in self.dilation:
            res_out, skip_out = self.ResidualBlock(block_in, dilation_index)
            skip_out_list.append(skip_out)
            block_in = res_out

        return skip_out_list


    def wavenet(self):
        inputs = Input(shape=(self.img_rows, self.img_columns, self.a_channel))
        causal_conv = Conv2D(self.r_channels, (self.filter_size, 1), padding='same')(inputs)
        skip_out_list = self.ResidualNet(causal_conv)
        skip_out = Add()(skip_out_list)
        skip_out = Activation('relu')(skip_out)
        skip_out = Conv2D(self.a_channel, (1,1), padding='same', activation='relu')(skip_out)
        prediction = Conv2D(self.a_channel, (1,1), padding='same')(skip_out)
        prediction = Flatten()(prediction)
        prediction = Dense(8000, activation='softmax')(prediction)

        model_wavenet = Model(inputs, prediction)

        model_wavenet.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model_wavenet.summary()

        return model_wavenet


    def train(self, x, y, epochs=200, batch_size=16):
        ## -----*----- 学習 -----*----- ##
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

        self.model.save_weights('model/model.h5')



def transform(x, mu=256):
    ## -----*----- μ-law変換 -----*----- ##
    x = x.astype(np.float32)
    y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
    return y.astype(np.int32)


def itransform(y, mu=256):
    ## -----*----- 逆μ-law変換 -----*----- ##
    y = y.astype(np.float32)
    y = 2 * y / mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x.astype(np.float32)


def to_spec(x, fs, to_log=True):
    ## -----*----- スペクトログラム -----*----- ##
    spec = signal.stft(x, fs=fs, nperseg=256)[2]

    if to_log:
        spec = np.where(spec == 0, 0.1 ** 10, spec)
        spec = np.log10(np.abs(spec))

    return spec

