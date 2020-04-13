#!/usr/bin/env python
'''
Blind sound source separation of multiple speakers on a single channel with GAN.
'''
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Multiply, Add, Lambda, Conv2D, Flatten, BatchNormalization, concatenate
import librosa
from scipy import signal
import yaml


class MelCNN(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        self.filter_size = 2
        self.img_rows = 129
        self.img_columns = 1
        self.a_channel = 1
        self.r_channels = 64
        self.s_channels = 256
        self.d_channels = 128
        self.n_loop = 4
        self.n_layer = 3
        self.dilation = [2 ** i for i in range(self.n_layer)] * self.n_loop

        self.config = yaml.load(open('config/wave.yml'), Loader=yaml.SafeLoader)

        self.model_path = self.config['path']['model']
        self.model = self.build_nn()


    def ResidualBlock(self, block_in, dilation_index):
        res = block_in
        tanh_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
                          dilation_rate=(dilation_index, 1), activation='tanh')(block_in)
        sigm_out = Conv2D(self.d_channels, (self.filter_size, 1), padding='same',
                          dilation_rate=(dilation_index, 1), activation='sigmoid')(block_in)
        marged = Multiply()([tanh_out, sigm_out])
        res_out = Conv2D(self.r_channels, (1, 1), padding='same')(marged)
        skip_out = Conv2D(self.s_channels, (1, 1), padding='same')(marged)
        res_out = Add()([res_out, res])

        return res_out, skip_out


    def ResidualNet(self, block_in):
        skip_out_list = []
        for dilation_index in self.dilation:
            res_out, skip_out = self.ResidualBlock(block_in, dilation_index)
            skip_out_list.append(skip_out)
            block_in = res_out

        return skip_out_list


    def build_nn(self):
        input1 = Input(shape=(self.img_rows, self.img_columns, self.a_channel))
        input2 = Input(shape=(1))

        x1 = Dense(32)(input1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x2 = Dense(32)(input2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        merge = Add()([x1, x2])

        causal_conv = Conv2D(self.r_channels, (self.filter_size, 1), padding='same')(merge)
        skip_out_list = self.ResidualNet(causal_conv)
        skip_out = Add()(skip_out_list)
        skip_out = Activation('relu')(skip_out)
        skip_out = Conv2D(self.a_channel, (1, 1), padding='same', activation='relu')(skip_out)
        prediction = Conv2D(self.a_channel, (1, 1), padding='same')(skip_out)
        prediction = Flatten()(prediction)
        #prediction = Dense(self.img_rows, activation='sigmoid')(prediction)
        prediction = Dense(4, activation='softmax')(prediction)

        # モデル定義とコンパイル
        model = Model([input1, input2], prediction)

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            #loss='binary_crossentropy',
            metrics=['accuracy']
        )

        #model.summary()

        return model


    def train(self, x, y, epochs=200, batch_size=256):
        ## -----*----- 学習 -----*----- ##
        n_term = 10
        for step in range(epochs // n_term):
            self.model.fit(
                x, y,
                initial_epoch=step * n_term,
                epochs=(step + 1) * n_term,
                batch_size=batch_size,
                validation_split=0.2
            )
            self.model.save_weights(self.model_path.replace('.', '_{0}.'.format((step + 1))))

        # 最終の学習モデルを保存
        self.model.save_weights(self.model_path)


    def load_model(self):
        ## -----*----- 学習済みモデルを読み込み -----*----- ##
        self.model.load_weights(self.config['path']['model'])


    def vocoder(self, wav, to_int=True):
        ## -----*----- 音声を生成 -----*----- ##
        spec = stft(wav, self.config['wave']['fs'], False).T
        mask = []

        # マスク推定
        for i, row in enumerate(spec):
            row = row.reshape((1, row.shape[0], 1, 1))
            index = np.array([i])
            mask.append(self.model.predict([row, index]))

        mask = np.reshape(mask, (spec.shape[0], spec.shape[1]))

        for i, row in enumerate(mask):
            spec[i] *= np.round(row)

        # 音声に戻す
        wav = istft(spec.T, self.config['wave']['fs'], to_int)

        return wav


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


def stft(x, fs, to_log=True):
    ## -----*----- 短時間フーリエ変換 -----*----- ##
    spec = signal.stft(x, fs=fs, nperseg=256)[2]

    if to_log:
        spec = np.where(spec == 0, 0.1 ** 10, spec)
        spec = np.log10(np.abs(spec))
        for i in range(spec.shape[0]):
            for j in range(spec.shape[1]):
                if spec[i][j] < 0.0:
                    spec[i][j] = 0.0

    return spec


def istft(spec, fs, to_int=True):
    ## -----*----- 逆短時間フーリエ変換 -----*----- ##
    wav = signal.istft(spec, fs=fs, nperseg=256)[1]

    if to_int:
        wav = np.array(wav, dtype='int16')

    return wav
