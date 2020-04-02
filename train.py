#!/usr/bin/env python
import numpy as np
import yaml
from glob import glob
import sklearn
import librosa
from scipy import signal
from melcnn import *


CONFIG = yaml.load(open('config/wave.yml'))
SIZE = int(CONFIG['wave']['fs'] * CONFIG['wave']['sec'])


def build_wave(files):
    ## -----*----- 固定長ベクトルに変換 -----*----- ##
    ret = []

    for file in files:
        # 音声読み込み
        wav, _ = librosa.load(file, sr=CONFIG['wave']['fs'])
        # 無音区間（20dB以下）を除去
        wav, _ = librosa.effects.trim(wav, top_db=20)
        # 最小値０，最大値１に正規化
        #wav = sklearn.preprocessing.minmax_scale(wav)
        # μ-law変換
        wav = transform(wav)
        # ゼロパディング
        n = SIZE - wav.shape[0] % SIZE
        wav = np.append(wav, np.zeros(n))
        # 固定長に分割
        wav = np.split(wav, wav.shape[0] / SIZE)

        for w in wav:  ret.append(w)

    return ret


if __name__ == '__main__':
    # ファイル一覧を取得
    target_files = glob(CONFIG['path']['target'] + '*')
    others_files = glob(CONFIG['path']['others'] + '*')

    # ファイルを全て固定長のベクトルに変換
    target_waves = build_wave(target_files)
    others_waves = build_wave(others_files)

    # データ個数（固定長ベクトル）
    n_data = max([len(target_waves), len(others_waves)])

    # 入力，正解ラベル
    x, y = [], []

    for i in range(n_data):
        w1 = target_waves[i % len(target_waves)]
        w2 = others_waves[i % len(others_waves)]
        # 合成して正規化
        mixed = w1 + w2
        #mixed = sklearn.preprocessing.minmax_scale(mixed)

        x.append(mixed)
        y.append(w1)

    x = np.array(x)
    x = x.reshape((x.shape[0], SIZE, 1, 1))
    y = np.array(y, dtype=np.int8)

    melcnn = MelCNN(SIZE)
    melcnn.train(x, y)

