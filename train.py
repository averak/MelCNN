#!/usr/bin/env python
import numpy as np
import yaml
from glob import glob
import sklearn
import librosa
from scipy import signal
from tqdm import tqdm
from melcnn import *


CONFIG = yaml.load(open('config/wave.yml'), Loader=yaml.SafeLoader)
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
    x1, x2, y = [], [], []

    for i in tqdm(range(n_data)):
        w1 = target_waves[i % len(target_waves)]
        w2 = others_waves[i % len(others_waves)]
        spec1 = stft(w1, CONFIG['wave']['fs']).T
        spec2 = stft(w2, CONFIG['wave']['fs']).T

        # 合成
        for i in range(spec1.shape[0]):
            # Noiseの音量を上下
            for snr in [0.25, 0.5, 1.0, 2.0, 4.0]:
                mixed = spec1[i] + (spec2[i] * snr)
                mixed = sklearn.preprocessing.minmax_scale(mixed)

                # バイナリマスク
                mask = []
                for c1, c2 in zip(spec1[i], (spec2[i] * snr)):
                    if c1 > c2:
                        mask.append(1)
                    else:
                        mask.append(0)

                x1.append(mixed)
                x2.append(i)
                y.append(mask)

    x1 = np.array(x1)
    x1 = x1.reshape((x1.shape[0], x1.shape[1], 1, 1))
    x2 = np.array(x2)
    y = np.array(y)

    melcnn = MelCNN()
    melcnn.train([x1, x2], y)

