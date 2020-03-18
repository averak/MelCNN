#!/usr/bin/env python
import os
import numpy as np
import sklearn
import librosa
import yaml
from glob import glob
from tqdm import tqdm


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


def build_wave(files, config):
    ## -----*----- 固定長ベクトルに変換 -----*----- ##
    ret = []

    for file in tqdm(files):
        # 音声読み込み
        wav, _ = librosa.load(file, sr=config['wave']['fs'])
        # 無音区間（20dB以下）を除去
        wav, _ = librosa.effects.trim(wav, top_db=20)
        # 最小値０，最大値１に正規化
        #wav = sklearn.preprocessing.minmax_scale(wav)
        # μ-law変換
        wav = transform(wav)
        # ゼロパディング
        n = 8000 - wav.shape[0] % config['wave']['fs']
        wav = np.append(wav, np.zeros(n))
        # 固定長に分割
        wav = np.split(wav, wav.shape[0] / config['wave']['fs'])

        for w in wav:  ret.append(w)

    return ret



if __name__ == '__main__':
    config = yaml.load(open('config/wave.yml'))

    # ファイル一覧を取得
    target_files = glob(config['path']['target'] + '*')
    others_files = glob(config['path']['others'] + '*')

    # ファイルを全て固定長のベクトルに変換
    target_waves = build_wave(target_files, config)
    others_waves = build_wave(others_files, config)

    # データ個数（固定長ベクトル）
    n_data = max([len(target_waves), len(others_waves)])

    # 入力，正解ラベル
    x, y = [], []

    for i in range(n_data):
        w1 = target_waves[i % len(target_waves)]
        w2 = others_waves[i % len(others_waves)]
        # 合成して正規化
        mixed = sklearn.preprocessing.minmax_scale(w1 + w2)

        x.append(mixed)
        y.append(w1)

    x = np.array(x)
    y = np.array(y, dtype=np.int8)
