# -*- coding: utf-8 -*-
import numpy as np
import sklearn
import librosa
import yaml
from glob import glob
from tqdm import tqdm


def transform(x, mu=256):
        x = x.astype(np.float32)
        y = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
        y = np.digitize(y, 2 * np.arange(mu) / mu - 1) - 1
        return y.astype(np.int32)

def itransform(y, mu=256):
        y = y.astype(np.float32)
        y = 2 * y / mu - 1
        x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
        return x.astype(np.float32)


if __name__ == '__main__':
    config = yaml.load(open('config/wave.yml'))
    print(config)

    target_path = config['path']['target']
    others_path = config['path']['others']

    for file in tqdm(glob(target_path+'*.wav')):
        wav, _ = librosa.load(file, sr=config['wave']['fs'])
        # 無音区間（20dB以下）を除去
        wav, _ = librosa.effects.trim(wav, top_db=20)
        # 正規化
        wav /= np.abs(wav).max()

        print(transform(wav))
        print(sklearn.preprocessing.minmax_scale(transform(wav)))