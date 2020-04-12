#!/usr/bin/env python
import numpy as np
import yaml
from glob import glob
import sklearn
import librosa
from scipy import signal
import scipy.io.wavfile as wf
import librosa
from melcnn import *
from train import *

CONFIG = yaml.load(open('config/wave.yml'), Loader=yaml.SafeLoader)
SIZE = int(CONFIG['wave']['fs'] * CONFIG['wave']['sec'])

# ファイル一覧を取得
target_files = glob(CONFIG['path']['target'] + '*')
others_files = glob(CONFIG['path']['others'] + '*')
# ファイルを全て固定長のベクトルに変換
target_waves = build_wave(target_files)

melcnn = MelCNN()
melcnn.load_model()
w = melcnn.vocoder(target_waves[0])
print(w)
wf.write('data/tmp/分離後.wav', CONFIG['wave']['fs'], w*100)
wf.write('data/tmp/分離前.wav', CONFIG['wave']['fs'], np.array(target_waves[0], dtype=np.int16)*100)
