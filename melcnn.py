# -*- coding: utf-8 -*-
'''
Blind sound source separation of multiple speakers on a single channel with GAN.
'''
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Reshape, UpSampling2D


class MelCNN(object):
    def __init__(self):
        ## -----*----- コンストラクタ -----*----- ##
        return
