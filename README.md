MelCNN
------

[![build](https://github.com/AjxLab/MelCNN/workflows/build/badge.svg)](https://github.com/AjxLab/MelCNN/actions)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Blind sound source separation of multiple speakers on a single channel with Dilated Convolution.

## Description
The purpose of this project is to realize end-to-end blind sound source separation.<br>
Develop auto encoder using Dilated Convolution.


## Requiremenst
* Ubuntu 18.04
* Python 3.7
* TensorFlow 2


## Usage
### Prepare teacher data
1. Record audio(target/others) with a sampling rate of 8000 Hz
2. Place them in data/(target/others)
### Train
```sh
$ ./train.py
```


## Installation
```sh
# clone
$ git clone <this repo>
$ cd <this repo>

# create venv
$ python -m venv venv
$ source source venv/bin/activate

# install python libs
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Author
* [Tatsuya Abe](https://github.com/AjxLab)
* ```abe12<at>mccc.jp```
