#!/usr/bin/env python
# encoding: utf-8

import random
import codecs
import time
import os
import sklearn.preprocessing
#import jieba.posseg as pseg
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from keras import backend as K
#from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import translate, translate_output, max_action
from build_model import GAN


# Add arguement
parser = ArgumentParser()
parser.add_argument("-s", dest = "RANDOM_SEED",         type = int, default = 5)
parser.add_argument("-l", dest = "MAX_SEQUENCE_LENGTH", type = int, default = 30)
parser.add_argument("-e", dest = "EMBEDDING_SIZE",      type = int, default = 150)
parser.add_argument("-p", dest = "EMBEDDING_POS",       type = int, default = -1) # No defining -> word only
parser.add_argument("-n", dest = "NOISE_SIZE",          type = int, default = 0)
parser.add_argument("-L", dest = "HIDDEN_SIZE_L",       type = int, default = 16)
parser.add_argument("-G", dest = "HIDDEN_SIZE_G",       type = int, default = 16)
parser.add_argument("-D", dest = "HIDDEN_SIZE_D",       type = int, default = 32)
parser.add_argument("-d", dest = "DROPOUT_RATE",      type = float, default = 0.1)
parser.add_argument("-m", dest = "MODEL_PATH",                      default = "model/")
parser.add_argument("-N", dest = "SAVENAME",                        default = "txt")
parser.add_argument("-i", dest = "INPUT_TEXT",                      default = "corpus/sample/text/test.mono.txt")
parser.add_argument("-o", dest = "OUTPUT_TEXT",                     default = None)
args = parser.parse_args()
