#!/usr/bin/env python
# encoding: utf-8

import random
import codecs
import time
import os
import sklearn.preprocessing
import jieba.posseg as pseg
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils import make_trainable, translate, translate_output, write_log, str2bool, get_action, evaluate_acc#, plot_loss
from build_model import GAN


# Add arguement
parser = ArgumentParser()
parser.add_argument("-l", dest = "MAX_SEQUENCE_LENGTH", type = int, default = 30)
parser.add_argument("-B", dest = "BATCH_SIZE",          type = int, default = 32)
parser.add_argument("-E", dest = "EPOCH_NUMBER",        type = int, default = 100)
parser.add_argument("-e", dest = "EMBEDDING_SIZE",      type = int, default = 150)
parser.add_argument("-p", dest = "EMBEDDING_POS",       type = int, default = -1) # No defining -> word only
parser.add_argument("-n", dest = "NOISE_SIZE",          type = int, default = 0)
parser.add_argument("-L", dest = "HIDDEN_SIZE_L",       type = int, default = 16)
parser.add_argument("-G", dest = "HIDDEN_SIZE_G",       type = int, default = 16)
parser.add_argument("-D", dest = "HIDDEN_SIZE_D",       type = int, default = 32)
parser.add_argument("-d", dest = "DROPOUT_RATE",      type = float, default = 0.1)
parser.add_argument("-t", dest = "GOPT",              type = float, default = 1e-3)
parser.add_argument("-T", dest = "DOPT",              type = float, default = 1e-2)
parser.add_argument("-m", dest = "MODEL_PATH",                      default = "model/")
parser.add_argument("-c", dest = "CORPUS_NAME",                     default = "sample")
args = parser.parse_args()

## initial parameter setting
MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH
BATCH_SIZE = args.BATCH_SIZE
EPOCH_NUMBER = args.EPOCH_NUMBER
EMBEDDING_SIZE = args.EMBEDDING_SIZE
EMBEDDING_POS = args.EMBEDDING_POS
NOISE_SIZE = args.NOISE_SIZE
HIDDEN_SIZE_G = args.HIDDEN_SIZE_G
HIDDEN_SIZE_L = args.HIDDEN_SIZE_L
HIDDEN_SIZE_D = args.HIDDEN_SIZE_D
DROPOUT_RATE = args.DROPOUT_RATE
MODEL_PATH = args.MODEL_PATH
CORPUS_NAME = args.CORPUS_NAME
gopt = args.GOPT
dopt = args.DOPT
