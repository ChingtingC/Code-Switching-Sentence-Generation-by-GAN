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

## Initial parameter setting
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


## Initial declaration
np.random.seed(0)
text_cs = []
text_zh = []
if not WORD_ONLY:
    pos_seq_cs = []
    pos_seq_zh = []

postag = dict()
word_index = dict()

# Set up loss storage vector
losses = {"d":[], "g":[]}
log_path = './logs/' + MODEL_PATH
callbacks = TensorBoard(log_path)

# Write log
log_g = 'train_loss_g'
log_d = 'train_loss_d'

if MODEL_PATH[-1] is not "/":
    MODEL_PATH = MODEL_PATH + "/"

try:
    os.stat(MODEL_PATH)
except:
    os.mkdir(MODEL_PATH)

try:
    os.stat(log_path)
except:
    os.mkdir(log_path)

if EMBEDDING_POS <= 0:
    WORD_ONLY = True
else:
    WORD_ONLY = False


print("========== LoadING various data")

## Define Pos tag dict
if not WORD_ONLY:
    with open("local/postag.txt", "r") as pos_dict:
        idx = 1
        for line in pos_dict:
            line = line.strip()
            postag[line] = idx
            idx = idx + 1

## Define word index dict
with open("local/dict.txt", "r") as word_dict:
    idx = 1
    for line in word_dict:
        line = line.strip()
        word_index[line] = idx
        idx = idx + 1

## Load code-switching text for training
with open("corpus/" + CORPUS_NAME + "/text/train.cs.txt", "r") as input_data:
    for line in input_data:
        text_cs.append(line.strip())

## Load chinese sentence for training
with open("corpus/" + CORPUS_NAME + "/text/train.mono.txt", "r") as input_data:
     for line in input_data:
         text_zh.append(line.strip())

## Load code-switching pos for training
if not WORD_ONLY:
    with open("corpus/" + CORPUS_NAME + "/pos/train.cs.txt", "r") as input_data:
        for line in input_data:
            line = line.strip().split(' ')
            pos_seq_cs.append(line)

## Load chinese POS for training
if not WORD_ONLY:
    with open("corpus/" + CORPUS_NAME + "/pos/train.mono.txt", "r") as input_data:
        for line in input_data:
            line = line.strip().split(' ')
            pos_seq_zh.append(line)

## Embed and zero pad data
tokenizer = Tokenizer(num_words = None, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = False, split = " ", char_level = False)
tokenizer.word_index = word_index

sequences_cs = tokenizer.texts_to_sequences(text_cs)
sequences_zh = tokenizer.texts_to_sequences(text_zh)

emb_cs = np.asarray(pad_sequences(sequences_cs, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post',
                    truncating = 'post', value = 0))
emb_zh = np.asarray(pad_sequences(sequences_zh, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post',
                    truncating = 'post', value = 0))
