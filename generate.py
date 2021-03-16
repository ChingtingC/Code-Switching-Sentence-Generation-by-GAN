#!/usr/bin/env python
# encoding: utf-8

import random
import codecs
import time
import os
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
parser.add_argument("-I", dest = "INPUT_POS",                      default = "corpus/sample/pos/test.mono.txt")
parser.add_argument("-o", dest = "OUTPUT_TEXT",                     default = None)
args = parser.parse_args()


## Initial parameter setting
RANDOM_SEED = args.RANDOM_SEED
MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH
EMBEDDING_SIZE = args.EMBEDDING_SIZE
EMBEDDING_POS = args.EMBEDDING_POS
NOISE_SIZE = args.NOISE_SIZE
HIDDEN_SIZE_G = args.HIDDEN_SIZE_G
HIDDEN_SIZE_L = args.HIDDEN_SIZE_L
HIDDEN_SIZE_D = args.HIDDEN_SIZE_D
DROPOUT_RATE = args.DROPOUT_RATE
MODEL_PATH = args.MODEL_PATH
SAVENAME = args.SAVENAME
INPUT_TEXT = args.INPUT_TEXT
INPUT_POS = args.INPUT_POS
OUTPUT_TEXT = args.OUTPUT_TEXT


## Initial declaration
np.random.seed(RANDOM_SEED)
text_zh = []
word_index = dict()

if EMBEDDING_POS <= 0:
    WORD_ONLY = True
else:
    WORD_ONLY = False

if not WORD_ONLY:
    pos_seq_zh = []
    postag = dict()


if MODEL_PATH[-1] != "/":
    MODEL_PATH = MODEL_PATH + "/"


try:
    os.stat("exp")
except:
    os.mkdir("exp")


LABEL_FILE = open("exp/label." + SAVENAME, "w")

if not OUTPUT_TEXT:
    TEXT_FILE = open("exp/cs." + SAVENAME, "w")


## Define Pos tag dict
if not WORD_ONLY:
    with open("local/postag", "r") as pos_dict:
        idx = 1
        for line in pos_dict:
            line = line.strip()
            postag[line] = idx
            idx = idx + 1

## Define word index dict
with open("local/dict", "r") as word_dict:
    idx = 1
    for line in word_dict:
        line = line.strip()
        word_index[line] = idx
        idx = idx + 1


## Load chinese data
with open(INPUT_TEXT, "r") as input_data:
    for line in input_data:
        text_zh.append(line.strip())


## Load pos tagging
if not WORD_ONLY:
    with open(INPUT_POS, "r") as input_data:
        for line in input_data:
            line = line.strip().split(' ')
            pos_seq_zh.append(line)

            
## Embed and zero pad data
tokenizer = Tokenizer(num_words = None, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = False, split = " ", char_level = False)
tokenizer.word_index = word_index

sequences_zh = tokenizer.texts_to_sequences(text_zh)
emb_zh = np.asarray(pad_sequences(sequences_zh, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post', truncating = 'post', value = 0))

if not WORD_ONLY:
    pos_seq_zh = pad_sequences(pos_seq_zh, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post', truncating = 'post', value = 0)


####### Build Model #######
model = GANmodel(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE, EMBEDDING_POS, NOISE_SIZE, HIDDEN_SIZE_L, HIDDEN_SIZE_G, HIDDEN_SIZE_D, DROPOUT_RATE)

#### Build Generative model ...
generator = model.generator

### Load model ###
generator.load_weights(MODEL_PATH + "gen.mdl")

noise_g = np.random.normal(0, 1, size = (emb_zh.shape[0], MAX_SEQUENCE_LENGTH, NOISE_SIZE))
reward = np.zeros((emb_zh.shape[0], 1))
if not WORD_ONLY:
    output_g = generator.predict([emb_zh, pos_seq_zh, noise_g, reward])
else:
    output_g = generator.predict([emb_zh, noise_g, reward])

action_g = max_action(output_g)
emb_g = translate(emb_zh, action_g)
text_g = translate_output(emb_zh, action_g)

for count, sen in enumerate(text_g):
    TEXT_FILE.write(sen)
    TEXT_FILE.write("\n")
    for i in output_g[count]:
        LABEL_FILE.write("%d " % np.argmax(i))
    LABEL_FILE.write("\n")

TEXT_FILE.close()
LABEL_FILE.close()
