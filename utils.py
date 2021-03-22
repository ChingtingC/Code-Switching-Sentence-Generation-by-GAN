# encoding: utf-8
import codecs
import numpy as np
import tensorflow as tf
import jieba.posseg as pseg
from tqdm import tqdm
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

## initial parameter setting
POS_SIZE = 64

postag = dict()
word_index = dict()
index_to_word = dict()
index_to_word[0] = ''
translation = dict()
translation['']=''

## Define Pos tag dict
with open("local/postag.txt", "r") as pos_dict:
    index = 1
    for line in pos_dict:
        line = line.strip()
        postag[line] = index
        index = index + 1

## Define word index dict
with open("local/sample/dict.txt", "r") as word_dict:
    index=1
    for line in word_dict:
        line = line.strip()
        word_index[line] = index
        index = index + 1

with codecs.open("local/sample/dict.txt", "r", encoding = 'utf-8') as word_dict:
    index = 1
    for line in word_dict:
        line = line.strip()
        index_to_word[index] = line
        index = index + 1

## Build translator ...
with codecs.open("local/sample/translator.txt", "r", encoding = 'utf-8') as _:
    for line in _:
        line = line.strip().split(' ')
        translation[line[0]] = ' '.join(line[1:])

tokenizer = Tokenizer(num_words = None, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, split = " ", char_level = False)
#tokenizer.fit_on_texts(text)
tokenizer.word_index = word_index

## Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def max_action(action_prob):
    action = []
    for action_prob_seq in action_prob:
        action_list = []
        for action_prob in action_prob_seq:
            action_list.append(np.argmax(action_prob))
        action.append(action_list)
    return action


def get_action(action_prob_batch):
    action_batch = []
    action_one_hot_batch = []
    for action_prob_seq in action_prob_batch:
        action_list = []
        action_one_hot_list = []
        for action_prob in action_prob_seq:
            action = np.random.choice(2, p = action_prob)
            action_list.append(action)
            action_one_hot = np_utils.to_categorical(action, num_classes = 2)
            action_one_hot_list.append(np.transpose(action_one_hot))
        action_batch.append(action_list)
        action_one_hot_batch.append(action_one_hot_list)
    action_one_hot_batch = np.asarray(action_one_hot_batch)
    return action_batch, action_one_hot_batch


## Use emb and G's action to translated embedding
def translate(text, action):
    text_new_all = []
    text_new = []
    MAX_SEQUENCE_LENGTH = len(action[0])
    for count,_ in enumerate(text):
        text_new = [index_to_word[i] for i in _]
        for index,ii in enumerate(action[count]):
            if (ii) == 1:
                if text_new[index] in translation:
                    text_new[index] = translation[(text_new[index])]
        temp = ''
        for ii in text_new:
           if len(temp) == 0:
               temp = temp + ii
           else:
               temp = temp + " " +  ii
        text_new_all.append(temp)
    seq_new =  tokenizer.texts_to_sequences(text_new_all)
    emb_new = pad_sequences(seq_new, maxlen = MAX_SEQUENCE_LENGTH, padding = 'post',
                            truncating = 'post', value = 0)
    return np.asarray(emb_new)


## Use emb and G's action to translated text
def translate_output(text, action):
    text_new_all = []
    text_new = []
    for count,_ in enumerate(text):
        text_new = [index_to_word[i] for i in _]
        for index,ii in enumerate(action[count]):
            if (ii) == 1:
                if text_new[index] in translation:
                    text_new[index] = translation[(text_new[index])]
        temp = ''
        for ii in text_new:
            if len(ii) == 0:
                break
            else:
                temp = temp + ii + " "
        text_new_all.append(temp)
    return text_new_all


def translate_output2(text, action):
    text_new_all = []
    text_new = []
    for count,_ in enumerate(text):
        text_new = [index_to_word[i] for i in _]
        for index, ii in enumerate(action[count]):
            if ii >= 0.5:
                if text_new[index] in translation:
                    text_new[index] = translation[(text_new[index])]
        temp = ''
        for ii in text_new:
            if len(ii) == 0:
                break
            else:
                temp = temp + ii + " "
        text_new_all.append(temp)
    return text_new_all


def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def evaluate_acc(y_real, y_pred, filename):
    tn = 0.
    fn = 0.
    tp = 0.
    fp = 0.
    filename.write("Predict %d true, %d false\n" % (sum(y_pred), (len(y_pred) - sum(y_pred))))
    for id, label in enumerate(y_real):
        if y_pred[id] == 0:
            if label == 0:
                tn = tn + 1
            else:
                fn = fn + 1
        else:
            if label == 1:
                tp = tp + 1
            else:
                fp = fp + 1
    filename.write("tp %d, tn %d, fp %d, fn %d\n" % (tp, tn, fp, fn))
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 1 / (1 / precision + 1 / recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    filename.write("precision = %.2f, recall = %.2f, accuracy = %.2f, f1 = %.2f \n\n"
                    % (precision, recall, accuracy, f1))
