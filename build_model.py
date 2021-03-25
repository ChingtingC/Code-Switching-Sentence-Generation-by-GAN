from keras.preprocessing.text import Tokenizer
from keras.regularizers import *
from keras.activations import *
from keras.constraints import *
from keras.optimizers import *
from keras.layers import Input, Embedding, concatenate, Flatten, Permute, multiply, Masking
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.normalization import *
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from argparse import ArgumentParser

POS_SIZE = 64

## Define word index dict
word_index = dict()
with open("local/sample/dict.txt", "r") as word_dict:
    idx = 1
    for line in word_dict:
        line = line.strip()
        word_index[line] = idx
        idx = idx + 1

tokenizer = Tokenizer(num_words = None, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower = True, split = " ", char_level = False)
tokenizer.word_index = word_index
vocabulary_size = len(word_index)

class GAN:
    def __init__(self, MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE, EMBEDDING_POS, NOISE_SIZE,
	             HIDDEN_SIZE_L, HIDDEN_SIZE_G, HIDDEN_SIZE_D, DROPOUT_RATE):
        self.opt = Adam(lr = 1e-4,  decay = .0, clipvalue = 10.)
        self.dopt = Adam(lr = 1e-3, decay = .0, clipvalue = 10.)
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.EMBEDDING_POS = EMBEDDING_POS
        self.NOISE_SIZE = NOISE_SIZE
        self.HIDDEN_SIZE_L = HIDDEN_SIZE_L
        self.HIDDEN_SIZE_G = HIDDEN_SIZE_G
        self.HIDDEN_SIZE_D = HIDDEN_SIZE_D
        self.DROPOUT_RATE = DROPOUT_RATE
        if EMBEDDING_POS <= 0:
            self.WORD_ONLY = True
        else:
            self.WORD_ONLY = False

        ## Shared embedding layer
        self.embedding_word = Embedding(input_dim = (vocabulary_size + 1),
		                                output_dim = self.EMBEDDING_SIZE,
										name = "word_embed")
        if not self.WORD_ONLY:
            self.embedding_pos = Embedding(input_dim = (POS_SIZE + 1),
			                               output_dim = self.EMBEDDING_POS,
										   name = "pos_embed")
        self.g_bi = Bidirectional(LSTM(units = self.HIDDEN_SIZE_L, return_sequences = True))
        self.g_d1 = Dense(self.HIDDEN_SIZE_G, name = "generator_hidden")
        self.g_d2 = Dense(2, activation = "softmax", name = "generator_output")

        ####### Build Model #######
        self.generator = self._build_g()
        self.discriminator = self._build_d()

    def _build_g(self):
        ## input emb
        g_input = Input(shape = [self.MAX_SEQUENCE_LENGTH])
        if not self.WORD_ONLY:
            g_pos = Input(shape = [self.MAX_SEQUENCE_LENGTH])
            wH = self.embedding_word(g_input)
            pH = self.embedding_pos(g_pos)
            H = concatenate([wH, pH], axis = 2)
        else:
            H = self.embedding_word(g_input)
        ## concate emb vector
        H = Masking(0.)(H)
        H = self.g_bi(H)
        ## noise
        nH = Input(shape = [self.MAX_SEQUENCE_LENGTH, self.NOISE_SIZE])
        ## concate emb vector and noise
        H = concatenate([nH, H], axis = 2)
        H = self.g_d1(H)
        H = LeakyReLU(.1)(H)
        g_V = self.g_d2(H)

        reward = Input(shape=[1], name="Reward_input")
        def reward_loss(one_hot_action, action_prob):
            action_prob = K.sum(action_prob * one_hot_action, axis = 2)
            log_action_prob = K.log(action_prob)
            loss = - K.mean(log_action_prob * reward)
            return loss

        if not self.WORD_ONLY:
            generator = Model(inputs = [g_input, g_pos, nH, reward], outputs = [g_V])
        else:
            generator = Model(inputs = [g_input, nH, reward], outputs = [g_V])
        generator.compile(loss = reward_loss, optimizer = self.opt)
        return generator

    def _build_d(self):
        d_input = Input(shape = [self.MAX_SEQUENCE_LENGTH])
        if not self.WORD_ONLY:
            d_pos = Input(shape = [self.MAX_SEQUENCE_LENGTH])
            wH = self.embedding_word(d_input)
            pH = self.embedding_pos(d_pos)
            H = concatenate([wH, pH], axis = 2)
        else:
            H = self.embedding_word(d_input)
        H = self.g_bi(H)
        A = Permute((2, 1))(H)
        A = Dense(self.MAX_SEQUENCE_LENGTH, activation = 'softmax')(A)
        A_probs = Permute((2, 1), name = 'attention_vec')(A)
        H = multiply([H, A_probs])
        H = Flatten()(H)
        H = Dense(self.HIDDEN_SIZE_D, name = "discriminator_hidden")(H)
        H = LeakyReLU(.1)(H)
        H = Dropout(self.DROPOUT_RATE)(H)
        H = Dense(1)(H)
        d_V = Activation('sigmoid')(H)
        if not self.WORD_ONLY:
            discriminator = Model(inputs = [d_input, d_pos], outputs = d_V)
        else:
            discriminator = Model(inputs = [d_input], outputs = d_V)
        discriminator.compile(loss='binary_crossentropy', optimizer=self.dopt)
        return discriminator


def main():
    # Add arguement
    parser = ArgumentParser()
    parser.add_argument("-l", dest = "MAX_SEQUENCE_LENGTH", type = int, default = 30)
    parser.add_argument("-e", dest = "EMBEDDING_SIZE",      type = int, default = 150)
    parser.add_argument("-p", dest = "EMBEDDING_POS",       type = int, default = 20)
    parser.add_argument("-n", dest = "NOISE_SIZE",          type = int, default = 0)
    parser.add_argument("-L", dest = "HIDDEN_SIZE_L",       type = int, default = 16)
    parser.add_argument("-G", dest = "HIDDEN_SIZE_G",       type = int, default = 16)
    parser.add_argument("-D", dest = "HIDDEN_SIZE_D",       type = int, default = 32)
    parser.add_argument("-d", dest = "DROPOUT_RATE",      type = float, default = 0.1)
    args = parser.parse_args()

    ## initial parameter setting
    MAX_SEQUENCE_LENGTH = args.MAX_SEQUENCE_LENGTH
    EMBEDDING_SIZE = args.EMBEDDING_SIZE
    EMBEDDING_POS = args.EMBEDDING_POS
    NOISE_SIZE = args.NOISE_SIZE
    HIDDEN_SIZE_L = args.HIDDEN_SIZE_L
    HIDDEN_SIZE_G = args.HIDDEN_SIZE_G
    HIDDEN_SIZE_D = args.HIDDEN_SIZE_D
    DROPOUT_RATE = args.DROPOUT_RATE
    model = GAN(MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE, EMBEDDING_POS, NOISE_SIZE,
	              HIDDEN_SIZE_L, HIDDEN_SIZE_G, HIDDEN_SIZE_D, DROPOUT_RATE)

    #### Build Generative model ...
    generator = model.generator
    generator.summary()

    #### Build Discriminative model ...
    discriminator = model.discriminator
    discriminator.summary()

if __name__ == "__main__":
    main()
