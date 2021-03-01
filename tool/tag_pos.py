import sys
import jieba
import jieba.posseg as pseg
import numpy as np
import sklearn.preprocessing


if len(sys.argv) < 3:
    print "Usage:", sys.argv[0], "<input_file> <output_file>"
    sys.exit(1)


jieba.add_word("A", freq=None, tag="eng")
jieba.add_word("B", freq=None, tag="eng")
jieba.add_word("C", freq=None, tag="eng")
jieba.add_word("D", freq=None, tag="eng")
jieba.add_word("E", freq=None, tag="eng")
jieba.add_word("F", freq=None, tag="eng")
jieba.add_word("G", freq=None, tag="eng")
jieba.add_word("H", freq=None, tag="eng")
jieba.add_word("I", freq=None, tag="eng")
jieba.add_word("J", freq=None, tag="eng")
jieba.add_word("K", freq=None, tag="eng")
jieba.add_word("L", freq=None, tag="eng")
jieba.add_word("M", freq=None, tag="eng")
jieba.add_word("N", freq=None, tag="eng")
jieba.add_word("O", freq=None, tag="eng")
jieba.add_word("P", freq=None, tag="eng")
jieba.add_word("Q", freq=None, tag="eng")
jieba.add_word("R", freq=None, tag="eng")
jieba.add_word("S", freq=None, tag="eng")
jieba.add_word("T", freq=None, tag="eng")
jieba.add_word("U", freq=None, tag="eng")
jieba.add_word("V", freq=None, tag="eng")
jieba.add_word("W", freq=None, tag="eng")
jieba.add_word("X", freq=None, tag="eng")
jieba.add_word("Y", freq=None, tag="eng")
jieba.add_word("Z", freq=None, tag="eng")


postag = dict()
with open("local/postag", "r") as word_dict:
    idx = 1
    for line in word_dict:
        line = line.strip()
        postag[line] = idx
        idx = idx + 1

file_pos = open(sys.argv[2], "w")

with open(sys.argv[1], "r") as input_data:
    for line in input_data:
        _ = 0
        test = 0
        words = pseg.cut(line)
        for w in words:
            if w.flag == "x":
                _ = 0
            elif _ == 0:
                file_pos.write("%s " % postag[w.flag])
                _ = 1
                test = 1
        if test == 1:
            file_pos.write("\n")

file_pos.close()
