# encoding: utf-8
import sys
import numpy as np
from argparse import ArgumentParser

# Add arguement
parser = ArgumentParser()
parser.add_argument("-i", "--input", dest="INPUT", default="corpus/sample/pos/train.cs.txt")
parser.add_argument("-o", "--output", dest="OUTPUT", default="exp/cs-rate.txt")
args = parser.parse_args()

INPUT = args.INPUT
OUTPUT = args.OUTPUT

output = open(OUTPUT,"w")
label_r = []

# Calculate & write cs rate of each line
with open(INPUT, "r") as fin:
    for line in fin:
        line = [int(x) for x in line.strip().split(' ')]
        output.write("%d %% \n" % (np.average(np.asarray(line))*100))
        label_r = label_r + line

# Calculate & write cs rate of overall
output.write("Total: %d %% \n" % (np.average(np.asarray(label_r))*100))
print("%d" % (np.average(np.asarray(label_r))*100))
output.close()
