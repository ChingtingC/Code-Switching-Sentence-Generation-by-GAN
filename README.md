# Code switching Sentence Generation by Generative Adversarial Networks and its Application to Data Augmentation

Ching-Ting Chang, Shun-Po Chuang, Hung-Yi Lee

[Interspeech 2019](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3214.pdf)

[arXiv:1811.02356](https://arxiv.org/abs/1811.02356)

## Abstract

Code-switching is about dealing with alternative languages in speech or text. It is partially speaker-depend and domain-related, so completely explaining the phenomenon by linguistic rules is challenging. Compared to most monolingual tasks, insufficient data is an issue for code-switching. To mitigate the issue without expensive human annotation, we proposed an unsupervised method for code-switching data augmentation. By utilizing a generative adversarial network, we can generate intra-sentential code-switching sentences from monolingual sentences. We applied proposed method on two corpora, and the result shows that the generated code-switching sentences improve the performance of code-switching language models.


## Outline
1. Introduction
2. Methodology
3. Experimental setup
    - Corpora
    - Model Setup
4. Results
    - Code-switching Point Prediction
    - Generated Text Quality
    - Language Modeling
    - [Examples](https://chingtingc.github.io/Code-Switching-Sentence-Generation-by-GAN/)
5. Conclusion

## Corpora

1. LectureSS: The recording of “Signal and System” (SS) course by one Tai-wanese instructor at National Taiwan University in 2006.
2. [SEAME](https://catalog.ldc.upenn.edu/LDC2015S04): South East Asia Mandarin-English, a conversational speech by Singapore and Malaysia speakers with almost balanced gender in Nanyang Technological University and Universities Sains Malaysia.

## Experimental setup

### Prerequisites
1. Python packages
    - python 3
    - keras 2
    - numpy
    - [jieba](https://github.com/fxsjy/jieba)
    - h5py
    - tqdm
3. Data
    - text files
        * Training set
            1. **corpus/XXX/text/train.mono.txt**: Mono sentences in H
            2. **corpus/XXX/text/train.cs.txt**: CS sentences
        * Development set
            1. **corpus/XXX/text/dev.mono.txt**: Mono sentences in H translated from CS sentences (aligned to 2.)
            2. **corpus/XXX/text/dev.cs.txt**: CS sentences
        * Testing set
            1. **corpus/XXX/text/test.mono.txt**: Mono sentences in H
        * Note
            * Sentences should be segmented into words by **space**.
            * **Words** are based on H language
            * If a **word** in H language is mapped to a **phrase** in G language, we use **dash** to connect the words into one word.
    - **local/XXX/translator.txt**: Translating table from H language to G language
    - **local/XXX/dict.txt**: Word list for traning word-embedding
    - **local/postag.txt**: POS tag list for traning pos-embedding

Type| Example
----|---------
CS  | Causality 這個 也是 你 所 讀 過 的 就是 指 我 output at-any-time 只 depend-on input
Mono from CS in H  | 因果性 這個 也是 你 所 讀 過 的 就是 指 我 輸出 在任意時間 只 取決於 輸入


* **Note**
    * **Mono**: monolingual
    * **CS**: code-switching
    * **H**: host (language)
    * **G**: guest (language)
    * **ASR**: automatic speech recognition

### Preprocess Data

* Use Jieba to get the part-of-speech (POS) tagger of text files for **proposed + POS**
    * Path:
        * Training set
            1. **corpus/XXX/pos/train.mono.txt**: POS of Mono sentences of training set
            2. **corpus/XXX/pos/train.cs.txt**: POS of CS sentences of training set
        * Development set
            1. **corpus/XXX/pos/dev.mono.txt**: POS of Mono sentences of development set set
        * Testing set
            1. **corpus/XXX/pos/test.mono.txt**: POS of Mono sentences of testing set

### Train Model

## Results

* Baselines:
    * ZH
    * EN
    * Random
    * Noun

### Code-switching Point Prediction

* Precision
* Recall
* F-measure
* BLEU-1
* Word Error Rate (WER)

### Generated Text Quality

#### Prerequisites

1. Installation
    - [srilm](http://www.speech.sri.com/projects/srilm/)

* N-gram model
* Recurrent Neural Networks based Language Model (RNNLM)

### Language Modeling

### Automatic Speech Recognition

It's the extended experiment which is not shown in paper.

#### Prerequisites
1. Installation
    - [kaldi](https://kaldi-asr.org/)
    - [srilm](http://www.speech.sri.com/projects/srilm/)
2. Data
    - speech wav files & its text files
