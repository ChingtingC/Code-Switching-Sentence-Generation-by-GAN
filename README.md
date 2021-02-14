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

### Corpora

1. LectureSS: The recording of “Signal and System” (SS) course by one Tai-wanese instructor at National Taiwan University in 2006.
2. [SEAME](https://catalog.ldc.upenn.edu/LDC2015S04): South East Asia Mandarin-English, a conversational speech by Singapore and Malaysia speakers with almost balanced gender in Nanyang Technological University and Universities Sains Malaysia.

## Implementation

### Prerequisites
1. Python packages
    - python 3
    - keras 2
    - numpy
    - [jieba](https://github.com/fxsjy/jieba)
    - h5py
    - tqdm
    - nltk
2. Data
    - text files
        * 2 kinds of data: 
            1. monolingual sentences
            2. code-switching sentences 
        * no need aligned
        * should be segmented into words by space (can be done by jieba)
    - speech file (for the extended experiment, i.e. automatic speech recognition(ASR))
3. Other installation
    - [kaldi](https://kaldi-asr.org/) (for the extended experiment, ASR)
    - [srilm](http://www.speech.sri.com/projects/srilm/)

### Preprocess Data
 * Use Jieba to get the part-of-speech (POS) tagger of text files

### Train Model

### Evaluate Result

* Baselines:
    * ZH
    * EN
    * Random
    * Noun

#### Code-switching Point Prediction

* Precision
* Recall
* F-measure
* BLEU-1
* Word Error Rate (WER)

#### Generated Text Quality

* N-gram model
* Recurrent Neural Networks based Language Model (RNNLM)

#### Language Modeling
