
import random
from copy import copy
import argparse
from collections import defaultdict, Counter
from lxml import etree
from sacremoses import MosesTokenizer
# import simplejson as json
import codecs
import random


def split_corpus(corpus,per=0.9):
    print('读取完成，开始切分')
    random.seed(77)
    random.shuffle(corpus)
    train_num = int(len(corpus) * per)
    test_num=int(len(corpus))-train_num
    data_train = corpus[:train_num]
    data_test = corpus[train_num:]
    print('切分完成')
    print("train数据的长度为",train_num)
    print("test的数据长度为",test_num)
    return data_train, data_test

def create_database(d_type='re'):
    if d_type=='re':
        inputpath='data/origin/Restaurants_Train_v2.json'

    else:
        inputpath='data/origin/Laptops_Train_v2.xml'

    corpus=read_sentence_target(inputpath)
    return list(corpus)

def read_sentence_target(file_path, max_offset_len=83):

    tokenize = MosesTokenizer()
    with open(file_path, 'r') as fopen:
        raw = fopen.read()
        root = etree.fromstring(raw.encode('utf-8'))

        for sentence in root:

            example = dict()
            example["sentence"] = sentence.find('text').text.lower()

            tokens =tokenize.tokenize(example['sentence'])

            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            example["aspect_sentiment"] = []
            example["left_right"] = []
            example['offset'] = []

            for c in terms:
                target = c.attrib['term'].lower()
                example["aspect_sentiment"].append((target, c.attrib['polarity']))


                left_index = int(c.attrib['from'])
                right_index = int(c.attrib['to'])
                example["left_right"].append((example['sentence'][:right_index],
                                              example['sentence'][left_index:],
                                              c.attrib['polarity']))


                left_word_offset = len(tokenize.tokenize(example['sentence'][:left_index]))
                right_word_offset = len(tokenize.tokenize(example['sentence'][right_index:]))
                token_index = list(range(len(tokens)))
                token_length = float(len(token_index))
                for i in range(len(tokens)):
                    if i < left_word_offset:
                        token_index[i] = 1 - (left_word_offset - token_index[i]) / token_length
                    elif i >= right_word_offset:
                        token_index[i] = 1 - (token_index[i] - (len(tokens) - right_word_offset) + 1) / token_length
                    else:
                        token_index[i] = 0
                token_index += [-1.] * (max_offset_len - len(tokens))
                example['offset'].append((token_index, target, c.attrib['polarity']))
            yield example

# train_,test_=split_corpus(create_database('re'))


