# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
from spacy import displacy
import data_preparation

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix(text):

    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))
    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename, train_data):

    idx2graph={}
    fout = open(filename + '.graph', 'wb')
    for i in range(len(train_data)):
        adj_matrix=dependency_adj_matrix(train_data[i]['sentence'])
        idx2graph[i]=adj_matrix
    pickle.dump(idx2graph,fout)
    fout.close()

if __name__ == '__main__':
    # train_re,test_re=data_preparation.split_corpus(data_preparation.create_database('re'))
    # process('re',train_re)
    text = "The pizza is delicious, but its price is high. "
    adj = dependency_adj_matrix(text)
