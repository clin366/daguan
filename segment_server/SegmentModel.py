# -*- coding: utf-8 -*-
# @Author: Koth Chen
# @Date:   2016-07-26 13:48:32
# @Last Modified by:   Koth
# @Last Modified time: 2017-04-07 23:04:45
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np

import tensorflow as tf
import os
from idcnn import Model as IdCNN

class SegmentModel:
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden, model_dict):
        self.max_sentence_len = 80
        self.num_hidden = 100
        self.embedding_size = 50
        self.num_tags = 4
        self.use_idcnn = True
        
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = self.load_w2v(c2vPath, self.embedding_size)
        self.words = model_dict['words:0']
        






        layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        if self.use_idcnn:
            self.model = IdCNN(layers, 3, self.num_hidden, self.embedding_size,
                               self.max_sentence_len, self.num_tags, model_dict)
        else:
            self.model = BiLSTM(
                self.num_hidden, self.max_sentence_len, self.num_tags)
        self.trains_params = None
        self.inp = tf.placeholder(tf.int32,
                                  shape=[None, self.max_sentence_len],
                                  name="input_placeholder")
        pass

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, X, reuse=None, trainMode=True):
        word_vectors = tf.nn.embedding_lookup(self.words, X)
        length = self.length(X)
        reuse = False if trainMode else True
        if self.use_idcnn:
            word_vectors = tf.expand_dims(word_vectors, 1)
            unary_scores = self.model.inference(word_vectors, reuse=None)
        else:
            unary_scores = self.model.inference(
                word_vectors, length, reuse=None)
        return unary_scores, length

    def loss(self, X, Y):
        P, sequence_length = self.inference(X)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length)
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def load_w2v(self, path, expectDim):
        fp = open(path, "r")
        print("load data from:", path)
        line = fp.readline().strip()
        ss = line.split(" ")
        total = int(ss[0])
        dim = int(ss[1])
        assert (dim == expectDim)
        ws = []
        mv = [0 for i in range(dim)]
        second = -1
        for t in range(total):
            if ss[0] == '<UNK>':
                second = t
            line = fp.readline().strip()
            ss = line.split(" ")
            assert (len(ss) == (dim + 1))
            vals = []
            for i in range(1, dim + 1):
                fv = float(ss[i])
                mv[i - 1] += fv
                vals.append(fv)
            ws.append(vals)
        for i in range(dim):
            mv[i] = mv[i] / total
        assert (second != -1)
        # append one more token , maybe useless
        ws.append(mv)
        if second != 1:
            t = ws[1]
            ws[1] = ws[second]
            ws[second] = t
        fp.close()
        return np.asarray(ws, dtype=np.float32)

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp,
                                            reuse=None,
                                            trainMode=False)
        return P, sequence_length
