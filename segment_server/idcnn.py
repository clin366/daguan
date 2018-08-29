#!/usr/bin/env python
# -*- coding:utf-8 -*-

# File: idcnn.py
# Project: /Users/tech/code/kcws
# Created: Mon Jul 31 2017
# Author: Koth Chen
# Copyright (c) 2017 Koth
#
# <<licensetext>>

import tensorflow as tf


class Model:
    def __init__(self,
                 layers,
                 filterWidth,
                 numFilter,
                 embeddingDim,
                 maxSeqLen,
                 numTags,
                 model_dict,
                 repeatTimes=4):
        self.layers = layers
        self.filter_width = filterWidth
        self.num_filter = numFilter
        self.embedding_dim = embeddingDim
        self.repeat_times = repeatTimes
        self.num_tags = numTags
        self.max_seq_len = maxSeqLen
        self.model_dict = model_dict


    def inference(self, X, reuse=False):
        with tf.variable_scope("idcnn", reuse=True):
            filter_weights = self.model_dict['idcnn/idcnn_filter:0']
            layerInput = tf.nn.conv2d(X,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else True):
                        variableScope = "atrous-conv-layer-%d" % i
                        fullScope = "idcnn/" + variableScope
                        w = self.model_dict[fullScope + "/filterW:0"]
                        b = self.model_dict[fullScope + "/filterB:0"]
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])

            finalW = self.model_dict['idcnn/finalW:0']
            finalB = self.model_dict['idcnn/finalB:0']

            scores = tf.nn.xw_plus_b(finalOut, finalW, finalB, name="scores")
        if reuse:
            scores = tf.reshape(scores, [-1, self.max_seq_len, self.num_tags],
                                name="Reshape_7")
        else:
            scores = tf.reshape(scores, [-1, self.max_seq_len, self.num_tags],
                                name=None)
        return scores
