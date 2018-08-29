# -*- coding: utf-8 -*-
# @Author: Yu Chen
# @Date:   2017-6-20 14:17:53
# @Last Modified by:   Yu Chen
# @Last Modified time: 2017-06-22 15:20:11
import time
import string
import numpy as np
import tensorflow as tf

# Define the function to generate the char vector dictionary
def generate_vec_dict(path):

  vec_result = open(path, "r")
  vec_dict = {}
  index = 0

  while True:
    line = vec_result.readline()

    if not line:
      break

    line = line.split(" ")
    vec_dict[line[0].decode("utf-8")] = index - 1
    index += 1

  vec_result.close()
  return vec_dict

# Define the function to read the parameters
def read_parameter(path):

    parameter_file = open(path, "r")
    parameter = {}

    while True:
        line = parameter_file.readline()

        if not line:
            break

        line = line.split(",")
        parameter[line[0]] = int(line[1])

    parameter_file.close()
    return parameter

# Two functions to generate result pre and post-CRF
def generate_result_pre_crf(sess, unary_score, test_sequence_length, transMatrix, inp, tX, batchSize):
    return generate_result(sess, unary_score, test_sequence_length, transMatrix, inp, tX, batchSize, pre_crf = True)

def generate_result_post_crf(sess, unary_score, test_sequence_length, transMatrix, inp, tX, batchSize):
    return generate_result(sess, unary_score, test_sequence_length, transMatrix, inp, tX, batchSize, pre_crf = False)


# Define the function to evaluate the text
def generate_result(sess, unary_score, test_sequence_length, transMatrix, inp, tX, batchSize, pre_crf):

    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1)/batchSize) + 1
    result = []

    
    for i in range(numBatch):

        endOff = (i + 1) * batchSize
        
        if endOff > totalLen:
            endOff = totalLen

        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run([unary_score, test_sequence_length], feed_dict)
        
        for tf_unary_scores_, sequence_length_ in zip(unary_score_val, test_sequence_length_val):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            
            # viterbi解码
            if len(tf_unary_scores_) == 0:
               result.append([])
            else:
                if (pre_crf):
                    result.append(tf_unary_scores_) 
                else:
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)
                    result.append(viterbi_sequence)

    return result
    