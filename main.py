# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:44:23 2018

@author: Eric
"""
# import random
# from sklearn import svm
# from sklearn import metrics
# from sklearn.externals import joblib
from logger_system import log
# from baseline import create_file_vector

import numpy as np
np.random.seed(123)  # for reproducibility
import json
import tensorflow as tf
import os
from idcnn import Model as IdCNN
from bilstm import Model as BiLSTM
FLAGS = tf.app.flags.FLAGS
from sklearn.metrics import accuracy_score

tf.app.flags.DEFINE_string('log_dir', "logs", 'The log  dir')
tf.app.flags.DEFINE_string("word2vec_path", "/Users/flynn/Desktop/DaGuan/new_data/word_dic_64.json",
                           "the word2vec data path")
tf.app.flags.DEFINE_integer("max_sentence_len", 100,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 64, "embedding size")
tf.app.flags.DEFINE_integer("num_tags", 14, "BMES")
tf.app.flags.DEFINE_integer("num_hidden", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 100, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 1000, "trainning steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_bool("use_idcnn", False, "whether use the idcnn")
tf.app.flags.DEFINE_integer("track_history", 6, "track max history accuracy")

class Model:
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden):
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = self.load_w2v(c2vPath)
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
        if FLAGS.use_idcnn:
            self.model = IdCNN(layers, 3, FLAGS.num_hidden, FLAGS.embedding_size,
                               FLAGS.max_sentence_len, FLAGS.num_tags)
        else:
            self.model = BiLSTM(
                FLAGS.num_hidden, FLAGS.max_sentence_len, FLAGS.num_tags)
        self.inp = tf.placeholder(tf.float32,
                                  shape=[None, FLAGS.max_sentence_len, FLAGS.embedding_size],
                                  name="input_placeholder")
        pass

    def length(self, data):
        if (tf.contrib.framework.is_tensor(data)):
            data = tf.reshape(data[:,:,0], [-1, FLAGS.max_sentence_len])
        # used = tf.sign(tf.abs(data[:, 0]))
        # used = tf.sign(tf.abs(data))
        # length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(used, tf.int32)
        return length

    def inference(self, X, reuse=None, trainMode=True):
        if (tf.contrib.framework.is_tensor(X)):
            word_vectors = X
        else:
            word_vectors = tf.convert_to_tensor(self.create_file_vector(X, embedding_size = FLAGS.embedding_size), dtype=tf.float32)
        length = self.length(X)
        reuse = False if trainMode else True
        if FLAGS.use_idcnn:
            word_vectors = tf.expand_dims(word_vectors, 1)
            unary_scores = self.model.inference(word_vectors, reuse=reuse)
        else:
            unary_scores = self.model.inference(
                word_vectors, length, reuse=reuse)
        return unary_scores, length

    def loss(self, X, Y):
        print("we have go to loss function!")
        P, sequence_length = self.inference(X)
        print("We have the inference P!!!")
        cross_entropy2_step1=tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=P)
        loss=tf.reduce_sum(cross_entropy2_step1)
        return P, loss
    
    def load_w2v(self, dic_path):
        print("Start loading dict")
        with open(dic_path, "r") as f:
            word_dic = json.load(f)
        print("End loading dict")
        return word_dic
    
    def create_file_vector(self, file_origin_list, embedding_size):
        word_dic = self.c2v
        #Create word vector array
        word_vector = []
        for article in file_origin_list:
            article_vector = []
            for word in article:
                if word_dic.has_key(word):
                    article_vector.append(np.array(word_dic[word]))
                else:
                    article_vector.append([0.0] * embedding_size)
            if (len(article_vector) < FLAGS.max_sentence_len):
                for comp_index in xrange(0, (FLAGS.max_sentence_len - len(article_vector))):
                    article_vector.append([0.0] * embedding_size)
            word_vector.append(np.array(article_vector))
        return np.array(word_vector)

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp,
                                            reuse=True,
                                            trainMode=False)
        return P, sequence_length


def test_evaluate(sess, unary_score, test_sequence_length, inp,
                  tX, tY):
    totalEqual = 0
    batchSize = FLAGS.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0
    
    pred_labels = []
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        label = []
        for i in unary_score_val:
            for j in i:
                max_val = np.argmax(j)
                label.append(max_val)
        pred_labels.extend(label)
                
        
        
    new_Y = tY.reshape(-1,)
    result = accuracy_score(pred_labels, new_Y)

    print("The accuracy score!!!")
    print(result)
    return result
def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)
            
def main():
    trainDataPath = '/Users/flynn/Desktop/DaGuan/daguan/trainMatrix.txt'
    graph = tf.Graph()

    load_Y = np.loadtxt("sep_label.txt")
    Y_single = []
    for i in load_Y:
        Y_single.append([int(i) - 1])
    load_Y = np.array(Y_single)

    X_matrix = np.loadtxt(trainDataPath)
    X_matrix = X_matrix.astype(int)

    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.num_tags,
                      FLAGS.word2vec_path, FLAGS.num_hidden)
        print("train data path:", trainDataPath)
        encoded_Y = tf.one_hot(load_Y, FLAGS.num_tags,
               on_value=1.0, off_value=0.0,
               axis=-1)
        P, total_loss = model.loss(X_matrix, encoded_Y)
        train_op = train(total_loss)
        test_unary_score, test_sequence_length = model.test_unary_score()
    #     sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        sv = tf.train.Supervisor(graph=graph, logdir=None)
        with sv.managed_session(master='') as sess:
            print("This is the Y label")
            print("the Y-matrix")
            # actual training loop
            training_steps = FLAGS.train_steps
            trackHist = 0
            bestAcc = 0

            print("NOW WE AER IN THE STEPS!")
            for step in range(training_steps):
                if sv.should_stop():
                    break
                try:
                    sess.run(train_op)
                    if (step + 1) % 10 == 0:
                        print("[%d] loss: [%r]" %
                              (step + 1, sess.run(total_loss)))
                        X_test = model.create_file_vector(X_matrix, embedding_size = FLAGS.embedding_size)
                        
                        acc = test_evaluate(sess, test_unary_score,
                                            test_sequence_length,
                                            model.inp, X_test, load_Y)
                        if acc > bestAcc:
                            bestAcc = acc
                            trackHist = 0
                        elif trackHist > FLAGS.track_history:
                            print(
                                "always not good enough in last %d histories, best accuracy:%.3f"
                                % (trackHist, bestAcc))
                            break
                        else:
                            trackHist += 1
                except KeyboardInterrupt, e:
                    raise e
        
if __name__=='__main__':
    main()