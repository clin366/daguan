# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:03:03 2018

@author: Eric
"""

import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from logger_system import log
  
def create_file_vector(train_file_path, dic_path, embedding_size):
    word_dic = load_dict(dic_path)
    file_origin_list = read_origin_data_file(train_file_path)
    
    #Create word vector array
    word_vector = []
    for article in file_origin_list:
        article_vector = []
        for word in article:
            if word_dic.has_key(word):
                article_vector.append(np.array(word_dic[word]))
            else:
                article_vector.append([0.0] * embedding_size)
        if (len(article_vector) < 100):
            for comp_index in xrange(0, (100 - len(article_vector))):
                article_vector.append([0.0] * embedding_size)
            # print "the length",
            # print len(article_vector)
        word_vector.append(np.array(article_vector))
    return np.array(word_vector)

def read_origin_data_file(data_file_path):
    log.info("Start reading the origin data file")
    file_origin_list = []
    with open(data_file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            line = line.split(' ')
            file_origin_list.append(line)
    return file_origin_list

def calculate_tf_idf_matrix(file_origin_list):
    """This is used to calculate the tf-idf matrix value"""
    log.info("Start calculating the tf idf matrix")
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(file_origin_list))

    word_list = vectorizer.get_feature_names()

    # convert tfidf to csr_matrix
    tfidf_sparse_matrix = csr_matrix(tfidf)
    return word_list, tfidf_sparse_matrix

def get_top_tfidf_value_word(word_list, tfidf_value):
    """This is used to get the top tfidf value words"""
    
    # 获取99分位数，仅保留tfidf值大于该值的词
    standard_tfidf = np.percentile(tfidf_value,99.5)
    key_word_list = []
    for i in range(len(tfidf_value[0])):
        if tfidf_value[0][i] >= standard_tfidf:
            key_word_list.append(word_list[i])
    return key_word_list

def count_average(key_word_list, word_dic, embedding_size):
    """This is used to count the file vector by averaging the key word vector"""
    vector_array = np.zeros(embedding_size)  
    word_count = 0
    
    for word in key_word_list: 
        if word_dic.has_key(word):
            word_count += 1
            vector_array += np.array(word_dic[word])
        else:
            #print ("This word %s is not in dict" % (word))
            pass
    
    # Check word exist
    if word_count > 0:
        vector_array = vector_array/(word_count * 1.0) # Get average
    return vector_array
        
def load_dict(dic_path):
    """Load pre-trained word2vec/char2vec vector dictionary."""
    log.info("Start loading dict")
    with open(dic_path, "r") as f:
        word_dic = json.load(f)  
    log.info("End loading dict")
    return word_dic