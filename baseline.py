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
  
def create_file_vector(train_file_path, test_file_path, dic_path, embedding_size):
    """This is used to create the file vector represent
    This baseline method is as follow:
        1. train word2vec
        2. calculate tf-idf
        3. sort tf-idf value, get Top 128 word
        4. get average vector value of Top 128 word as file vector represent
    """
    # Read train file
    train_origin_list = read_origin_data_file(train_file_path)
    test_origin_list = read_origin_data_file(test_file_path)
    file_origin_list = train_origin_list + test_origin_list
        
    word_list, tf_idf_sparse_matrix = calculate_tf_idf_matrix(file_origin_list)
    word_dic = load_dict(dic_path)
    file_vector_list = []
    
    # Create word vector array
    word_vector = []
    for word in word_list:
        if word_dic.has_key(word):
            word_vector.append(word_dic[word])
        else:
            word_vector.append([0.0]*64)
    word_vector = np.array(word_vector)
    
    # Loop all file to create file vector representation
    log.info("The number of file is %d" %(len(file_origin_list)))
    log.info("The number of word list is %d" %(len(word_list)))
    for i in range(len(file_origin_list)):
        if i % 1000 == 0:
            log.info("Now has processed %d file" %(i))
            
        normal_array = tf_idf_sparse_matrix[i].toarray()
        cur_file_vector = calculate_file_vector(word_vector, normal_array)
        file_vector_list.append(cur_file_vector)
        
    return file_vector_list[:len(train_origin_list)], file_vector_list[len(train_origin_list):]

def read_origin_data_file(data_file_path):
    """This is used to read the data file"""
    log.info("Start reading the origin data file")
    file_origin_list = []
    with open(data_file_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("\n", "")
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

def calculate_file_vector(word_vector, tfidf_value):
    """This is used to calculate the file vector by matrix operation"""
    standard_tfidf = np.percentile(tfidf_value,99.5)
    temp_matrix = tfidf_value > standard_tfidf
    temp_matrix = temp_matrix[0]
    result = word_vector[temp_matrix == 1]
    return np.mean(result, axis = 0)
           
def load_dict(dic_path):
    """Load pre-trained word2vec/char2vec vector dictionary."""
    log.info("Start loading dict")
    with open(dic_path, "r") as f:
        word_dic = json.load(f)  
    log.info("End loading dict")
    return word_dic
