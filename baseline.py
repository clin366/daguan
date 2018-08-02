# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:03:03 2018

@author: Eric
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from logger_system import log
    
def create_file_vector(train_file_path, dic_path):
    """This is used to create the file vector represent
    This baseline method is as follow:
        1. train word2vec
        2. calculate tf-idf
        3. sort tf-idf value, get Top 128 word
        4. get average vector value of Top 128 word as file vector represent
    """
    # Read train file
    file_origin_list = read_origin_data_file(train_file_path)
    word_list, tf_idf_matrix = calculate_tf_idf_matrix(file_origin_list)
    word_dic = load_dict(dic_path)
    file_vector_list = []
    
    # Loop all file to create file vector representation
    for i in range(len(file_origin_list)):
        if i % 100 == 0:
            log.info("Now has processed %d file" %(i))
        key_word_list = get_top_tfidf_value_word(word_list, tf_idf_matrix[i])
        cur_file_vector = count_average(key_word_list, word_dic, 128)
        file_vector_list.append(cur_file_vector)
        
    return file_vector_list

def read_origin_data_file(data_file_path):
    """This is used to read the data file"""
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
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(file_origin_list))

    word_list = vectorizer.get_feature_names()
    tf_idf_matrix = tfidf.toarray()
    return word_list, tf_idf_matrix

def get_top_tfidf_value_word(word_list, tfidf_value, num_top=128):
    """This is used to get the top tfidf value words"""
    word_value_list = []
    for i in range(len(word_list)):
        word_value_list.append([tfidf_value[i], word_list[i]])
    word_value_list.sort(reverse=True)
    key_word = []
    for i in range(num_top):
        key_word.append(word_value_list[i][1])
    return key_word

def count_average(key_word_list, word_dic, embedding_size=128):
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