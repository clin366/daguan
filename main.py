# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 18:44:23 2018

@author: Eric
"""
import random
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from logger_system import log
from baseline import create_file_vector

class BaselineModel(object):
    """This is used to do the baseline predict"""
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
    
    def fit(self, value_c, model_save_path, train_vector, train_label):
        log.info("Start fitting the model")
        classifier = svm.LinearSVC(random_state = 0, C = value_c)
        classifier.fit(train_vector, train_label)
        joblib.dump(classifier, model_save_path, compress=3)
        log.info("End fitting the modle")
        
    def predict(self, model_save_path, file_vector_list):
        log.info("Start predicting the model")
        classifier = joblib.load(model_save_path)
        predict_result = classifier.predict(file_vector_list)
        return predict_result
        
def generate_model_input_data(train_word_file, train_label_file, dict_file, cv_ratio=0.8):
    """This is used to generate the input data of model, also do the cv work"""
    label = []
    with open(train_label_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace("\n", "")
            label.append(int(line))
    
    train_file_vector = create_file_vector(train_word_file, dict_file)
    train_vector = []
    train_label = []
    train_cv_vector = []
    train_cv_label = []
    
    for i in range(len(train_file_vector)):
        if random.random() < cv_ratio:
            train_vector.append(train_file_vector[i])
            train_label.append(label[i])
        else:
            train_cv_vector.append(train_file_vector[i])
            train_cv_label.append(label[i])
    return train_vector, train_label, train_cv_vector, train_cv_label
            
def main():
    train_word_file = "/home/chenyu/daguan/data/train_word"
    train_label_file = "/home/chenyu/daguan/data/train_label"
    dict_file = "/home/chenyu/daguan/output/word_dic_128.json"
    #model_save_path = "/home/chenyu/daguan/model/basic_svm"
    basic_model = BaselineModel(128)
    
    train_vector, train_label, train_cv_vector, train_cv_label = generate_model_input_data(train_word_file, train_label_file, dict_file)
    
    c_value = [1, 10, 50, 100, 500]
    for c in c_value:
        model_save_path = "/home/chenyu/daguan/model/basic_svm_" + str(c)
        basic_model.fit(c, model_save_path, train_vector, train_label)
        train_predict_result = basic_model.predict(model_save_path, train_vector)
        cv_predict_result = basic_model.predict(model_save_path, train_cv_vector)
        
        accuracy_score = metrics.precision_score(train_label, train_predict_result, average='micro') 
        F1_score = metrics.f1_score(train_label, train_predict_result, average='weighted')  
        log.info("The following is the result for c value " + str(c))
        log.info("The accuracy for train data is " + str(accuracy_score))
        log.info("The f1 score for train data is " + str(F1_score))
        
        accuracy_score = metrics.precision_score(train_cv_label, cv_predict_result, average='micro') 
        F1_score = metrics.f1_score(train_cv_label, cv_predict_result, average='weighted')  
        log.info("The accuracy for cv data is " + str(accuracy_score))
        log.info("The f1 score for cv data is " + str(F1_score))
        
if __name__=='__main__':
    main()