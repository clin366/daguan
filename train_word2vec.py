# -*- coding: utf-8 -*-
"""
Created on Thu May 03 11:25:13 2018

@author: Eirc
"""
import json
import word2vec

def train_word2vec(word2vec_size=128):
    seg_file = "/home/chenyu/intent_reco/output/seg.txt"
    word2vec_output_file = "/home/chenyu/intent_reco/output/word2vec_" + str(word2vec_size) + ".bin"
    print "Start training word2vec"
    word2vec.word2vec(seg_file, word2vec_output_file, size=word2vec_size, verbose=True)
    print "End training word2vec"
    
    print "Start creating dictionary ..."
    word_dic = {}
    model = word2vec.load(word2vec_output_file)
    voc_size = model.vocab.size
    for i in range(voc_size):
        word_dic[model.vocab[i]] = model.vectors[i].tolist()
    print "End creating dictionary"
    
    word_dict_path = "/home/chenyu/intent_reco/output/word_dic_" + str(word2vec_size) + ".json"
    print "Start storing dictionary ..."
    with open(word_dict_path, "w") as f:
        json.dump(word_dic, f)
    print "End storing dictionary"

def main():
    train_word2vec()

if __name__=='__main__':
    main()
