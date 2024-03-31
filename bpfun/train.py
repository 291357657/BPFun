# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 15:13
# @Author  : 孙昊
# @File    : train.py


import os
import math
import torch
import keras
import pickle
import evaluation
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model,Model
from sklearn.utils.class_weight import compute_class_weight
from transformer import TransformerEncoder,PositionalEmbedding

from keras.backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


def load_data():
    #1:seq 2:one-hot 3:prot_t5
    tr_label = np.load('data/tr_seqs_label.npy')

    train_data_1 = np.load('data/tr_seqs.npy')
    train_data_2 = keras.utils.to_categorical(train_data_1)
    train_data_3 = torch.load('data/traindata.pt')
    train_data_4 = np.load('data/esmtr_seqs.npy')
    train_data_5 = np.load('data/aaindex1tr_seqs.npy')

    train_label = keras.utils.to_categorical(tr_label)

    return train_data_1,train_data_2,train_data_3,train_label,train_data_4,train_data_5

def mytrain(data1,data2,data3,data4,data5,label,para,length,out_length,model_num):
    # y = [one_label.tolist().index(1) for one_label in label]
    for count in range(1, model_num + 1):
        model = mymodel(length, out_length, para)
        history = model.fit([data1, data2, data3, data4, data5], label, epochs=50, batch_size=64,verbose=1)
        each_model = os.path.join('model','Independently_tested_model_' + str(out_length) + 'type' + str(count) + '.h5')
        model.save(each_model)

from model import mymodel,feature
if __name__ == '__main__':
    length = 32
    out_length = 7
    model_num = 9

    ed = 64
    ps = 3
    fd = 1024
    dp = 0.3
    lr = 1e-5
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    #load data and label
    train_data1,train_data2,train_data3,train_label,train_data4,train_data5 = load_data()

    #train model
    mytrain(train_data1,train_data2,train_data3,train_data4,train_data5,train_label,para,length,out_length,model_num)

    print('Finish!')





