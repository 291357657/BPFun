# -*- coding: utf-8 -*-
# @Time    : 2023/10/15 16:27
# @Author  : 孙昊
# @File    : predictor.py


import os
import keras
import torch
import numpy as np
from keras.models import load_model
from transformer import TransformerEncoder,PositionalEmbedding


def predict(test, h5_model):
    for ii in range(0, len(h5_model)):
        # 1.loading model
        h5_model_path = os.path.join(h5_model[ii])
        load_my_model = load_model(h5_model_path,custom_objects={'TransformerEncoder':TransformerEncoder,
                                                                 'PositionalEmbedding':PositionalEmbedding})

        # 2.predict

        score = load_my_model.predict(test)

        # print('predicting...')
        # for i in range(len(score)):
        #     for j in range(len(score[i])):
        #         if score[i][j] < 0.5:
        #             score[i][j] = 0;
        #         else:
        #             score[i][j] = 1

        # 3.getting score
        if ii == 0:
            temp_score = score
        else:
            temp_score += score
        print('finish')


    # getting prediction label
    score_label = temp_score / len(h5_model)
    np.set_printoptions(threshold=np.inf)
    print(score_label)
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5: score_label[i][j] = 0
            else: score_label[i][j] = 1

    return score_label




if __name__ == '__main__':
    seqs = []
    names =[]
    with open('test/test sequence.txt') as f:
    # with open('test/Duplicate categories.txt') as f:
        for each in f:
            if each == '\n':
                continue
            elif each[0] == '>':
                names.append(each[:-1])
            else:
                seqs.append(each.rstrip())
    # print(seqs)
    # print(names)

    from aaindex1_test import encod
    traindata = encod(seqs)
    tr_seqs = np.array(traindata)
    print(tr_seqs.shape)
    np.save('test/aaindex1.npy', tr_seqs)

    from esm_test import enc
    traindata = enc(seqs)
    tr_seqs = np.array(traindata)
    print(tr_seqs.shape)
    np.save('test/esm2.npy', tr_seqs)

    from getdata import PadEncode
    data = PadEncode(seqs,32)
    tr_seqs = np.array(data)
    np.save('test/seqs.npy', tr_seqs)

    from prot_t5 import prot_t5
    num_trans = 1
    print('第' + str(num_trans) + '个')
    traindata = prot_t5(seqs[0])
    tr_data = seqs[1:]
    for i in tr_data:
        num_trans += 1
        print('第' + str(num_trans) + '个')
        i = prot_t5(i)
        traindata = np.concatenate((traindata, i), axis=0)
    torch.save(traindata, 'test/prott5.pt')


    data = []
    train_data_1 = np.load('test/seqs.npy')
    train_data_2 = keras.utils.to_categorical(train_data_1)
    train_data_3 = torch.load('test/prott5.pt')
    train_data_4 = np.load('test/esm2.npy')
    train_data_5 = np.load('test/aaindex1.npy')

    data.append(train_data_1)
    data.append(train_data_2)
    data.append(train_data_3)
    data.append(train_data_4)
    data.append(train_data_5)

    #print(data)

    h5_model = []
    model_num = 9
    for i in range(1, model_num + 1):
        h5_model.append('model/Independently_tested_model_7type' + str(i) + '.h5')

    result = predict(data, h5_model)

    # label
    peptides = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP', 'AAP', 'AOP']
    functions = []
    for e in result:
        temp = ''
        for i in range(len(e)):
            if e[i] == 1:
                temp = temp + peptides[i] + ','
            else:
                continue
        if temp == '':
            temp = 'none'
        if temp[-1] == ',':
            temp = temp.rstrip(',')
        functions.append(temp)


    output_file = os.path.join('test/result.txt')
    with open(output_file, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + '\n')
            f.write('functions:' + functions[i] + '\n')

