# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 14:41
# @Author  : 孙昊
# @File    : getdata.py


import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)
    return data, max_len

def catch(data, label):
    # preprocessing label and data
    chongfu = 0
    tmpdata = []
    tmplabel = []

    while len(data) > 0:
        l = len(data)
        tmpdata.append(data[0])
        tmplabel.append(label[0])
        tmpd = []
        tmpl = []
        j = 1
        while j < l:
            if data[j] != data[0]:
                tmpd.append(data[j])
                tmpl.append(label[j])
            else:
                chongfu += 1
                # print(label[0], label[j])
                # print(data[0],data[j])
            j += 1
        data = tmpd
        label = tmpl

    print('total number of the same data: ', chongfu)

    return tmpdata,tmplabel

def PadEncode(data, max_len):
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        elif length > max_len:
            elemt = elemt[:max_len]
        data_e.append(elemt)
    return data_e

def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD_.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())
    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test

def datapro(tr_data,typenum):
    tmpdata = []
    tmplabel = []
    lenth = 2000
    for i in tr_data:
        tmpdata.append(i)
        tmplabel.append(typenum)
    while len(tmpdata)<lenth:
        for i in tr_data:
            j = random.randrange(0, len(i))
            tmpdata.append(i[0:j] + "X" + i[j + 1:len(i)])
            tmplabel.append(typenum)
    # print(tmpdata[:lenth])
    return tmpdata[:lenth],tmplabel[:lenth]

if __name__ == '__main__':
    path = 'data'
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP', 'AAP', 'AOP']
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(path, dir, count)
        count += 1

        #increase the data
        print(len(tr_x))
        if len(tr_x)<2000:
           tr_x,tr_y = datapro(tr_x,tr_y[0])
        print(len(tr_x))

        # getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te

        # dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y

    print('----------')
    print('deduplication...')
    tr_data, tr_label = catch(tr_data, tr_label)
    te_data, te_label = catch(te_data, te_label)
    print('----------')

    len_tr = len(tr_label)
    len_te = len(te_label)

    from aaindex1_test import encod
    traindata = encod(tr_data)
    tr_seqs = np.array(traindata)
    print(tr_seqs.shape)
    np.save('data/aaindex1tr_seqs.npy', tr_seqs)
    testdata = encod(te_data)
    te_seqs = np.array(testdata)
    print(te_seqs.shape)
    np.save('data/aaindex1te_seqs.npy', te_seqs)

    from esm_test import enc
    traindata = enc(tr_data)
    np.save('data/esmtr_seqs.npy', tr_seqs)
    tr_seqs = np.array(traindata)
    testdata = enc(te_data)
    te_seqs = np.array(testdata)
    np.save('data/esmte_seqs.npy', te_seqs)

    from prot_t5 import t5
    t5(tr_data,te_data)

    traindata = PadEncode(tr_data, 32)
    testdata = PadEncode(te_data, 32)

    # save data
    tr_seqs = np.array(traindata)
    np.save('data/tr_seqs.npy', tr_seqs)

    te_seqs = np.array(testdata)
    np.save('data/te_seqs.npy', te_seqs)

    # save label
    train_label = np.array(tr_label)
    np.save('data/tr_seqs_label.npy', train_label)

    test_label = np.array(te_label)
    np.save('data/te_seqs_label.npy', test_label)









