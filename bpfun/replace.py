# -*- coding: utf-8 -*-
# @Time    : 2023/10/20 12:37
# @Author  : 孙昊
# @File    : replace.py




import numpy as np

def replace(seq):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in seq:
        str = ''
        for j in i:
            str = str + amino_acids[j]
        data_e.append(str)
    return data_e

if __name__ == '__main__':
    train_data_1 = np.load('data/tr_seqs.npy')
    seqs = replace(train_data_1)
    print(len(seqs))
    # from prot_t5 import t5
    # t5(seqs)




