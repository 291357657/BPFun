# -*- coding: utf-8 -*-
# @Time    : 2023/9/16 18:24
# @Author  : 孙昊
# @File    : aaindex1_test.py


import pandas as pd
import numpy as np

df = pd.read_csv("data/aaindex1.my.csv",encoding="utf-8",header=None)
df_array = np.array(df)
df_list = df_array.tolist()

def encode(seq):
    # print(seq)
    amino_acids = 'ARNDCQEGHILKMFPSTWYVX'
    enseq = []
    for i in seq:
        index = amino_acids.index(i)
        if index == 20:
            aacode = [0]*566
            enseq.append(aacode)
        else:
            aacode = df_list[index][1:]
            enseq.append(aacode)
    return enseq

def encod(seqs):
    data = []
    for i in seqs:
        if len(i) < 32:
            i = i + 'X' * (32 - len(i))
            tmp = encode(i)
            data.append(tmp)
        else:
            i = i[:32]
            tmp = encode(i)
            data.append(tmp)
    return data









