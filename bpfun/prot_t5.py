# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 14:54
# @Author  : 孙昊
# @File    : prot_t5.py


import torch
import numpy as np
from transformers import T5Tokenizer, T5Model

#prot_t5***********************
tokenizer_file = 'prot_t5'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_file, do_lower_case=False)
model_file='prot_t5'
model = T5Model.from_pretrained(model_file)

def prot_t5(X_train):
    sequences_Example = [" ".join(list(X_train))]

    # ids = tokenizer.batch_encode_plus(sequences_Example,add_special_tokens=True,return_tensors='pt',padding='max_length',max_length=128,truncation=True)
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding='max_length', max_length=32,
                                      truncation=True)

    input_ids = torch.as_tensor(ids['input_ids'])
    attention_mask = torch.as_tensor(ids['attention_mask'])

    with torch.no_grad():
        # embedding = model(input_ids=input_ids,attention_mask=attention_mask,decoder_input_ids=input_ids)[0][:,:,:128]
        embedding = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)[2]

    # For feature extraction we recommend to use the encoder embedding
    embedding = embedding.cpu().numpy()

    re = np.zeros((1, 32, 1024), dtype='float32')
    for i, x in enumerate(embedding):
        re[i] = x
    return re
#******************************


def t5(tr_data,te_data):
    #process and save train data
    num_trans = 1
    print('第' + str(num_trans) + '个')
    traindata = prot_t5(tr_data[0])
    tr_data = tr_data[1:]
    for i in tr_data:
        num_trans += 1
        print('第' + str(num_trans) + '个')
        i = prot_t5(i)
        traindata = np.concatenate((traindata, i), axis=0)
    torch.save(traindata, 'data/traindata.pt')

    # process and save test data
    num_test = 1
    print('第' + str(num_test) + '个')
    testdata = prot_t5(te_data[0])
    te_data = te_data[1:]
    for i in te_data:
        num_test += 1
        print('第' + str(num_test) + '个')
        i = prot_t5(i)
        testdata = np.concatenate((testdata, i), axis=0)
    torch.save(testdata, 'data/testdata.pt')


# seqs = ['GIPCGESCVFIPCXXGAIGCSCKSKVCYRN','YSPFSSFPRX']
# for i in seqs:
#     tmp = prot_t5(i)
