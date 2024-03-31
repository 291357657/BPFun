# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 16:05
# @Author  : 孙昊
# @File    : test.py


import os
import torch
import keras
import pickle
import numpy as np
import evaluation
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model,Model
from transformer import TransformerEncoder,PositionalEmbedding


def load_data():
    #1:seq 2:one-hot 3:prot_t5
    te_label = np.load('data/te_seqs_label.npy')

    test_data_1 = np.load('data/te_seqs.npy')
    test_data_2 = keras.utils.to_categorical(test_data_1)
    test_data_3 = torch.load('data/testdata.pt')
    test_data_4 = np.load('data/esmte_seqs.npy')
    test_data_5 = np.load('data/aaindex1te_seqs.npy')

    test_label = keras.utils.to_categorical(te_label)

    return test_data_1,test_data_2,test_data_3,test_label,test_data_4,test_data_5

def predict(X_test, y_test, thred, weights, h5_model, dir, out_length):
    #y = [one_label.tolist().index(1) for one_label in y_test]
    for ii in range(0, len(weights)):
        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path, custom_objects={'TransformerEncoder': TransformerEncoder,
                                                                  'PositionalEmbedding': PositionalEmbedding})
        #print(load_my_model.summary())
        print("Prediction is in progress")
        score = load_my_model.predict(X_test)
        # np.save('data/pred.npy',score)
        for i in range(len(score)):
            for j in range(len(score[i])):
                if score[i][j] < thred:
                    score[i][j] = 0;
                else:
                    score[i][j] = 1
        a, b, c, d, e, g = evaluation.evaluate(score, y_test)
        print(a, b, c, d, e, g)
        if ii == 0:
            score_label = score
        else:
            score_label += score

    score_label = score_label / len(h5_model)

    with open(os.path.join(dir, 'prediction_prob_' + str(out_length) + 'type' + '.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    with open(os.path.join(dir, 'prediction_prob_' + str(out_length) + 'type' + '.pkl'), 'wb') as f:
        pickle.dump(score_label, f)

    aiming, coverage, accuracy, f1, absolute_true, absolute_false = evaluation.evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('f1:',f1)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)

    out = dir
    Path(out).mkdir(exist_ok=True, parents=True)
    out_path2 = os.path.join(out, 'result_test_' + str(out_length) + 'type' + '.txt')
    with open(out_path2, 'w') as fout:
        fout.write('aiming:{}\n'.format(aiming))
        fout.write('coverage:{}\n'.format(coverage))
        fout.write('accuracy:{}\n'.format(accuracy))
        fout.write('f1:{}\n'.format(f1))
        fout.write('absolute_true:{}\n'.format(absolute_true))
        fout.write('absolute_false:{}\n'.format(absolute_false))
        fout.write('\n')

if __name__ == '__main__':
    length = 32
    out_length = 7
    model_num = 9

    weights = []
    jsonFiles = []
    h5_model = []

    for i in range(1, model_num + 1):
        weights.append("Independently_tested_model_" + str(out_length) + "type{}.hdf5".format(str(i)))
        jsonFiles.append("Independently_tested_model_" + str(out_length) + "type{}.json".format(str(i)))
        h5_model.append("Independently_tested_model_" + str(out_length) + "type{}.h5".format(str(i)))

    test_data1, test_data2, test_data3, test_label,test_data4,test_data5 = load_data()

    predict([test_data1, test_data2, test_data3, test_data4, test_data5], test_label, 0.5, weights, h5_model, 'model', out_length)

    print('Finish!')
