# -*- coding: utf-8 -*-
# @Time    : 2023/9/13 16:13
# @Author  : 孙昊
# @File    : evaluation.py

import numpy as np
from sklearn import metrics

def Aiming(y_hat, y):
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k / n

def Coverage(y_hat, y):
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])

    return sorce_k / n

def Accuracy(y_hat, y):
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k / n

def F1Measure(y_hat, y):
    count = 0
    for i in range(y.shape[0]):
        if (sum(y[i]) == 0) and (sum(y_hat[i]) == 0):
            continue
        p = sum(np.logical_and(y[i], y_hat[i]))
        q = sum(y[i]) + sum(y_hat[i])
        count += (2 * p) / q
    return count / y.shape[0]

def AbsoluteTrue(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n

def AbsoluteFalse(y_hat, y):
    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def evaluate(y_hat, y):
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    f1 = F1Measure(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return aiming, coverage, accuracy,f1, absolute_true, absolute_false


def evaluates(pred,test_label):
    pred_res = pred.flatten()
    test_label = test_label.flatten()

    pred_label = [1 if x > 0.5 else 0 for x in pred_res]

    acc = metrics.accuracy_score(y_true=test_label, y_pred=pred_label)
    precise = metrics.precision_score(y_true=test_label, y_pred=pred_label)
    recall = metrics.recall_score(y_true=test_label, y_pred=pred_label)
    f1 = metrics.f1_score(y_true=test_label, y_pred=pred_label)

    return acc,precise,recall,f1