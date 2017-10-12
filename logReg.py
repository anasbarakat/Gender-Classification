#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 03:06:55 2017

@author: anasbarakat
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression

# Critere de performance
def compute_pred_score(y_true, y_pred):
    y_pred_unq =  np.unique(y_pred)
    for i in y_pred_unq:
        if((i != -1) & (i!= 1) & (i!= 0) ):
            raise ValueError('The predictions can contain only -1, 1, or 0!')
    y_comp = y_true * y_pred
    score = float(10*np.sum(y_comp == -1) + np.sum(y_comp == 0))
    score /= y_comp.shape[0]
    return score

X_train_fname = 'training_templates.csv'
y_train_fname = 'training_labels.txt'
X_test_fname  = 'testing_templates.csv'
X_train = pd.read_csv(X_train_fname, sep=',', header=None).values
X_test  = pd.read_csv(X_test_fname,  sep=',', header=None).values
y_train = np.loadtxt(y_train_fname, dtype=np.int)

clf = LogisticRegression(C= 10)
#X_train = X_train[:,[1, 6, 13, 14, 19, 25, 32, 34, 40, 44, 49, 52, 57, 65, 67, 69, 74, 76, 77, 79, 86, 87, 102, 105, 106, 108, 116, 121, 122, 126]
#]
#X_test = X_test[:,[1, 6, 13, 14, 19, 25, 32, 34, 40, 44, 49, 52, 57, 65, 67, 69, 74, 76, 77, 79, 86, 87, 102, 105, 106, 108, 116, 121, 122, 126]
#]

clf.fit(X_train, y_train)

# Prediction
y_pred_train =  clf.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)

y_pred = clf.predict(X_test)

np.savetxt('y_pred.txt', y_pred, fmt='%d')