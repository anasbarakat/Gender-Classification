import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

import hyperopt.tpe
import hpsklearn
import hpsklearn.demo_support
import time


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

n_estimators = 200

estimator = hpsklearn.HyperoptEstimator(
                                        preprocessing=hpsklearn.components.any_preprocessing('pp'),
                                        classifier=hpsklearn.components.any_classifier('clf'),
                                        algo=hyperopt.tpe.suggest,
                                        trial_timeout=15.0, # seconds
                                        max_evals=15,
                                        )

print('Best classifier:\n', estimator._best_learner)
