import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from time import time

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



rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, min_samples_leaf = 1)

params = {
    'max_depth': [1,3,7,8,12,None],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1,5,10,50,100,200,500],
    'n_estimators': [50, 100, 200, 500, 1000]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=params, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)