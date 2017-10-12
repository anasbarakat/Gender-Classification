import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC

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


# SVC fitting
svc = SVC(C=1)
#clfSVM = model.fit(X_train, y_train)


# Grid Searh cross val
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['polynomial'], 'C': [1, 10, 100, 1000]}]

CV_rfc = GridSearchCV(estimator= svc, param_grid= tuned_parameters, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)