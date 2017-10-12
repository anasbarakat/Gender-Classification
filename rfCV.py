import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

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

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=200, oob_score = True, random_state =50, min_samples_leaf = 50)

param_grid = {
    #'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1,5,10,50,100]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)