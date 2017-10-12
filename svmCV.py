import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC

from time import time

print("data loading ...")

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


print()
print("grid search...")
time1 = time()
Cs = [1, 10, 100, 1000]
gammas = [ 0.0001, 0.001, 0.01, 0.1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid_search = GridSearchCV(SVC(kernel='poly'), param_grid, cv= 5)
time2 = time()
print()
print("fitting the model...")
grid_search.fit(X_train, y_train)
print()
print("fitting finished")
tps = time2 - time1
print(grid_search.best_params_)
print()
print("execution time:", tps)
