import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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


n_estimators = 100

# RF fitting
model = RandomForestClassifier(n_estimators=n_estimators, max_features = 0.05, n_jobs = -1, min_samples_leaf = 10)
clfRF = model.fit(X_train, y_train)

# Prediction
y_pred_train =  clfRF.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)

#print(np.mean(cross_val_score(model, X_train, y_train, cv=10, scoring = loss)))

# Prediction
y_pred = clfRF.predict(X_test)

proba = clfRF.predict_proba(X_test)
print(proba[0:10,])

# predict 0 when the probabilities of predicting -1 and 1 are close/ thresholding
#
for k in range(1,y_pred.shape[0]):
    if 0.1<=proba[k,1]<=0.9:
        y_pred[k] = 0

np.savetxt('y_pred3.txt', y_pred, fmt='%d')