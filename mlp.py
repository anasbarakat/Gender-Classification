import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier

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



#Multi Layer Perceptron

time1 = time()
clfMLP = MLPClassifier(solver='adam',alpha=1e-4,hidden_layer_sizes=(500,), random_state=1)
clfMLP.fit(X_train, y_train)

y_pred = clfMLP.predict(X_test)

proba = clfMLP.predict_proba(X_test)
#print(proba[0:10,])

for k in range(1,y_pred.shape[0]):
    if 0.4<proba[k,1]<0.6:
        y_pred[k] = 0

time2 = time()

print("execution time:", time2 - time1)
np.savetxt('y_pred.txt', y_pred, fmt='%d')