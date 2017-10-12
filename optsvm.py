import optunity
import optunity.metrics
from sklearn.utils import shuffle



# comment this line if you are running the notebook
import sklearn.svm
import numpy as np
import pandas as pd

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



X, y = shuffle(X_train, y_train, random_state=0)

X = X[:10000,:]
y = y[:10000]

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x= X_train, y= y_train, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

time3 = time()

print("perform tuning...")
# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-3, 3], logGamma=[-5, 1], pmap = optunity.pmap)

time1 = time()


print("optimization time:", time1 - time3)

print()
print("train model on the full training set with tuned hyperparameters...")
# train model on the full training set with tuned hyperparameters
clfSVM = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X, y)
print()
print("fitting finished")
# Prediction
y_pred_train =  clfSVM.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)

# Prediction
y_pred = clfSVM.predict(X_test)

proba = clfSVM.predict_proba(X_test)

# predict 0 when the probabilities of predicting -1 and 1 are close/ thresholding
#
for k in range(1,y_pred.shape[0]):
    if 0.4<=proba[k,0]<=0.6:
        y_pred[k] = 0

np.savetxt('y_pred25.txt', y_pred, fmt='%d')

time2 = time()

print("fitting time:", time2 - time1)
