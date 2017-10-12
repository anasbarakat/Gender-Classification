from sklearn import svm
import numpy as np
import pandas as pd

from sigopt_sklearn.search import SigOptSearchCV
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


client_token = 'GFJDYYPTYCVYTFQVLUMHBPHUIQXQGXWHASZNAZHJIWJWXRWY'


# Define domains for the Random Forest parameters
random_forest_parameters = dict(
                                max_features=[1,  128],
                                n_estimators=[1, 100],
                                min_samples_leaf=[1, 10],
                                )

# define sklearn estimator
random_forest = RandomForestClassifier()

# define SigOptCV search strategy
clf = SigOptSearchCV(
                     random_forest,
                     random_forest_parameters,
                     cv=5,
                     client_token=client_token,
                     n_iter=60
                     )

time1 = time()
clf.fit(X_train, y_train)
time2 = time()

# Prediction
y_pred_train =  clfSVM.predict(X_train)

# Compute the score
score = compute_pred_score(y_train, y_pred_train)
print('Score sur le train : %s' % score)

# Prediction
y_pred = clf.predict(X_test)

proba = clf.predict_proba(X_test)

# predict 0 when the probabilities of predicting -1 and 1 are close/ thresholding
#
for k in range(1,y_pred.shape[0]):
    if 0.4<=proba[k,0]<=0.6:
        y_pred[k] = 0

np.savetxt('y_pred26.txt', y_pred, fmt='%d')

print("fitting time:", time2 - time1)

print(clf.best_score_) #contains CV score for best found estimator
print(clf.best_params_ )#contains best found param configuration