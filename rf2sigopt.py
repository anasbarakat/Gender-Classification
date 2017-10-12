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


# Use SigOpt to tune a Random Forest Classifier in Python
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python

# Run `pip install sigopt` to download the python API client
# Run `pip install sklearn` to install scikit-learn, a machine learning
# library in Python (http://scikit-learn.org)
from sigopt import Connection
from sklearn import cross_validation, datasets
from sklearn.ensemble import RandomForestClassifier
import numpy

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
conn = Connection(client_token= client_token)



# Create a SigOpt experiment for the Random Forest parameters
experiment = conn.experiments().create(
                                       name="Random Forest (Python)",
                                       parameters=[
                                                   dict(name="max_features", type="int", bounds=dict(min=1, max=128)),
                                                   dict(name="n_estimators", type="int", bounds=dict(min=1, max=100)),
                                                   dict(name="min_samples_leaf", type="int", bounds=dict(min=1, max=10))
                                                   ]
                                       )
print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

# Our object metric is the mean of cross validated accuracies
# We use cross validation to prevent overfitting
def evaluate_model(assignments, X, y):
    # evaluate cross folds for accuracy
    cv = cross_validation.ShuffleSplit(
                                       X.shape[0],
                                       n_iter=5,
                                       test_size=0.3,
                                       )
    classifier = RandomForestClassifier(
                                            n_estimators=assignments['n_estimators'],
                                            max_features=assignments['max_features'],
                                            min_samples_leaf=assignments['min_samples_leaf']
                                            )
    cv_accuracies = cross_validation.cross_val_score(classifier, X, y, cv=cv)
    return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

# Run the Optimization Loop between 10x - 20x the number of parameters
for _ in range(60):
    # Receive a Suggestion from SigOpt
    suggestion = conn.experiments(experiment.id).suggestions().create()
    
    # Evaluate the model locally
    (value, std) = evaluate_model(suggestion.assignments, X_train, y_train)
    
    # Report an Observation (with standard deviation) back to SigOpt
    conn.experiments(experiment.id).observations().create(
                                                          suggestion=suggestion.id,
                                                          value=value,
                                                          value_stddev=std,
                                                          )

# Re-fetch the best observed value and assignments
best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

# To wrap up the Experiment, fit the RandomForest on the best assigments
# and train on all available data
clf = RandomForestClassifier(
                            n_estimators=best_assignments['n_estimators'],
                            max_features=best_assignments['max_features'],
                            min_samples_leaf=best_assignments['min_samples_leaf']
                            )

time1 = time()
clf.fit(X_train, y_train)
time2 = time()


# Prediction
y_pred_train =  clf.predict(X_train)

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