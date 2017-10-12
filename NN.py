
import numpy as np
import pandas as pd

#for neural network models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD

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

input_num_units = 128
hidden1_num_units = 40
hidden2_num_units = 40
hidden3_num_units = 40
hidden4_num_units = 40
hidden5_num_units = 40
hidden6_num_units = 40
output_num_units = 1

#epochs = 5 #score 0.265301318267
#epochs = 10 #score  0.243644067797

epochs = 10

batch_size = 20

dropout_ratio = 0.2

# create model
model = Sequential([
                    Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
                    Dropout(dropout_ratio),
                    Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu'),
                    Dropout(dropout_ratio),
                    Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu'),
                    Dropout(dropout_ratio),
                    Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu'),
                    Dropout(dropout_ratio),
                    #Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu'),
                    #Dropout(dropout_ratio),
                    
                    Dense(output_dim=output_num_units, input_dim=hidden5_num_units, activation='sigmoid'),
                    ])

# compile the model with necessary attributes
sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])

for k in range(1,y_train.shape[0]):
    if y_train[k]==-1:
        y_train[k] = 0


print("starting fitting")
# Fit the model
model.fit(X_train, y_train, nb_epoch= epochs, batch_size= batch_size,  verbose=1)
print("fitting finished")

# calculate predictions
print("calculate predictions xtest")
y_pred = model.predict_classes(X_test)

for k in range(1,y_pred.shape[0]):
    if y_pred[k]==0:
        y_pred[k] = -1

print("calculate predictions probas")
proba = model.predict_proba(X_test)

# predict 0 when the probabilities of predicting -1 and 1 are close
#
for k in range(1,y_pred.shape[0]):
    if 0.4<proba[k]<0.6:
        y_pred[k] = 0

np.savetxt('y_pred3.txt', y_pred, fmt='%d')