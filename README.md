# Gender-Classification
Data challenge 


Kaggle-like data challenge consisting of predicting a person's gender using features extracted from his/her photo. Data were provided by Morpho company. 

The classification problem involves 3 classes (-1 label for woman, 1 for man and 0 for "I don't know" ). Misclassification is penalized and costs 10 points whereas predicting 0 costs only 1 point. 

Final solution proposed: 
- Binary classification with MLP classifier using sklearn library in Python. 
- Tuning the model by applying a cross validation (gridSearchCV on sklearn) to choose the best classifier parameters.
- Bagging with the MLP classifier to reduce the variance. 
- Hard thresholding of the probabilities to predict classes and choosing 0 when the prediction is uncertain.
