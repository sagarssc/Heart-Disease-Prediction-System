import pandas as pd 
import numpy as np
#import random as rnd
import sys
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('processed_cleveland_data_train.csv')
test_data = pd.read_csv('processed_cleveland_data_test.csv')
testing = pd.read_csv('test1.csv')
# Parameter of interest is num, for now this is a binary problem
def make_prediction_var_binary(df):
    #print df
    df['num'] = df['num'].replace([1, 2, 3, 4, 5, 6], 1)
    #print df
        
make_prediction_var_binary(train_data)
make_prediction_var_binary(test_data)
#make_prediction_var_binary(testing)

#train_data.sample(5)

#sns.pointplot(x='age', y='num', data=train_data)


#train_data.head()

from sklearn.model_selection import train_test_split

X_train = train_data.drop(['num'], axis=1)
y_train = train_data['num']

input1 = testing
X_test = test_data.drop(['num'], axis=1)
y_test = test_data['num']

#print(y_train.head())



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose classifer 
clf = RandomForestClassifier()
#print clf
# Choose Parameters and stuff
parameters = {'n_estimators': [4,6,9],
             'max_features': ['log2', 'sqrt', 'auto'],
             'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 5, 10],
             'min_samples_split': [2, 3, 5],
             'min_samples_leaf': [1, 5, 8]}

# Type of scoring to compare parameter combos 
acc_scorer = make_scorer(accuracy_score)
#print acc_scorer
#print('after grid')
# Run grid search 
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
#print grid_obj,"\n\n\n\n\n\n\n"
grid_obj = grid_obj.fit(X_train, y_train)
#print grid_obj,"\n\n\n\n\n\n\n"
# Pick the best combination of parameters
clf = grid_obj.best_estimator_
#print clf
# Fit the best algorithm to the data 
clf.fit(X_train, y_train)
# In[13]:
predictions = clf.predict(X_test)
predictions2 = clf.predict(input1)
#print(y_test,"  ",predictions)
print"accuracy ====>",accuracy_score(y_test, predictions)

# Cross Validation with KFold
from sklearn.cross_validation import KFold

all_data = pd.read_csv('processed_cleveland_data.csv')
make_prediction_var_binary(all_data)

X_all = all_data.drop(['num'], axis=1)
y_all = all_data['num']


y_all.head()

def run_kfold(clf):
    kf = KFold(297, n_folds=5)
    outcomes = []
    fold = 0
    #print "kf"
    #print kf
    for train_index, test_index in kf:
        #print "train_index======>",train_index,"     test_index=====>",test_index
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        #predictions2 = clf.predict(input1)
        #print('Result\n',predictions2)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy)) 
        mean_outcome = np.mean(outcomes)
        #print("Mean Accuracy: {0}".format(mean_outcome)) 
    print("Mean Accuracy with KFold CrossValidation:{0}".format(mean_outcome))
    predictions3 = clf.predict(input1)
    #print"By Random Forest and GridSearchCV	::",predictions2
    print"By Random Forest and KFold       	::",predictions3
run_kfold(clf)
    

