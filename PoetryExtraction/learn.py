from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
import random

#### load svm data file
X,Y = load_svmlight_file('joined_data')
feature_names = ['L_Margin', 'R_Margin', 'T_Margin', 'B_Margin', 'Syllables', 'Prv_L_Margin', 'Prv_R_Margin', 'Prv_T_Margin', 'Prv_B_Margin', 'Prv_Syllables', 'Nxt_L_Margin', 'Nxt_R_Margin', 'Nxt_T_Margin', 'Nxt_B_Margin', 'Nxt_Syllables']
target_names = np.array(["Non-Poetry", "Poetry"])
X = X.toarray()
Y = Y.astype(int)
XY = np.hstack((X,target_names[np.array(Y)][:,np.newaxis]))
df = pd.DataFrame(XY)
df.columns= feature_names + ["Class"]
df[feature_names] = df[feature_names].astype(float)

#### load datapoint names
names = []
with open('joined_data') as f:
	lines = f.readlines()
for line in lines:
	pos = line.find('#') + 2
	names.append(line[pos:-1]) # -1 to get rid of newline char
names = np.asarray(names)


#### 5-fold crossvalidation w/rbf model
param_grid = [{'C': [0.01,0.1,1, 10, 100], 'kernel': ['rbf'],'gamma': [0.1,1,10,100]}]
Xlearn,Xtest,Ylearn,Ytest,names_learn,names_test = cross_validation.train_test_split(X, Y, names, test_size=0.25, random_state=random.randint(1, 100))
clf = GridSearchCV(svm.SVC(C=1, probability=True), param_grid, cv=5)
clf.fit(Xlearn, Ylearn)

#### create best guess for test data
Yhat = clf.predict(Xtest)
prob = clf.predict_proba(Xtest)
confusion = metrics.confusion_matrix(Ytest, Yhat)
Err  = 1 - metrics.accuracy_score(Ytest, Yhat)
F1 = metrics.f1_score(Ytest, Yhat, average=None)

print "Predicted:",Yhat
print "Actual:",Ytest
print "F1: ",F1
print("Test Error Rate is: %.4f"%(Err,))
print "Confusion Matrix"
print confusion
print "Probabilities"
print np.hstack((np.reshape(names_test, (len(names_test), 1))[Ytest==1], prob[Ytest==1]))