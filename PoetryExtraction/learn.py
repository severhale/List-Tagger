from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix, andrews_curves
import random


#### load svm data file
X,Y = load_svmlight_file('joined_data')
target_names = np.array(["Non-Poetry", "Poetry"])
X = X.toarray()
Y = Y.astype(int)
XY = np.hstack((X,target_names[np.array(Y)][:,np.newaxis]))
feature_names = range(len(X[0]))
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

# X = X[:,[1,2]]

#### 5-fold crossvalidation w/rbf model
# param_grid = [{'C':[10], 'kernel': ['linear']}]
# param_grid = [{'C': [0.01,0.1,1, 10], 'kernel': ['rbf'],'gamma': [0.1,1,10]}]
Xlearn,Xtest,Ylearn,Ytest,names_learn,names_test = cross_validation.train_test_split(X, Y, names, test_size=0.25, random_state=random.randint(1, 100))
# clf = GridSearchCV(svm.SVC(C=1, probability=True), param_grid, cv=2)
clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, random_state=random.randint(1, 100))
# clf = GridSearchCV(clf, param_grid, cv=2)
clf.fit(Xlearn, Ylearn)
print clf.feature_importances_

#### create best guess for test data
Yhat = clf.predict(Xtest)
prob = clf.predict_proba(Xtest)
confusion = metrics.confusion_matrix(Ytest, Yhat)
Err  = 1 - metrics.accuracy_score(Ytest, Yhat)
F1 = metrics.f1_score(Ytest, Yhat, average=None)

# plt.figure(1)
# plot_classifier(Xlearn,Ylearn,clf,"SVM with Learning Set")

# #Plot the classification function with test set
# plt.figure(2)
# plot_classifier(Xtest,Ytest,clf,"SVM with Test Set")
# plt.show()

# print "Predicted:",Yhat
# print "Actual:",Ytest
print "F1: ",F1
print("Test Error Rate is: %.4f"%(Err,))
print "Confusion Matrix"
print confusion
# print "Probabilities"
# print np.hstack((np.reshape(names_test, (len(names_test), 1))[Ytest==1], prob[Ytest==1]))
# scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
# plt.figure()
# andrews_curves(df, 'Class')
# plt.show()

joblib.dump(clf, "Model/model.pkl")