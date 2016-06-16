from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix, andrews_curves
import random
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import sklearn
import scipy.stats

def crfformat(X, Y):
	X1 = [{str(i):x[i] for i in range(len(x))} for x in X]
	Y1 = [str(i) for i in Y]
	return X1, Y1


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
param_grid = {'c1':scipy.stats.expon(scale=.5), 'c2':scipy.stats.expon(scale=.5)}
Xlearn,Xtest,Ylearn,Ytest,names_learn,names_test = cross_validation.train_test_split(X, Y, names, test_size=0.25, random_state=random.randint(1, 100))
# clf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, random_state=random.randint(1, 100))
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=.1, c2=.1, max_iterations=100, all_possible_transitions=True)
f1_scorer = make_scorer(metrics.flat_f1_score)
clf = RandomizedSearchCV(crf, param_grid, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer)
Xlearn, Ylearn = crfformat(Xlearn, Ylearn)
Xtest, Ytest = crfformat(Xtest, Ytest)
clf.fit(Xlearn, Ylearn)
# print clf.feature_importances_

#### create best guess for test data
Yhat = clf.predict(Xtest)
# prob = clf.predict_marginal(Xtest)
print metrics.flat_accuracy_score(Ytest, Yhat)
# confusion = metrics.confusion_matrix(Ytest, Yhat)
# Err  = 1 - metrics.accuracy_score(Ytest, Yhat)
# F1 = metrics.f1_score(Ytest, Yhat, average=None)

# plt.figure(1)
# plot_classifier(Xlearn,Ylearn,clf,"SVM with Learning Set")

# #Plot the classification function with test set
# plt.figure(2)
# plot_classifier(Xtest,Ytest,clf,"SVM with Test Set")
# plt.show()

# print "Predicted:",Yhat
# print "Actual:",Ytest
# print "F1: ",F1
# print("Test Error Rate is: %.4f"%(Err,))
# print "Confusion Matrix"
# print confusion
# print "Probabilities"
# print np.hstack((np.reshape(names_test, (len(names_test), 1))[Ytest==1], prob[Ytest==1]))
# scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
# plt.figure()
# andrews_curves(df, 'Class')
# plt.show()

joblib.dump(clf, "Model/model.pkl")