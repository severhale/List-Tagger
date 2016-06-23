from sklearn.datasets import load_svmlight_file
# from sklearn import metrics
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
from poetryhelper import *
import time;

visualize = False

#### load svm data file
X,Y = load_svmlight_file('joined_data')
target_names = np.array(["Non-Poetry", "Begin Poem", "Middle Poem", "End Poem", "Title", "Author"])
X = X.toarray()
Y = Y.astype(int)
if visualize:
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

#### 3-fold crossvalidation w/rbf model
# param_grid = {'c1':scipy.stats.expon(scale=.5), 'c2':scipy.stats.expon(scale=.05)}


clf = sklearn_crfsuite.CRF(algorithm='pa', all_possible_transitions=True)
f1_scorer = make_scorer(metrics.flat_f1_score, average='macro', labels=target_names)
start = time.clock()
# clf = RandomizedSearchCV(crf, param_grid, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer)
print "Beginning data format"
start = time.clock()
X, Y, names = crfformat(X, Y, names)
print "Finished in %.4f seconds" % (time.clock() - start)

Xlearn, Xtest, Ylearn, Ytest, names_learn, names_test = splitcrf(X, Y, names, .25)

print "Beginning fit"
start = time.clock()
clf.fit(Xlearn, Ylearn)
print "Finished in %.4f seconds" % (time.clock() - start)
# print clf.feature_importances_

#### create best guess for test data
print "Beginning prediction"
start = time.clock()
Yhat = clf.predict(Xtest)
print "Finished prediction in %.4f seconds" % (time.clock() - start)
# prob = clf.predict_marginal(Xtest)
print metrics.flat_accuracy_score(Ytest, Yhat)
confusion = sklearn.metrics.confusion_matrix(lsum(Ytest), lsum(Yhat))
# Err  = 1 - metrics.accuracy_score(Ytest, Yhat)
F1 = metrics.flat_f1_score(Ytest, Yhat, average=None)

# print "Predicted:",Yhat
# print "Actual:",Ytest
print "F1: ",F1
# print("Test Error Rate is: %.4f"%(Err,))
print "Confusion Matrix"
print confusion
# print "Probabilities"
# print np.hstack((np.reshape(names_test, (len(names_test), 1))[Ytest==1], prob[Ytest==1]))

treeclf = RandomForestClassifier(n_estimators=20, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=random.randint(1, 100))
X, Y, names = uncrfformat(X, Y, names)
Xlearn, Xtest, Ylearn, Ytest, names_learn, names_test = sklearn.cross_validation.train_test_split(X, Y, names, test_size=.25, random_state = random.randint(1, 100))
treeclf.fit(Xlearn, Ylearn)
Yhat = treeclf.predict(Xtest)
confusion = sklearn.metrics.confusion_matrix(Ytest, Yhat)
F1 = sklearn.metrics.f1_score(Ytest, Yhat, average=None)
print "Tree Confusion"
print confusion
print "Tree F1"
print F1

if visualize:
	scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
	plt.figure()
	andrews_curves(df, 'Class')
	plt.show()

joblib.dump(clf, "Model/model.pkl")