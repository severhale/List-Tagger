from sklearn.datasets import load_svmlight_file
# from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
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
target_names = np.array(["Non-Poetry", "Begin Poem", "Middle Poem", "End Poem"])
X = X.toarray()
Y = Y.astype(int)
Y[Y>3] = 0

if visualize:
	XY = np.hstack((X,target_names[np.array(Y)][:,np.newaxis]))
	feature_names = range(len(X[0]))
	df = pd.DataFrame(XY)
	df.columns = feature_names + ["Class"]
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
classify_crf = False
if classify_crf:
	clf = sklearn_crfsuite.CRF(algorithm='pa', all_possible_transitions=True)
	# f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=target_names)
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
	X, Y, names = uncrfformat(X, Y, names)
pages = pages_from_names(names)
X, Y, names, pages = crfformat(X, Y, names, pages)
Xlearn, Xtest, Ylearn, Ytest, Nlearn, Ntest, Plearn, Ptest = splitcrf(X, Y, names, pages, .5)
Xdev, Xtest, Ydev, Ytest, Ndev, Ntest, Pdev, Ptest = splitcrf(Xtest, Ytest, Ntest, Ptest, .5)
Xlearn1, Xlearn2, Ylearn1, Ylearn2, Nlearn1, Nlearn2, Plearn1, Plearn2 = splitcrf(Xlearn, Ylearn, Nlearn, Plearn, .5)

XL1, YL1, NL1 = uncrfformat(Xlearn1, Ylearn1, Nlearn1)
XL2, YL2, NL2 = uncrfformat(Xlearn2, Ylearn2, Nlearn2)
treeclf = RandomForestClassifier(n_estimators=40, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=random.randint(1, 100))
treeclf.fit(XL1, YL1)
Yhat = treeclf.predict(XL2)
confusion = sklearn.metrics.confusion_matrix(YL2, Yhat)
F1 = sklearn.metrics.f1_score(YL2, Yhat, average=None)
print "Tree Confusion"
print confusion
print "Tree F1"
print F1

clf = sklearn_crfsuite.CRF(algorithm='pa')
# crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)
# params = {
# 	'c1':scipy.stats.expon(scale=.5),
# 	'c2':scipy.stats.expon(scale=.05),
# }
# f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')
# clf = RandomizedSearchCV(crf, params, cv=5, n_jobs=-1, n_iter=50, scoring=f1_scorer)

CXlearn = get_crf_data(treeclf, Xlearn2, Nlearn2, Plearn2)
CXtest = get_crf_data(treeclf, Xtest, Ntest, Ptest)
CXdev = get_crf_data(treeclf, Xdev, Ndev, Pdev)
clf.fit(CXlearn, Ylearn2, CXdev, Ydev)
CYhat = clf.predict(CXtest)

confusion = sklearn.metrics.confusion_matrix(lsum(Ytest), lsum(CYhat))
F1 = metrics.flat_f1_score(Ytest, CYhat, average=None)

print metrics.flat_accuracy_score(Ytest, CYhat)
print "F1: ",F1
print "Confusion Matrix"
print confusion

# param_grid = {'C':[.1,1,10], 'gamma':[.1,1,10]}
# svc = svm.SVC(kernel='rbf')
# svcclf = GridSearchCV(svc, param_grid)
# svcclf.fit(Xlearn, Ylearn)
# Yhat = svcclf.predict(Ytest)
# confusion = sklearn.metrics.confusion_matrix(Ytest, Yhat)
# F1 = sklearn.metrics.f1_score(Ytest, Yhat, average=None)
# print "SVC Confusion"
# print confusion
# print "SVC F1"
# print F1

if visualize:
	scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
	plt.figure()
	andrews_curves(df, 'Class')
	plt.show()

joblib.dump(treeclf, "Model/treemodel.pkl")
joblib.dump(clf, "Model/model.pkl")