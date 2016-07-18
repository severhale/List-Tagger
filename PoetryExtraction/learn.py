from sklearn.datasets import load_svmlight_file
# from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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
import time

visualize = False
CRF_EVAL = False

#### load svm data file
X,Y = load_svmlight_file('joined_data')
target_names = np.array(["Non-Poetry", "Begin Poem", "Middle Poem", "End Poem"])
X = X.toarray()
Y = Y.astype(int)
Y[Y>3] = 0
Y[Y==1] = 2
Y[Y==3] = 2

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


# pages = pages_from_names(names)
# X, Y, names, pages = crfformat(X, Y, names, pages)
	
# XL, YL, NL = uncrfformat(Xlearn1, Ylearn1, Nlearn1)
# XT, YT, NT = uncrfformat(Xlearn2, Ylearn2, Nlearn2)
XL, XT, YL, YT, NL, NT = split_books(X, Y, names)
YL = np.asarray(YL)
YT = np.asarray(YT)
treeclf = RandomForestClassifier(n_estimators=45, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=42)
# clf = AdaBoostCLassifier(n_estimators=100)
treeclf.fit(XL, YL)
Yhat = treeclf.predict(XT)
print "======TREE PERFORMANCE======"
print_performance(YT, Yhat)

if CRF_EVAL:
	# Xlearn, Xtest, Ylearn, Ytest, Nlearn, Ntest, Plearn, Ptest = splitcrf(X, Y, names, pages, .2)
	# Xdev, Xtest, Ydev, Ytest, Ndev, Ntest, Pdev, Ptest = splitcrf(Xtest, Ytest, Ntest, Ptest, .5)
	# Xlearn1, Xlearn2, Ylearn1, Ylearn2, Nlearn1, Nlearn2, Plearn1, Plearn2 = splitcrf(Xlearn, Ylearn, Nlearn, Plearn, .625)
	clf = sklearn_crfsuite.CRF(algorithm='pa')
	# crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)
	# params = {
	# 	'c1':scipy.stats.expon(scale=.5),
	# 	'c2':scipy.stats.expon(scale=.05),
	# }
	# f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')
	# clf = RandomizedSearchCV(crf, params, cv=5, n_jobs=-1, n_iter=50, scoring=f1_scorer)
	
	XL, YL, NL = uncrfformat(Xlearn1, Ylearn1, Nlearn1)
	XT, YT, NT = uncrfformat(Xlearn2, Ylearn2, Nlearn2)
	YL = np.asarray(YL)
	YT = np.asarray(YT)
	YL[YL!='0']='2'
	YT[YT!='0']='2'
	treeclf = RandomForestClassifier(n_estimators=40, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=42)
	treeclf.fit(XL, YL)

	learnguesses = [treeclf.predict(chunk) for chunk in Xlearn2]
	testguesses = [treeclf.predict(chunk) for chunk in Xtest]
	devguesses = [treeclf.predict(chunk) for chunk in Xdev]

	# p=1 # make 100% of the data noise
	# learnguesses, testguesses, devguesses = simulate_data(p)

	CXlearn = get_crf_data(learnguesses, Xlearn2, Nlearn2, Plearn2)
	CXtest = get_crf_data(testguesses, Xtest, Ntest, Ptest)
	CXdev = get_crf_data(devguesses, Xdev, Ndev, Pdev)
	clf.fit(CXlearn, Ylearn2, CXdev, Ydev)
	CYhat = clf.predict(CXtest)
	print "======CRF PERFORMANCE======"
	print_performance(lsum(Ytest), lsum(CYhat))

	# joblib.dump(clf, "Model/model.pkl")

if visualize:
	scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
	plt.figure()
	andrews_curves(df, 'Class')
	plt.show()

joblib.dump(treeclf, "Model/treemodel.pkl")