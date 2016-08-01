## This was basically my personal notepad while working on stuff. I just called it a python file so it would format nicely but not all of it is code.


results = []
for i in range(X.shape[1]):
#	tXL = np.delete(XL, i, axis=1)
#	tXT = np.delete(XT, i, axis=1)
	tX = np.delete(X, i, axis=1)
	results.append(cvtest(10, treeclf, tX, Y, names)[0])
#	results.append(sklearn.metrics.f1_score(YT, treeclf.predict(tXT), average=None)[1])

# treeclf = treeclf.fit(XL, YL)
# baseline = sklearn.metrics.f1_score(YT, treeclf.predict(XT), average=None)[1]

baseline = cvtest(10, treeclf, X, Y, names)


plt.bar(range(len(means)), means, color='Tomato', yerr=stds, error_kw={'ecolor':'MediumSlateBlue', 'linewidth':2})
plt.xticks(range(len(means)), feature_names, rotation=70)
plt.axis([0, 34, min(means)-max(stds)-.01, max(means)+max(stds)+.01])
plt.gca().set_autoscale_on(False)

for i in range(2, 53, 5):
	treeclf.n_estimators = i
	print "%d trees" % i
	print cvtest(5, treeclf, X, Y, names)

def fimps(names):
	return [treeclf.feature_importances_[feature_names[f]] for f in names]

def feat(n):
	prev = [["%s%dp" % (f, n-i) for f in f_context] for i in range(n)]
	curr = [f_context]
	next = [["%s%dn" % (f, i) for f in f_context] for i in range(1, n+1)]
	return prev + [f_context] + next + [f_line]

def plot(features, n):
	plt.boxplot(features, labels=range(-n, n+1)+['c'])
	plt.xlabel('Feature Type')
	plt.ylabel('Feature Importance')
	plt.title('Adjacent Line Feature Importances')


def ablation_test(n):
	feature_names = get_feature_names(n)
	margins = [f for f in feature_names if 'marg' in f]
	length = [f for f in feature_names if (('line_length' in f) or ('syl' in f) or ('plength' in f))]
	linenum = [f for f in feature_names if f=='linenum' or f=='lines_remaining']
	pos = [f for f in feature_names if 'pnoun' in f or 'nums' in f or 'adj' in f or 'det' in f]
	cap = [f for f in feature_names if 'cap' in f]
	prob = [f for f in feature_names if 'prob' in f]
	features = [margins, length, linenum, pos, cap, prob]
	n = ['Margin', 'Length', 'Line Number', 'Part of Speech', 'Capitalization', 'Language Model']
	results = []
	tXL = []
	tXT = []
	for set,title in zip(features, n):
		if title==n[0]:
			tXL = XL[:,[feature_names[i] for i in set]]
			tXT = XT[:,[feature_names[i] for i in set]]
		else:
			tXL = np.hstack((tXL, XL[:,[feature_names[i] for i in set]]))
			tXT = np.hstack((tXT, XT[:,[feature_names[i] for i in set]]))
#		tXL = XL[:,[feature_names[i] for i in set]]
#		tXT = np.asarray(XT)[:,[feature_names[i] for i in set]]
#		tXL = np.delete(XL, [feature_names[i] for i in set], axis=1)
#		tXT = np.delete(XT, [feature_names[i] for i in set], axis=1)
		print "Removing all " + title + " features."
		clf = sklearn.base.clone(treeclf)
		clf.fit(tXL, YL)
		results.append(sklearn.metrics.f1_score(YT, clf.predict(tXT), average='binary', pos_label=2).tolist())
	return results

n = ['Margin', 'Length', 'Line Number', 'Part of Speech', 'Capitalization', 'Language Model']
results = ablation_test(5)
results = np.hstack((np.reshape(n, (-1,1)), np.reshape(results, (-1,1))))
treeclf.fit(XL, YL)
results = np.vstack((results, ['Baseline'] + [sklearn.metrics.f1_score(YT, treeclf.predict(XT), average='binary', pos_label=2)]))
print results

results = []
for book in np.unique(books):
	mask = books==book
#	pos_label = 2 if len(np.unique(YT[mask]))>1 else 0
#	results.append(sklearn.metrics.f1_score(YT[mask], treeclf.predict(XT[mask]), pos_label=pos_label))
	results.append(sklearn.metrics.accuracy_score(YT[mask], treeclf.predict(XT[mask])))

from sklearn.metrics import *
Y[Y==2]=1

def test(c):
	scorers = [accuracy_score, precision_score, recall_score, f1_score]
	results = []
	n=5
	labels = [l[0] for l in map(parse_tag, names)]
	lkf = LabelKFold(labels, n_folds=n)
	for train,test in lkf:
		results.append([])
		clf = sklearn.base.clone(c)
		clf.fit(X[train], Y[train])
		for s in scorers:
			results[-1].append(s(Y[test], clf.predict(X[test])))
	return results

RANDOM FOREST RESULTS
[['Accuracy', '0.952937830063'],
 ['Precision', '0.965771255298'],
 ['Recall', '0.849879554121'],
 ['F-Score', '0.898071773666']]

GRADIENT BOOSTED RESULTS
[['Accuracy', '0.9549689948'],
 ['Precision', '0.936072193248'],
 ['Recall', '0.8810601367'],
 ['F-Score', '0.9029883862']]

 LOGISTIC REGRESSION (C=10 from GridSearchCV [.01, .1, 1, 10, 100])
 [ 0.92689499,  0.88974599,  0.88322987,  0.88440956]

 ADA BOOST (n=100)
 [ 0.94110117,  0.91880302,  0.89443726,  0.90459537]

 EXTREMELY RANDOM FOREST (n=50)
 [ 0.9437804 ,  0.96973656,  0.85940657,  0.91025596]
 n=100
 [ 0.94418249,  0.97050493,  0.86046115,  0.91129509]
 n=1000
 

def load_data(classification, n):
	all_lines = classification[:,0]
	books = np.vectorize(lambda x:parse_tag(x)[0])(all_lines)
	Y = np.vectorize(int)(classification[:,1])
	Y[Y!=0]=2
	resultsX = None
	resultsY = None
	resultsN = None
	for book in np.unique(books):
		fname = "../AllText/" + book + "_data"
		data = np.loadtxt(fname, dtype='string')
		names = data[:,0]
		X = data[:,1:]
		X = np.vectorize(lambda x:float(x.split(':')[1]))(X)
		lines = all_lines[books==book]
		sortinds = np.argsort(names)
		names = names[sortinds]
		X = X[sortinds]
		inds = np.searchsorted(names, lines, side='left')
		X = make_feature_vecs(n, X.tolist(), inds)
		if resultsX is None:
			resultsX = X
		else:
			resultsX = np.vstack((resultsX, X))
		if resultsY is None:
			resultsY = Y[books==book]
		else:
			resultsY = np.hstack((resultsY, Y[books==book]))
		if resultsN is None:
			resultsN = lines
		else:
			resultsN = np.hstack((resultsN, lines))
	return resultsX, resultsY, resultsN

import numpy as np
from poetryhelper import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

classification = np.loadtxt('classification', dtype='string')
treeclf = RandomForestClassifier(n_estimators=45, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=42)
results = []
for i in range(5, 10):
	X, Y, N = load_data(classification, i)
	results.append(cvtest(5, treeclf, X, Y, N))

NEIGHBORS FROM 0 to 9
[ 0.89151922,  0.90499038,  0.90809055,  0.90966025,  0.90959273,
  0.9099101 ,  0.9084678 ,  0.90769665,  0.90826293,  0.90439632]

import numpy as np
from poetryhelper import *
with open('classification', 'r') as f:
	lines = f.readlines()
lines = np.asarray([parse_tag(l.split()[0]) for l in lines])
pgs = np.unique(['_'.join(l) for l in lines[:,:2]])
pglines = np.asarray([p+'_0' for p in pgs])
pages = pages_from_names(pglines)

d = {
	' ':0
	'w':1,
	'e':2,
	'r':3,
	't':4,
	'a':5,
}

missing = []
# try:
for i in range(len(pages)):
	book,page = parse_tag(pglines[i])[:2]
	classified = lines[(lines[:,0]==book) & (lines[:,1]==page), 2]
	classified = np.vectorize(int)(classified)
	actual = get_all_lines(pages[i])
	if len(actual)-len(classified) != 0:
		missing = [i for in range(len(actual)) if i not in classified]
		for i in missing:
			line = get_line_text(actual[i])
			c = raw_input(line)

		missing += [book+'_'+page+'_'+"%d" % i for i in range(actual) if i not in classified]
# except:
# 	print "Problem. Exiting loop."