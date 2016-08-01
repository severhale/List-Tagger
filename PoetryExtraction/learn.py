## learn.py
## Given a file in svmlight format, trains an ML model and outputs the model to Model/treemodel.pkl.

from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from poetryhelper import *


#### load svm data file
X,Y = load_svmlight_file('joined_data')
target_names = np.array(["Non-Poetry", "Begin Poem", "Middle Poem", "End Poem"])
X = X.toarray()
Y = Y.astype(int)
#### Classification includes extra info like begin poem, end poem. That's unnecessary for now.
Y[Y>3] = 0
Y[Y==1] = 2
Y[Y==3] = 2

#### load datapoint names
names = []
with open('joined_data') as f:
	lines = f.readlines()
for line in lines:
	pos = line.find('#') + 2
	names.append(line[pos:-1]) # -1 to get rid of newline char
names = np.asarray(names)

# Split data and train model
XL, XT, YL, YT, NL, NT = split_books(X, Y, names)
YL = np.asarray(YL)
YT = np.asarray(YT)
treeclf = RandomForestClassifier(n_estimators=45, criterion='entropy', max_features='auto', bootstrap=True, oob_score=True, n_jobs=2, class_weight="balanced", random_state=42)
treeclf.fit(XL, YL)
Yhat = treeclf.predict(XT)
print "======TREE PERFORMANCE======"
print_performance(YT, Yhat)

# Save model
joblib.dump(treeclf, "Model/treemodel.pkl")