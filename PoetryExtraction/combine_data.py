import numpy as np
from poetryhelper import *

n=5

fname = 'single_feature_data'
data = np.loadtxt(fname, dtype='string')
names = data[:,0]
X = data[:,1:]
X = np.vectorize(lambda x:float(x.split(':')[1]))(X)

fname = 'classification'
classification = np.loadtxt(fname, dtype='string')
lines = classification[:,0]
Y = np.vectorize(int)(classification[:,1])

sortinds = np.argsort(names)
names = names[sortinds]
X = X[sortinds]
inds = np.searchsorted(names, lines, side='left')

X = make_feature_vecs(n, X.tolist(), inds)

save_data(np.asarray(X), lines, mode='w')