import numpy as np
from poetryhelper import *

n=5

fname = 'classification'
classification = np.loadtxt(fname, dtype='string')
all_lines = classification[:,0]
books = np.vectorize(lambda x:parse_tag(x)[0])(all_lines)
Y = np.vectorize(int)(classification[:,1])

with open('training_data', 'w') as f:
	pass

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

	save_data(np.asarray(X), lines, 'training_data', mode='a')