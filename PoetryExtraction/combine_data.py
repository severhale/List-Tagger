## combiner_data.py
## This script loads classification data ('classification') and files containing all line data for each book in the classification data.
## It creates a final feature file for each line in the classification file.
## Data files are expected to be in the format '<data_folder>/<book>_data'
## n determines the number of context lines on each side of a line when creating feature vectors

import numpy as np
from poetryhelper import *

n=5
data_folder = '../AllText'

fname = 'classification'
classification = np.loadtxt(fname, dtype='string')
all_lines = classification[:,0]
books = np.vectorize(lambda x:parse_tag(x)[0])(all_lines)
Y = np.vectorize(int)(classification[:,1])

with open('training_data', 'w') as f:
	pass

for book in np.unique(books):
	fname = data_folder + '/' + book + "_data"
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