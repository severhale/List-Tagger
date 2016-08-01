## cluster_classify.py
## Usage: python cluster_classify.py <input_file> <output_file>
## Given a feature vector file <input_file>, classify each line and output to <output_file>

import sys
import pickle
import gzip
import time
import os
import numpy as np
from sklearn.externals import joblib
from poetryhelper import *

n=5

if len(sys.argv) != 3:
	print "Invalid arguments! Must have input and output files only."
	exit()

clf = joblib.load("Model/treemodel.pkl")

out_name = sys.argv[2]
if os.path.dirname(out_name)!='' and not os.path.exists(os.path.dirname(out_name)):
	try:
		os.makedirs(os.path.dirname(out_name))
	except:
		print "Couldn't create directory!"

X = []
tags = []
mils = 0
with (gzip.open(sys.argv[1], 'r') if sys.argv[1].endswith('.gz') else open(sys.argv[1], 'r')) as f:
	for line in f:
		tok = line.split('\t')
		tag = tok[0]
		data = []
		for i in range(1, len(tok)-1):
			data.append(float(tok[i].split(':')[1]))
		X.append(data)
		tags.append(tag)
		if len(X) >= 500000:
			mils += .5
			print "%.1f million lines" % mils
			X = make_feature_vecs(n, X)
			Y = clf.predict_proba(X)[:,1]
			with open(out_name, 'a') as out:
				for i in range(len(Y)):
					out.write("%s %.4f\n" % (tags[i], Y[i]))
			X = []
			tags = []
	X = make_feature_vecs(n, X)
	Y = clf.predict_proba(X)[:,1]
	with open(out_name, 'a') as out:
		for i in range(len(Y)):
			out.write("%s %.4f\n" % (tags[i], Y[i]))