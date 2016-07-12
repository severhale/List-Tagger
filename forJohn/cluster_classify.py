import sys
import pickle
import time
import os
import numpy as np
from sklearn.externals import joblib
from poetryhelper import *

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
with open(sys.argv[1], 'r') as f:
	for line in f:
		tok = line.split('\t')
		tag = tok[0]
		data = []
		for i in range(1, len(tok)-1):
			data.append(float(tok[i].split(':')[1]))
		X.append(data)
		tags.append(tag)
		if len(X) >= 1000:
			Y = clf.predict(X)
			with open(out_name, 'a') as out:
				for i in range(len(Y)):
					out.write("%s %d\n" % (tags[i], Y[i]))
			X = []
			tags = []
	Y = clf.predict(X)
	with open(out_name, 'a') as out:
		for i in range(len(Y)):
			out.write("%s %d\n" % (tags[i], Y[i]))