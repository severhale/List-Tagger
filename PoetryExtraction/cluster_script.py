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
freq_dict = pickle.load(open('worddict.p', 'rb'))

out_name = sys.argv[2]
if os.path.dirname(out_name)!='' and not os.path.exists(os.path.dirname(out_name)):
	try:
		os.makedirs(os.path.dirname(out_name))
	except:
		print "Couldn't create directory!"

book_counter = 0
start_time = time.time()
books = []
with open(sys.argv[1], 'r') as f:
	books = [line.strip() for line in f.readlines()]
for book_file in books:
	try:
		pages = get_pg_iterator(book_file)
		pg_nums = get_page_numbers(pages)
		tags, data = easy_feature_table(pages, freq_dict)
		results = clf.predict_proba(data)[:,1]
		with open(out_name, 'a') as f:
			for i in range(len(results)):
				f.write("%s:%.4f" % (tags[i],results[i]))
		book_counter += 1
		print "Book count is %d after %.2f minutes" % (book_counter, (time.time() - start_time)/60)
	except Exception as ex:
		template = "An exception of type {0} occured. Arguments:\n{1!r}"
		message = template.format(type(ex).__name__, ex.args)
		print message
		print "Error with book %s" % book_file