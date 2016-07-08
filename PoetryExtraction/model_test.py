import poetryhelper
import pickle
import time
import os
import numpy as np
from sklearn.externals import joblib

fname = '../../500Poems'

clf = joblib.load("Model/treemodel.pkl")
freq_dict = pickle.load(open('worddict.p', 'rb'))

out_name = '500 Poems Results/ambiguous_results'
if not os.path.exists(os.path.dirname(out_name)):
	try:
		os.makedirs(os.path.dirname(out_name))
	except:
		print "Couldn't create directory!"

book_counter = 0
start_time = time.time()
for book_file in os.listdir(os.getcwd() + '/' + fname):
	try:
		pg_iter = poetryhelper.get_pg_iterator(fname + '/' + book_file)
		pages = list(pg_iter)
		pg_nums = poetryhelper.get_page_numbers(pages)
		tags, data = poetryhelper.easy_feature_table(pages, freq_dict)
		results = clf.predict_proba(data)[:,1]

		posmask = (results<=.65) & (results >=.5)
		negmask = (results<=.5) & (results>=.35)
		poems = results[posmask]
		nonpoems = results[negmask]
		ptags = tags[posmask]
		ntags = tags[negmask]
		with open(out_name+"_positive", 'a') as f:
			for i in range(len(ptags)):
				book, pg_num, line_num = poetryhelper.parse_tag(ptags[i])
				page = pages[pg_nums.index(pg_num)]
				f.write("%s:%.4f:%s\n" % (ptags[i],poems[i],poetryhelper.get_line_text(poetryhelper.get_line(page, int(line_num)))))
		with open(out_name+"_negative", 'a') as f:
			for i in range(len(ntags)):
				book, pg_num, line_num = poetryhelper.parse_tag(ntags[i])
				page = pages[pg_nums.index(pg_num)]
				f.write("%s:%.4f:%s\n" % (ntags[i],nonpoems[i],poetryhelper.get_line_text(poetryhelper.get_line(page, int(line_num)))))
		book_counter += 1
		print "Book count is %d after %.2f minutes" % (book_counter, (time.time() - start_time)/60)
	except:
		print "Error with book %s" % book