import poetryhelper
import pickle
import os
from sklearn.externals import joblib

fname = '../Test'

clf = joblib.load("Model/model.pkl")
freq_dict = pickle.load(open('worddict.p', 'rb'))

out_name = '../Test/results'

for book in os.listdir(os.getcwd() + '/' + fname):
	pg_iter = poetryhelper.get_pg_iterator(fname + '/' + book)
	pages = list(pg_iter)
	pg_nums = poetryhelper.get_page_numbers(pages)
	tags, data = poetryhelper.easy_feature_table(pages, freq_dict)
	data = poetryhelper.concat_neighbors(data, 2)
	results = clf.predict_proba(data)[:,1]

	f = open(out_name, 'a')
	mask = ((results >= .5))
	poems = results[mask]
	ptags = tags[mask]
	for i in range(len(ptags)):
		book, pg_num, line_num = poetryhelper.parse_tag(ptags[i])
		page = pages[pg_nums.index(pg_num)]
		f.write("%s:%.4f:%s\n" % (ptags[i],poems[i],poetryhelper.get_line_text(poetryhelper.get_line(page, int(line_num)))))
	f.close()