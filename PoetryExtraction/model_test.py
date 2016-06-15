import poetryhelper
import pickle
from sklearn.externals import joblib

fname = '../All Text/poemsofwaltwhit00whit_djvu.xml'

clf = joblib.load("Model/model.pkl")
freq_dict = pickle.load(open('worddict.p', 'rb'))

guesses = []

pg_iter = poetryhelper.get_pg_iterator(fname)
pages = list(pg_iter)
pg_nums = poetryhelper.get_page_numbers(pages)
tags, data = poetryhelper.easy_feature_table(pages, freq_dict)
data = poetryhelper.concat_neighbors(data, 2)
results = clf.predict_proba(data)[:,1]

for tag in tags[results==.5]:
	book, pg_num, line_num = poetryhelper.parse_tag(tag)
	page = pages[pg_nums.index(pg_num)]
	print pg_num,line_num,":",poetryhelper.get_line_text(poetryhelper.get_line(page, int(line_num)))