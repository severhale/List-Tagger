from __future__ import division
import pickle
import os
import os.path
import operator
from poetryhelper import *

dictname = 'worddict.p'

if os.path.isfile(dictname):
	d = pickle.load(open(dictname, 'rb'))
else:
	d = {}
# d={}

total = sum(d.itervalues())
s = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
for i in s:
	try:
		print i[0].encode('utf-8'),':',i[1]/total
	except:
		continue


# startpage = 28
# endpage = 429
# fname = "../All Text/anthologyofengli00whit_djvu.xml"

# pages = get_pages(fname, startpage, endpage)
# words = []
# for page in pages:
# 	words += [i.text for i in page.findall(".//WORD")]
# for i in os.listdir(os.getcwd() + "/../Poetry"):
# 	words = []
# 	with open("../Poetry/" + i, 'r') as f:
# 		words = f.read().split()
# 	for w in words:
# 		word = clean_word(w)
# 		if not word in d:
# 			d[word] = 1
# 		else:
# 			d[word] += 1

# # print d

# pickle.dump(d, open(dictname, 'wb'))