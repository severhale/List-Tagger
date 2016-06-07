from __future__ import division
import nltk
import numpy as np
import locale
import xml.etree.ElementTree as ET
from lxml import etree
import string
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
import os
import datahelper

class Type:
	POETRY = 3
	PROSE = 2
	MIXED = 1
	LIST = 0

save_classification = 0
save_data = 0
line_start = 335
line_end = 336
value = Type.MIXED
name, nums, text = datahelper.tokenized_from_xml_range("/home/severhal/List-Tagger/Mixed/acd5869.0032.001.umich.edu_djvu.xml", line_start, line_end)


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
feature_names = ['Numbers', 'Determiners', 'Proper Nouns']
target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])
pos_tags = ['CD', 'DT', 'NNP']
X = datahelper.get_data(text, pos_tags)
Y = np.ones(len(X), dtype=np.int) # if file is PROSE
Y *= value

# code to write data to classification file        ***DO NOT DELETE***
if save_classification!=0:
	f = open('classification', 'a')
	for i in range(len(X)):
		f.write("%s_%s\t%s\n" % (name, nums[i], Y[i]))
	f.close()

# code to write feature data to training_data file        ***DO NOT DELETE***
if save_data!=0:
	nums = np.array(nums)
	nums = np.reshape(nums, (len(nums), 1))
	final_table = np.hstack((nums, X))
	f = open('training_data', 'ab')
	np.savetxt(f, final_table, fmt=name + "_%s\t" + "1:%s\t2:%s\t3:%s")
	f.close()

# # code to display data viz
# XY = np.hstack((X,target_names[Y][:,np.newaxis]))
# df = pd.DataFrame(XY)
# df.columns= feature_names + ["Class"]
# df[feature_names] = df[feature_names].astype(float)
# print df.corr()
# df.hist(bins=np.arange(0,1,0.005),sharex=True);
# scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
# plt.show()