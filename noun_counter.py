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

class Type:
	POETRY = 3
	PROSE = 2
	MIXED = 1
	LIST = 0

# Count up all unique parts of speech and return two arrays
# unique contains all the different POS tags found
# counts contains the counts of each unique POS tag in the order they are found in unique
def pos_count(tokenized_text):
	tagged = nltk.pos_tag(tokenized_text)
	tags = np.array([i[1] for i in tagged])
	unique, counts = np.unique(tags, return_counts=True)
	return unique, counts

def load_text(fname):
	f = open(fname, 'r')
	text = f.read()
	f.close()
	return text

# def tokenized_from_xml(fname, pg_num):
# 	tree = ET.parse(fname)
# 	pg_iterator = tree.iter(tag="OBJECT")
# 	for i in range(pg_num-1):
# 		pg_iterator.next()
# 	page = pg_iterator.next()
# 	pg_num_tag = page.find(".//PARAM")
# 	pg_num = pg_num_tag.get('value')[-9:-5]
# 	single_page = page.findall(".//WORD")
# 	# single_page = pages[pg_num].findall(".//WORD")
# 	return pg_num, [i.text for i in single_page]

# Parse pages from pg_start to pg_end inclusive
# pg_start and pg_end are 1-indexed
def tokenized_from_xml_range(fname, pg_start, pg_end):
	pg_start -= 1
	pg_end -= 1
	tree = ET.fromstring(load_text(fname))
	name = tree.find(".//OBJECT").find(".//PARAM").get('value')[:-10]
	pg_iterator = tree.iter(tag="OBJECT")
	i=0
	while i<pg_start:
		pg_iterator.next()
		i += 1
	pages = []
	page_nums = []
	while i<=pg_end:
		page = pg_iterator.next()
		page_text = page.findall(".//WORD")
		pages.append([j.text for j in page_text])
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		print "Page",pg_num
		print pages[-1]
		page_nums.append(pg_num)
		i += 1
	return name,page_nums,pages

def tokenized_from_rawtext(fname):
	f = open(fname, 'r')
	text = f.read()
	text = text.decode('utf-8')
	return nltk.word_tokenize(text)
	f.close()

def get_data(tokenized_text_array, pos_tags):
	data = []
	for i in range(len(nums)):
		all_tags, all_data = pos_count(tokenized_text_array[i])
		page_data = []
		size = np.sum(all_data)
		for j in range(len(pos_tags)):
			index = np.argwhere(all_tags==pos_tags[j])
			if (len(index > 0) and len(index[0] > 0)):
				page_data.append(all_data[index[0][0]]/size)
			else:
				page_data.append(0)
		data.append(page_data)
	data = np.asarray(data)
	return data

save_classification = False
save_data = False
line_start = 74
line_end = 104
value = Type.POETRY
name, nums, text = tokenized_from_xml_range("/home/severhal/List-Tagger/Mixed/acd5869.0032.001.umich.edu_djvu.xml", line_start, line_end)


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
feature_names = ['Numbers', 'Determiners', 'Proper Nouns']
target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])
pos_tags = ['CD', 'DT', 'NNP']
X = get_data(text, pos_tags)
Y = np.ones(len(X), dtype=np.int) # if file is PROSE
Y *= value

# code to write data to classification file        ***DO NOT DELETE***
if save_classification:
	f = open('classification', 'a')
	for i in range(len(X)):
		f.write("%s_%s\t%s\n" % (name, nums[i], Y[i]))
	f.close()

# code to write feature data to training_data file        ***DO NOT DELETE***
if save_data:
	nums = np.array(nums)
	nums = np.reshape(nums, (len(nums), 1))
	final_table = np.hstack((nums, X))
	f = open('training_data', 'ab')
	np.savetxt(f, final_table, fmt=name + "_%s\t" + "1:%s\t2:%s\t3:%s")
	f.close()

# code to display data viz
# XY = np.hstack((X,target_names[Y][:,np.newaxis]))
# df = pd.DataFrame(XY)
# df.columns= feature_names + ["Class"]
# df[feature_names] = df[feature_names].astype(float)
# print df.corr()
# df.hist(bins=np.arange(0,1,0.005),sharex=True);
# scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
# plt.show()