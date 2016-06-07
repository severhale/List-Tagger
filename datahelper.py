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
		# print "Page",pg_num
		# print pages[-1]
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
	for i in range(len(tokenized_text_array)):
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