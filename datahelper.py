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
	flat = [item for sublist in tokenized_text for item in sublist]
	tagged = nltk.pos_tag(flat)
	tags = np.array([i[1] for i in tagged])
	unique, counts = np.unique(tags, return_counts=True)
	return unique, counts

def load_text(fname):
	f = open(fname, 'r')
	text = f.read()
	f.close()
	return text

def load_xml_tree(fname):
	return ET.fromstring(load_text(fname))

# Parse pages from pg_start to pg_end inclusive
# pg_start and pg_end are 1-indexed
def tokenized_from_xml_range(tree, pg_start, pg_end):
	pg_start -= 1
	pg_end -= 1
	name = get_book_name(tree)
	pg_iterator = tree.iter(tag="OBJECT")
	i=0
	while i<pg_start:
		pg_iterator.next()
		i += 1
	pages = []
	page_nums = []
	while i<=pg_end:
		page = pg_iterator.next()
		pages.append([])
		line_iterator = page.iter(tag="LINE")
		eop = False
		while not eop:
			try:
				line = line_iterator.next()
				line_text = [j.text for j in line.findall(".//WORD")]
				pages[-1].append(line_text)
			except StopIteration:
				eop = True


		# page_text = page.findall(".//WORD")
		# pages.append([j.text for j in page_text])
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		print "Page",pg_num
		print pages[-1]
		page_nums.append(pg_num)
		i += 1
	return name,page_nums,pages

def get_book_name(tree):
	return tree.find(".//OBJECT").find(".//PARAM").get('value')[:-10]

def tokenized_from_rawtext(fname):
	f = open(fname, 'r')
	text = f.read()
	text = text.decode('utf-8')
	return nltk.word_tokenize(text)
	f.close()

# get data all parts of speech in pos_tags, as well as 
# the frequency of lines that start with a capital letter
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

def get_paragraph_length(tree, pg_start, pg_end):
	pg_start -= 1
	pg_end -= 1
	pg_iterator = tree.iter(tag="OBJECT")
	i = 0
	paragraph_lengths = []
	while(i<pg_start):
		pg_iterator.next()
		i += 1
	while(i<=pg_end):
		page = pg_iterator.next()
		paragraphs = page.findall(".//PARAGRAPH")
		pg_plengths = np.asarray([len(j.findall(".//WORD")) for j in paragraphs])
		paragraph_lengths.append(pg_plengths.mean() if len(pg_plengths)>0 else 0)
		i += 1
	return np.resize(np.array(paragraph_lengths), (len(paragraph_lengths), 1))