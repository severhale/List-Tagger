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
from compiler.ast import flatten
import re

words = nltk.corpus.cmudict.dict()

# Count up all unique parts of speech and return two arrays
# unique contains all the different POS tags found
# counts contains the counts of each unique POS tag in the order they are found in unique
def pos_count(tokenized_text):
	flat = flatten(tokenized_text)
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

def get_pages(fname, pg_start, pg_end):
	pg_start -= 1
	pg_end -= 1
	tree = load_xml_tree(fname)
	pg_iterator = tree.iter(tag="OBJECT")
	i=0
	while i<pg_start:
		pg_iterator.next()
		i += 1
	pages = []
	page_nums = []
	while i<=pg_end:
		try:
			page = pg_iterator.next()
			pages.append(page)
		except StopIteration:
			break;
		i += 1
	return pages


# Return 2D array of words, first dimension is page,
# second is index of word in page
def tokenize_pages(page_trees):
	pages = []
	page_nums = []
	for page in page_trees:
		page_text = page.findall(".//WORD")
		pages.append([j.text for j in page_text])
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		page_nums.append(pg_num)
		print "Page",pg_num,pages[-1]
	return page_nums,pages

def get_page_numbers(page_trees):
	page_nums = []
	for page in page_trees:
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		page_nums.append(pg_num)
	return page_nums

def get_book_name(page_trees):
	return page_trees[0].find(".//PARAM").get('value')[:-10]

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

def get_paragraph_length(page_trees):
	paragraph_lengths = []
	for page in page_trees:
		paragraphs = page.findall(".//PARAGRAPH")
		pg_plengths = np.asarray([len(j.findall(".//WORD")) for j in paragraphs])
		paragraph_lengths.append(pg_plengths.mean() if len(pg_plengths)>0 else 0)
	return np.resize(np.array(paragraph_lengths), (len(paragraph_lengths), 1))

# find concentration of rhyming lines for each page
def get_rhymes(page_trees, level):
	rhymes = []
	for page in page_trees:
		num = page.find(".//PARAM").get('value')[-9:-5]
		count = 0
		lines = page.findall(".//LINE")
		for i in range(len(lines)-2):
			lastw = last_word(lines[i])
			if lastw == '':
				continue
			lastw2 = last_word(lines[i+1])
			lastw3 = last_word(lines[i+2])
			rhyme1 = check_rhyme(lastw, lastw2, level)
			rhyme2 = check_rhyme(lastw, lastw3, level)
			count += rhyme1 + rhyme2
			# print "Page %s: %s and %s %s rhyme" % (num, lastw, lastw2, "do" if rhyme1 else "don\'t")
			# print "Page %s: %s and %s %s rhyme" % (num, lastw, lastw3, "do" if rhyme2 else "don\'t")
		rhymes.append(count/(2*len(lines)) if len(lines) > 0 else 0)
	return np.resize(np.array(rhymes), (len(rhymes), 1))

def last_word(line):
	text = [i.text for i in line.findall(".//WORD")]
	word = text[-1] if len(text)>0 else ''
	return re.sub(r'\W+', '', word)

def check_rhyme(word1, word2, level):
	if (word1 in words and word2 in words):
		pron1 = words[word1][0]
		pron2 = words[word2][0]
		result = pron1[-level:]==pron2[-level:]
	else:
		result = word1[-2:]==word2[-2:]
	# if result:
	# 	print "%s and %s rhyme!" % (word1.encode('utf-8'), word2.encode('utf-8'))
	return result