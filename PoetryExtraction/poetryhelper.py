from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
import nltk
import copy
import string
import math
import random
import operator
import sklearn.metrics

classes = ['0', '1', '2', '3']

crfd = {
	'syl_c':0, # int
	'syl_pc':1, # bool
	'syl_cn':2, # bool
	'prob_c':3, # int
	'lmarg_c':4, # float.1
	'lmarg_pc':5, # bool
	'lmarg_cn':6, # bool
	'rmarg_c':7, # float.1
	'rmarg_pc':8, # bool
	'rmarg_cn':9, # bool
	'line1':10, # [0,1]
	'line2':11, # [0,1]
	'capital_c':12, # bool
	'rf_c':13, # [0,1]
	'rf_pc':14, # [0,1]-[0,1]
	'rf_cn':15, # [0,1]-[0,1]
	'lmarg_pn':16,
	'rmarg_pn':17,
	'syl_pn':18,
	'tmarg_c':19,
	'tmarg_p':20,
	'bmarg_c':21,
	'bmarg_n':22,
	'tbmarg_c':23,
	'tmarg-line_c':24,
	'capital_cn':25,
	'capital_pc':26,
	'capital_pn':27,
	'repetition_pc':28,
	'repetition_cn':29,
	'repetition_pn':30,
	'capital_cn-lmarg_cn':31, # bool-bool-[-1,0,1]
	'lines_remaining':32, # int
	'punc_end':33, # bool
}

treed = {
	'syl_c':0, # int
	'syl_p':1, # int
	'syl_n':2, # int
	'prob_c':3, # float
	'lmarg_c':4, # float
	'lmarg_p':5, # float
	'lmarg_n':6, # float
	'rmarg_c':7, # float
	'rmarg_p':8, # float
	'rmarg_n':9, # float
	'tmarg_c':10, # float
	'tmarg_p':11, # float
	'tmarg_n':12, # float
	'bmarg_c':13, # float
	'bmarg_p':14, # float
	'bmarg_n':15, # float
	'linenum':16, # int
	'adj':17, # float
	'pnoun':18, # float
	'det':19, # float
	'nums':20, # float
	'plength':21,
	'cap_lines':22,
}

invcrfd = {v:k for k,v in crfd.items()}

words = nltk.corpus.cmudict.dict()

# Load all text from file as an array of strings
def load_text(fname):
	f = open(fname, 'r')
	text = f.read()
	f.close()
	return text

# Get the full XML tree from DjVu file
def load_xml_tree(fname):
	return ET.fromstring(load_text(fname))

# Get an iterator over all page XML objects in a DjVu file
def get_pg_iterator(fname):
	tree = load_xml_tree(fname)
	return tree.iter(tag="OBJECT")

# Extract the pages in the specified range as XML objects from a DjVu file
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

# Get list of all image numbers of pages
def get_page_numbers(pages):
	page_nums = []
	for page in pages:
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		page_nums.append(pg_num)
	return page_nums

# Get IMAGE number of page specififed. Almost never actually the page number on the page
def get_page_number(page):
	return page.find(".//PARAM").get('value')[-9:-5]

# Get the book name
def get_book_name(pages):
	return pages[0].find(".//PARAM").get('value')[:-10]

# Get all line XML objects in all pages
def get_lines_in_pages(pages):
	return pages.findall(".//LINE")

# Get all XMl objects of lines on the page
def get_all_lines(page):
	return page.findall(".//LINE")

# Get line XMl object of specified line
def get_line(page, line_num):
	lines = page.findall(".//LINE")
	return lines[line_num]

# Get line text as a single string
def get_line_text(line):
	s = ' '.join(i.text for i in line.findall(".//WORD"))
	return s.encode('utf-8')

# Get (width, height) tuple of page size in pixels
def get_page_dimensions(page):
	return (int(page.attrib['width']), int(page.attrib['height']))

# Get bounding box of word
def get_full_coords(word):
	return [int(i) for i in word.attrib['coords'].split(',')]

# Return coordinates of top left(?confirm this?) corner of bounding box of word
def get_word_pos(word):
	coords = get_full_coords(word)
	return (coords[0], coords[1])

# Get height of bounding box of word
def get_word_size(word):
	coords = get_full_coords(word)
	return coords[1] - coords[3]

# Count all syllables in line
def syllable_count(line):
	text = [i.text.encode('utf-8') for i in line.findall(".//WORD")]
	return sum((word_syls(i) for i in text))

# Count the syllables in word. If not in the CMU corpus, estimates by number of vowels surrounded by consonants
def word_syls(word):
	if word in words:
		pronounciation = words[word]
		return len(list(y for y in pronounciation[0] if y[-1].isdigit()))
	else:
		count = 0
		vowels = "aeiouy"
		vow = False
		for i in range(len(word)):
			c = word[i]
			if i != len(word) - 1:
				if c in vowels:
					if not vow:
						count += 1
						vow = True
				else:
					vow = False
		return count

# Get counts of various parts of speech in a line
def pos_count(line):
	words = ' '.join([i.text for i in line.findall(".//WORD")])
	tok = nltk.word_tokenize(words)
	tagged = nltk.pos_tag(tok)
	tags = np.array(tagged)[:,1]
	unique, counts = np.unique(tags, return_counts=True)
	total = np.sum(counts)
	counts = counts.astype(float)
	counts /= total
	result = []
	for c in ['JJ', 'NNP', 'DT', 'CD']:
		if c in unique:
			result.append(counts[unique==c][0])
		else:
			result.append(0)
	return result

# Get a dictionary which maps from XML elements to their parents in pages
def get_parent_map(pages):
	return {c:p for page in pages for p in page.iter() for c in p}

# Get numpy table with all data for all pages
def easy_feature_table(pages, freq_dict):
	parent_map = get_parent_map(pages)
	dict_sum = sum(freq_dict.itervalues())
	tags, data = get_feature_table(parent_map, pages, freq_dict, dict_sum)
	return tags, data

# Get a numpy table containing all single feature vectors of all lines on all pages
def get_feature_table(parent_map, pages, freq_dict, dict_sum):
	data = []
	tags = []
	for i in range(len(pages)):
	 	ltags, ldata = get_feature_table_pg(parent_map, pages[i], freq_dict, dict_sum)
		if len(ldata) > 0:
			data += ldata
			tags += ltags
	return np.asarray(tags), np.vstack(data)

# Get a numpy array containing all single feature vectors of all lines on the page specified
def get_feature_table_pg(parent_map, page, freq_dict, dict_sum):
	lines = get_all_lines(page)
	pg_dim = get_page_dimensions(page)
	name = get_book_name([page])
	num = get_page_number(page)
	# if len(lines) == 0:
	# 	data = np.array([get_feature_vec(parent_map, [], -1, pg_dim, freq_dict, dict_sum)])
	# 	tags = []
	# else:
	tags = []
	data = []
	for i in range(len(lines)):
		tags.append("%s_%s_%d" % (name, num, i))
		data.append(get_single_feature_vec(parent_map, lines, i, pg_dim, freq_dict, dict_sum))
	result = data
	# result = [lsum([getelement(data, j) for j in range(i-2,i+3)]) for i in range(len(data))]
	return tags, result

# Get the single feature vector of the line specified by page_index and line_num
def get_feature_vec_pg(parent_map, pages, page_index, line_num, freq_dict, dict_sum):
	result = []
	page = pages[page_index]
	lines = get_all_lines(pages[page_index])
	pg_dim = get_page_dimensions(page)
	# result = get_single_feature_vec(parent_map, lines, line_num, pg_dim, freq_dict, dict_sum)
	prev_lines=[]
	next_lines=[]
	if line_num < 1:
		if page_index > 0:
			prev_lines = get_all_lines(pages[page_index-1])
	if line_num > len(lines)-2:
		if page_index < len(pages)-1:
			next_lines = get_all_lines(pages[page_index-1])
	result = get_feature_vec(parent_map, prev_lines+lines+next_lines, line_num+len(prev_lines), pg_dim, freq_dict, dict_sum)
	return result

# Get feature vector of just the line specified with no neighbor data as a list of values
def get_single_feature_vec(parent_map, lines, line_num, pg_dim, freq_dict, dict_sum):
	vec = [0] * len(treed)
	if line_num < len(lines) and line_num >= 0:
		line = lines[line_num]
		if len(line.findall(".//WORD")) > 0:
			first_pos = get_word_pos(line[0])
			last_pos = get_word_pos(line[-1])
			l_margin = first_pos[0]
			r_margin = pg_dim[0] - last_pos[0]
			t_margin = 0
			b_margin = 0
			found_prev = True
			if line_num > 0:
				prev_line = lines[line_num - 1]
				if len(prev_line) > 0:
					prev_pos = get_word_pos(prev_line[0])
					if prev_pos[1] < first_pos[1]:
						t_margin = first_pos[1] - prev_pos[1]
					else:
						found_prev = False
				else:
					found_prev = False
			if not found_prev:
				t_margin = first_pos[1]
			found_next = True
			if line_num < len(lines) - 1:
				next_line = lines[line_num + 1]
				if len(next_line) > 0:
					next_pos = get_word_pos(next_line[0])
					if next_pos[1] > first_pos[1]:
						b_margin = next_pos[1] - first_pos[1]
					else:
						found_next = False
				else:
					found_next = False
			else:
				found_next = False
			if not found_next:
				b_margin = pg_dim[1] - first_pos[1]
			syllables = syllable_count(line)
			para = parent_map[line]
			plength = len(para.findall(".//WORD"))
			margins = [l_margin/pg_dim[0], r_margin/pg_dim[0], t_margin/pg_dim[1], b_margin/pg_dim[1], syllables, plength]
			pos = pos_count(line)
			prob = get_probability(line, freq_dict, dict_sum)
			vec[treed['syl_c']] = syllables
			vec[treed['prob_c']] = prob
			vec[treed['lmarg_c']] = margins[0]
			vec[treed['rmarg_c']] = margins[1]
			vec[treed['tmarg_c']] = margins[2]
			vec[treed['bmarg_c']] = margins[3]
			vec[treed['linenum']] = line_num
			vec[treed['adj']] = pos[0]
			vec[treed['pnoun']] = pos[1]
			vec[treed['det']] = pos[2]
			vec[treed['nums']] = pos[3]
			vec[treed['plength']] = plength
			first_words = []
			paralines = para.findall(".//LINE")
			for line in paralines:
				if len(line) > 0:
					first_words.append(line[0].text)
			vec[treed['cap_lines']] = sum(i[0].istitle() for i in first_words)/len(paralines)
	return vec

def get_crf_data(tree_clf, X1, names, pages):
	X = []
	for i in range(len(X1)):
		X.append(X1[i][:]) # deep copy so as not to change X1
	# need to go through and get each line referenced
	for i in range(len(X)):
		tree_guess = tree_clf.predict(X[i])
		for j in range(len(X[i])):
			line = X[i][j]
			linedata = parse_tag(names[i][j])
			linenum = int(linedata[2])
			vec = [0] * len(crfd)
			vec[crfd['syl_c']] = "%d" % line[treed['syl_c']]
			vec[crfd['syl_pc']] = "%s" % (abs(line[treed['syl_p']]-line[treed['syl_c']]) <= 1)
			vec[crfd['syl_cn']] = "%s" % (abs(line[treed['syl_c']]-line[treed['syl_n']]) <= 1)
			vec[crfd['syl_pn']] = "%s" % (abs(line[treed['syl_p']]-line[treed['syl_n']]) <= 1)
			# vec[crfd['syl_pc']] = "%d-%d" % (line[treed['syl_p']], line[treed['syl_c']])
			# vec[crfd['syl_cn']] = "%d-%d" % (line[treed['syl_c']], line[treed['syl_n']])
			vec[crfd['prob_c']] = "%d" % (line[treed['prob_c']]/10)
			vec[crfd['lmarg_c']] = "%.1f" % line[treed['lmarg_c']]
			# vec[crfd['lmarg_pc']] = "%.1f-%.1f" % (line[treed['lmarg_p']], line[treed['lmarg_c']])
			# vec[crfd['lmarg_cn']] = "%.1f-%.1f" % (line[treed['lmarg_c']], line[treed['lmarg_n']])
			vec[crfd['lmarg_pc']] = "%s" % (abs(line[treed['lmarg_p']]-line[treed['lmarg_c']])<=.05)
			vec[crfd['lmarg_cn']] = "%s" % (abs(line[treed['lmarg_c']]-line[treed['lmarg_n']])<=.05)
			vec[crfd['lmarg_pn']] = "%s" % (abs(line[treed['lmarg_p']]-line[treed['lmarg_n']])<=.05)
			
			vec[crfd['rmarg_c']] = "%.1f" % line[treed['rmarg_c']]
			vec[crfd['rmarg_pc']] = "%s" % (abs(line[treed['rmarg_p']]-line[treed['rmarg_c']])<=.05)
			vec[crfd['rmarg_cn']] = "%s" % (abs(line[treed['rmarg_c']]-line[treed['rmarg_n']])<=.05)
			vec[crfd['rmarg_pn']] = "%s" % (abs(line[treed['rmarg_p']]-line[treed['rmarg_n']])<=.05)
			# vec[crfd['rmarg_pc']] = "%.1f-%.1f" % (line[treed['rmarg_p']], line[treed['rmarg_c']])
			# vec[crfd['rmarg_cn']] = "%.1f-%.1f" % (line[treed['rmarg_c']], line[treed['rmarg_n']])
			vec[crfd['line1']] = "1" if linenum==0 else "0"
			vec[crfd['line2']] = "1" if linenum==1 else "0"
			linetext = get_line(pages[i][j], linenum)
			curr_word = linetext[0].text if len(linetext)>0 else ''
			if curr_word.isdigit() and len(linetext)>1:
				curr_word = linetext[1].text
			curr_cap = curr_word.istitle()
			vec[crfd['capital_c']] = "%s" % curr_cap
			vec[crfd['rf_c']] = "%s" % tree_guess[j]
			if j==0:
				rf_p = "null"
			else:
				rf_p = tree_guess[j-1]
			if j==len(X[i])-1:
				rf_n = "null"
			else:
				rf_n = tree_guess[j+1]
			vec[crfd['rf_pc']] = "%s-%s" % (rf_p, tree_guess[j])
			vec[crfd['rf_cn']] = "%s-%s" % (tree_guess[j], rf_n)
			vec[crfd['tmarg_c']] = "%.1f" % (line[treed['tmarg_c']]*3)
			vec[crfd['tmarg_p']] = "%.1f" % (line[treed['tmarg_p']]*3)
			vec[crfd['bmarg_c']] = "%.1f" % (line[treed['bmarg_c']]*3)
			vec[crfd['bmarg_n']] = "%.1f" % (line[treed['bmarg_n']]*3)
			vec[crfd['tbmarg_c']] = "%s" % (abs(line[treed['tmarg_c']] - line[treed['bmarg_c']])<=.02)
			vec[crfd['tmarg-line_c']] = "%s-%s" % (vec[crfd['tmarg_c']], (linenum==0 or linenum==1))
			prev_cap = False
			prev_word = ''
			if j>0:
				prev_line = get_line(pages[i][j-1], int(parse_tag(names[i][j-1])[2]))
				prev_word = prev_line[0].text if len(prev_line)>0 else ''
				if prev_word.isdigit() and len(prev_line)>1:
					prev_word = prev_line[1].text
				prev_cap = prev_word.istitle()
			next_cap = False
			next_word = ''
			if j<len(X[i])-1:
				next_line = get_line(pages[i][j+1], int(parse_tag(names[i][j+1])[2]))
				next_word = next_line[0].text if len(next_line)>0 else ''
				if next_word.isdigit() and len(next_line)>1:
					next_word = next_line[1].text
				next_cap = next_word.istitle()
			vec[crfd['capital_cn']] = "%s-%s" % (vec[crfd['capital_c']], next_cap)
			vec[crfd['capital_pc']] = "%s-%s" % (prev_cap, vec[crfd['capital_c']])
			vec[crfd['capital_pn']] = "%s-%s" % (prev_cap, next_cap)
			vec[crfd['repetition_pc']] = "%s" % (prev_word.lower()==curr_word.lower())
			vec[crfd['repetition_cn']] = "%s" % (curr_word.lower()==next_word.lower())
			vec[crfd['repetition_pn']] = "%s" % (prev_word.lower()==next_word.lower())
			vec[crfd['lines_remaining']] = "%d" % (len(get_all_lines(pages[i][j])) - linenum)
			vec[crfd['punc_end']] = "%s" % (linetext[-1].text[-1] in string.punctuation if len(linetext)>0 and len(linetext[-1].text)>0 else False)
			lmarg_sign = 0
			if vec[crfd['lmarg_cn']]=="False":
				lmarg_sign = 1 if line[treed['lmarg_n']] > line[treed['lmarg_c']] else -1
			vec[crfd['capital_cn-lmarg_cn']] = "%s-%s-%d" % (curr_cap, next_cap, lmarg_sign)
			
			X[i][j] = make_dict(vec)
	return X

# Get the feature vector of the specified line as a list of values along with four surrounding lines
def get_feature_vec(parent_map, lines, line_num, pg_dim, freq_dict, dict_sum):
	vec = get_single_feature_vec(parent_map, lines, line_num, pg_dim, freq_dict, dict_sum)
	pvec = vec
	nvec = vec
	if line_num > 0:
		pvec = get_single_feature_vec(parent_map, lines, line_num-1, pg_dim, freq_dict, dict_sum)
	if line_num < len(lines):
		nvec = get_single_feature_vec(parent_map, lines, line_num+1, pg_dim, freq_dict, dict_sum)
	vec[treed['syl_p']] = pvec[treed['syl_c']]
	vec[treed['syl_n']] = nvec[treed['syl_c']]
	vec[treed['lmarg_p']] = pvec[treed['lmarg_c']]
	vec[treed['lmarg_n']] = nvec[treed['lmarg_c']]
	vec[treed['rmarg_p']] = pvec[treed['rmarg_c']]
	vec[treed['rmarg_n']] = nvec[treed['rmarg_c']]
	vec[treed['bmarg_p']] = pvec[treed['bmarg_c']]
	vec[treed['bmarg_n']] = nvec[treed['bmarg_c']]
	vec[treed['tmarg_p']] = pvec[treed['tmarg_c']]
	vec[treed['tmarg_n']] = nvec[treed['tmarg_c']]
	return vec

# Find the log-product probability that a line fits the language model contained in freq_dict. dict_sum is the sum of all frequencies of words in the model
def get_probability(line, freq_dict, dict_sum):
	text = get_line_text(line)
	eps = .01
	total = 1
	for w in line:
		word = clean_word(w.text)
		p = eps
		if word in freq_dict:
			p += (1 - eps) * freq_dict[word] / dict_sum
		p = math.log(p)
		total += p
	return total

# Get element in arr at index. If index < 0, get arr[0] and if index > len(arr)-1, get arr[-1]
def getelement(arr, index):
	if index < 0:
		return arr[0]
	elif index >= len(arr):
		return arr[-1]
	else:
		return arr[index]


### NOTE: ONLY WORKS ON CONTIGUOUS DATA!!!!!!!!!!
# Also probably useless with a crf
def concat_neighbors(data, num_neighbors):
	print data.shape
	data = data.tolist()
	#### add surrounding line data to each element
	datacopy = copy.deepcopy(data)
	for i in range(0, len(datacopy)):
		for j in range(i-2,i+3):
			data[i] += getelement(datacopy, j) + getelement(datacopy, j)
	data = np.asarray(data)
	print data.shape
	return data

# Save data to the training_data file
def save_data(data, names):
	names = np.array(names)
	names = np.reshape(names, (len(names), 1))
	print data.shape
	final_table = np.hstack((names, data))
	fmt_string = "%s\t"
	for i in range(1, data.shape[1] + 1):
		fmt_string += str(i) + ":%s\t"
	f = open('training_data', 'wb')
	np.savetxt(f, final_table, fmt=fmt_string)
	f.close()

# Trim whitespace and punctuation
def clean_word(word):
	return word.strip(string.punctuation + string.whitespace).lower()

# Take a tag in book_page_line format and return the tuple (book, page, line)
def parse_tag(tag):
	result = tag.split('_')
	return '_'.join(result[:-2]), result[-2], result[-1]

# Concatenate all elements in a list of lists together
def lsum(list):
	result = []
	for l in list:
		result += l
	return result

# Return True if the two lines specified are adjacent, False o.w.
# prev_page is page pg_num1
def check_adjacent(pg_num1, line1, pg_num2, line2, prev_page):
	if pg_num1 == pg_num2:
		return abs(line1-line2)==1
	else:
		if pg_num1 > pg_num2:
			# swap two lines
			tmp = pg_num1
			pg_num1 = pg_num2
			pg_num2 = tmp
			tmp = line1
			line1 = line2
			line2 = tmp
		# now we know that pg_num1 < pg_num2
		if line2 == 0 and pg_num1 == pg_num2 - 1:
			return line1 == len(get_all_lines(prev_page)) - 1
		else:
			return False

def pages_from_names(names):
	result = []
	lines = np.asarray(map(parse_tag, names))
	lines[:,2] = [i.zfill(4) for i in lines[:,2]]
	inds = np.lexsort(lines.T[::-1])
	revinds = [np.where(inds==i)[0][0] for i in range(len(inds))]
	lines = lines[inds]
	print len(lines),"lines"
	index = 0
	for book in np.unique(lines[:,0]):
		pages = list(get_pg_iterator("../All Text/" + book + "_djvu.xml"))
		pg_nums = get_page_numbers(pages)
		last_pg_num = ''
		while index < len(lines) and lines[index,0]==book:
			num = pg_nums.index(lines[index,1])
			result.append(pages[num])
			last_pg_num = lines[index,1]
			index += 1
	print len(result),"pages"
	return [result[i] for i in revinds]

# Make X into a list of lists of dicts containing feature data
# Y is a list of lists of truth data
# Names is a list of lists of names
# pages is a list of all pages
# BUG(?): Can't convert pages to a numpy array because it turns it into a table
# where each row contains a page object and column contains the child objects/parameters
# of the page object
# .ugh.
def crfformat(X, Y, names, pages):
	lines = np.asarray(map(parse_tag, names))
	lines[:,2] = [i.zfill(4) for i in lines[:,2]]
	nameinds = np.lexsort(lines.T[::-1])
	lines = lines[nameinds]
	names = np.asarray(names)[nameinds]
	X = X[nameinds]
	Y = Y[nameinds]
	pages = [pages[i] for i in nameinds]

	### X and Y are now sorted in lexical order according to names
	index = 0
	X1 = []
	Y1 = []
	names1 = []
	pages1 = []
	for book in np.unique(lines[:,0]):
		contig_X = []
		contig_Y = []
		contig_names = []
		contig_pages = []

		prev_line = int(lines[index,2])
		contig_X.append(X[index])
		contig_Y.append(str(Y[index]))
		contig_names.append(names[index])
		contig_pages.append(pages[index])
		index += 1
		while index < len(lines) and lines[index,0]==book:
			if len(contig_X) > 0 and not check_adjacent(int(lines[index-1,1]), int(lines[index-1,2]), int(lines[index, 1]), int(lines[index, 2]), pages[index-1]):
				X1.append(contig_X)
				Y1.append(contig_Y)
				names1.append(contig_names)
				pages1.append(contig_pages)
				contig_X = []
				contig_Y = []
				contig_names = []
				contig_pages = []
			contig_X.append(X[index])
			contig_Y.append(str(Y[index]))
			contig_names.append(names[index])
			contig_pages.append(pages[index])
			index += 1
		if len(contig_X) > 0:
			X1.append(contig_X)
			Y1.append(contig_Y)
			names1.append(contig_names)
			pages1.append(contig_pages)
	return X1, Y1, names1, pages1

# Takes a dict and returns its values in the order specified in crfd
def flatten(d):
	order = sorted(crfd.items(), key=operator.itemgetter(1))
	return [d[i[0]] for i in order]

# Flattens data in crf format to sklearn-compatible arrays
def uncrfformat(cX, cY, cnames):
	X = lsum(cX)
	Y = lsum(cY)
	names = lsum(cnames)
	return np.asarray(X), np.asarray(Y), names

# Turn a list of features into a dict from feature name to value
def make_dict(feature_vec):
	return {invcrfd[i]:feature_vec[i] for i in range(len(feature_vec))}

def splitcrf(X, Y, names, pages, test_percentage, state=42):
	random.seed(state)
	num_lines = sum(len(x) for x in X)
	line_count = 0
	Xlearn = X[:]
	Ylearn = Y[:]
	Nlearn = names[:]
	Plearn = pages[:]
	Xtest = []
	Ytest = []
	Ntest = []
	Ptest = []
	while line_count < test_percentage * num_lines:
		ind = random.randint(0, len(Xlearn) - 1)
		Xtest.append(Xlearn.pop(ind))
		Ytest.append(Ylearn.pop(ind))
		Ntest.append(Nlearn.pop(ind))
		Ptest.append(Plearn.pop(ind))
		line_count += len(Xtest[-1])
	return Xlearn, Xtest, Ylearn, Ytest, Nlearn, Ntest, Plearn, Ptest

# return all indices where the true value was truth but prediction was predicted
def where_predicted(flat_ytest, flat_yhat, truth, prediction):
	return np.where((np.asarray(flat_ytest)==truth) & (np.asarray(flat_yhat)==prediction))[0]

def print_lines(names, pages):
	lines = [int(i.split('_')[-1]) for i in names]
	for i in range(len(lines)):
		print names[i]+": "+get_line_text(get_line(pages[i], lines[i]))

def print_mistakes(Ytest, Yhat, names, pages, truth, prediction):
	yt = lsum(Ytest)
	yh = lsum(Yhat)
	nms = lsum(names)
	pgs = lsum(pages)
	indices = where_predicted(yt, yh, truth, prediction)
	for i in indices:
		linenum = int(nms[i].split('_')[-1])
		text = get_line_text(get_line(pgs[i], linenum))
		print nms[i] + ": " + text

def print_performance(Ytest, Yhat):
	yt = lsum(Ytest)
	yh = lsum(Yhat)
	print sklearn.metrics.f1_score(yt, yh, average=None)
	print sklearn.metrics.confusion_matrix(yt, yh)

# return a list of tuples of start and end lines of poems
# def poem_indices(Y, names):
# 	names = map(parse_tag, names)
# 	for y,n in zip(Y, names):
# 		