from __future__ import division
import xml.etree.ElementTree as ET
import numpy as np
import nltk
import copy

words = nltk.corpus.cmudict.dict()

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

def get_page_numbers(pages):
	page_nums = []
	for page in pages:
		pg_num = page.find(".//PARAM").get('value')[-9:-5]
		page_nums.append(pg_num)
	return page_nums

def get_book_name(pages):
	return pages[0].find(".//PARAM").get('value')[:-10]

def get_lines_in_pages(pages):
	return pages.findall(".//LINE")

def get_all_lines(page):
	return page.findall(".//LINE")

def get_line(page, line_num):
	lines = page.findall(".//LINE")
	return lines[line_num]

def get_line_text(line):
	s = ' '.join(i.text for i in line.findall(".//WORD"))
	return s.encode('utf-8')

def get_page_dimensions(page):
	return (int(page.attrib['width']), int(page.attrib['height']))

def get_full_coords(word):
	return [int(i) for i in word.attrib['coords'].split(',')]

def get_word_pos(word):
	coords = get_full_coords(word)
	return (coords[0], coords[1])

def get_word_size(word):
	coords = get_full_coords(word)
	return coords[1] - coords[3]

def syllable_count(line):
	text = [i.text.encode('utf-8') for i in line.findall(".//WORD")]
	return sum((word_syls(i) for i in text))

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

def get_feature_vec_pg(parent_map, page, line_num):
	lines = get_all_lines(page)
	pg_dim = get_page_dimensions(page)
	return get_feature_vec(parent_map, lines, line_num, pg_dim)
	
def pos_count(line):
	words = [i.text for i in line.findall(".//WORD")]
	tagged = nltk.pos_tag(words)
	tags = np.array(tagged)[:,1]
	unique, counts = np.unique(tags, return_counts=True)
	total = np.sum(counts)
	counts = counts.astype(float)
	counts /= total
	result = []
	for c in ['CD', 'DT', 'NNP']:
		if c in unique:
			result.append(counts[unique==c][0])
		else:
			result.append(0)
	return result

def get_parent_map(pages):
	return {c:p for page in pages for p in page.iter() for c in p}

def get_feature_vec(parent_map, lines, line_num, pg_dim):
	vec = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	if len(lines) > line_num and line_num >= 0:
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
			pos = pos_count(line)
			vec = [l_margin/pg_dim[0], r_margin/pg_dim[0], t_margin/pg_dim[1], b_margin/pg_dim[1], syllables, plength]
			# vec = [syllables, plength]
			vec += pos
	return vec

def getelement(arr, index):
	if index < 0:
		return arr[0]
	elif index >= len(arr):
		return arr[-1]
	else:
		return arr[index]

def save_data(data, names, num_neighbors):
	#### add surrounding line data to each element
	datacopy = copy.deepcopy(data)
	for i in range(0, len(datacopy)):
		for j in range(1, num_neighbors+1):
			data[i] += getelement(datacopy, i+j) + getelement(datacopy, i-j)
	data = np.asarray(data)

	names = np.array(names)
	names = np.reshape(names, (len(names), 1))
	print data.shape
	print names.shape
	final_table = np.hstack((names, data))
	fmt_string = "%s\t"
	for i in range(1, data.shape[1] + 1):
		fmt_string += str(i) + ":%s\t"
	f = open('training_data', 'ab')
	np.savetxt(f, final_table, fmt=fmt_string)
	f.close()