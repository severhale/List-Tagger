import poetryhelper
import random
import copy
import numpy as np

def classify_line(book_name, pg_num, line_num, state):
	f = open('classification', 'a')
	f.write("%s_%s_%s %d\n" % (book_name, pg_num, line_num, state))
	f.close()

feature_names = ["Left Margin", "Right Margin", "Top Margin", "Bottom Margin", "Syllables", "Paragraph Length", "Number", "Determiner", "Proper Noun"]
pages = poetryhelper.get_pages("../All Text/anthologyofmagaz1917brai_djvu.xml", 400, 401)
pg_nums = poetryhelper.get_page_numbers(pages)
name = poetryhelper.get_book_name(pages)
tags = []
data = []
for i in range(len(pg_nums)):
	lines = poetryhelper.get_all_lines(pages[i])
	pg_dim = poetryhelper.get_page_dimensions(pages[i])
	num = pg_nums[i]
	for j in range(len(lines)):
		data.append(poetryhelper.get_feature_vec(lines, j, pg_dim))
		tags.append("%s_%s_%d" % (name, num, j))
		print '\n' + poetryhelper.get_line_text(lines[j])
		inp = raw_input("Poetry? ")
		state = 0
		if inp == 'y':
			state = 1
		classify_line(name, num, j, state)
poetryhelper.save_data(data, tags, 1)

# for j in range(len(pages)):
# 	page = pages[j]
# 	num = pg_nums[j]
# 	print "Page %s" % (num)
# 	lines = poetryhelper.get_all_lines(page)
# 	for line_num in range(len(lines)):
# 		print '\n' + poetryhelper.get_line_text(lines[line_num])
# 		inp = raw_input("Poetry? ")
# 		state = 0
# 		if inp == 'y':
# 			state = 1
# 		classify_line(name, num, line_num, state)