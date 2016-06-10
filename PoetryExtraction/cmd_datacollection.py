import poetryhelper
import random
import copy
import numpy as np

def classify_line(book_name, pg_num, line_num, state):
	f = open('classification', 'a')
	f.write("%s_%s_%s %d\n" % (book_name, pg_num, line_num, state))
	f.close()
def write_data(X, names):
	names = np.array(names)
	names = np.reshape(names, (len(names), 1))
	print X.shape
	print names.shape
	final_table = np.hstack((names, X))
	fmt_string = "%s\t"
	for i in range(1, X.shape[1] + 1):
		fmt_string += str(i) + ":%s\t"
	f = open('training_data', 'ab')
	np.savetxt(f, final_table, fmt=fmt_string)
	f.close()

feature_names = ["Left Margin", "Right Margin", "Top Margin", "Bottom Margin", "Syllables"]
pages = poetryhelper.get_pages("../Mixed/bookman37unkngoog_djvu.xml", 50, 50)
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
datacopy = copy.deepcopy(data)
data[0] += datacopy[0] + datacopy[1]
data[-1] += datacopy[-2] + datacopy[-1]
for i in range(1, len(datacopy) - 1):
	data[i] += datacopy[i-1] + datacopy[i+1]
data = np.asarray(data)
write_data(data, tags)

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