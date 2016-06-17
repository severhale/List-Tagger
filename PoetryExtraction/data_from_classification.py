import numpy as np
import pickle
from poetryhelper import *

classification = open('classification', 'r')
lines = classification.readlines()
classification.close()
targets = [i.split(' ')[0].split('_') for i in lines]

freq_dict = pickle.load(open('worddict.p', 'rb'))
dict_sum = sum(freq_dict.itervalues())

arr = np.asarray(targets)
books = np.unique(arr[:,0])
data = []
tags = []
for book in books:
	#### get pages required
	fname = "../All Text/" + book + "_djvu.xml"
	lines = arr[:,1:][arr[:,0]==book].astype(int)
	pg_start = min(lines[:,0])
	pg_end = max(lines[:,0])
	pages = get_pages(fname, pg_start, pg_end)
	nums = get_page_numbers(pages)
	actual_start = int(nums[0])
	actual_end = int(nums[-1])
	#### some books' page numbers don't start at 0001
	if (actual_start > pg_start):
		diff = actual_start - pg_start
		pages = get_pages(fname, pg_start - diff, pg_start - 1) + pages

	if (actual_end < pg_end):
		diff = pg_end - actual_end
		pages = pages + get_pages(fname, pg_end + 1, pg_end + diff)
	nums = get_page_numbers(pages)
	nums = [int(n) for n in nums]
	parent_map = get_parent_map(pages)
	#### now have all pages in an array that definitely includes all pages needed

	for lineinfo in lines:
		pg_num = lineinfo[0]
		line_num = lineinfo[1]
		pg_index = nums.index(pg_num)

		d = get_feature_vec_pg(parent_map, pages, pg_index, line_num, freq_dict, dict_sum)
		data.append(d)
		str_pgnum = str(pg_num).zfill(4)
		tags.append(book + '_' + str_pgnum + '_' + str(lineinfo[1]))
data = np.asarray(data)
save_data(data, tags)