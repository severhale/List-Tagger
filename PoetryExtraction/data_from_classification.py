import numpy as np
import pickle
import time
import os
from poetryhelper import *

classification = open('classification', 'r')
lines = classification.readlines()
classification.close()
targets = []
freq_dict = pickle.load(open('worddict.p', 'rb'))

lines = [line.split(' ')[0] for line in lines]
books = np.unique(np.asarray(map(parse_tag, lines))[:,0])

start = time.time()
count = 0
for book in books:
	if os.path.isfile("../AllText/" + book + "_data"):
		continue
	#### get pages required
	fname = "../AllText/" + book + "_djvu.xml"
	if not os.path.isfile(fname):
		fname = "../AllText/" + book + ".xml"
	if not os.path.isfile(fname):
		fname = "../../DownloadedText/" + book + "_djvu.xml"
	pages = list(get_pg_iterator(fname))
	tags, data = easy_feature_table(0, pages, freq_dict)

	# lines = arr[:,1:][arr[:,0]==book].astype(int)
	# pg_start = min(lines[:,0])
	# pg_end = max(lines[:,0])
	# pages = get_pages(fname, pg_start, pg_end)
	# nums = get_page_numbers(pages)
	# actual_start = int(nums[0])
	# actual_end = int(nums[-1])
	# #### some books' page numbers don't start at 0001
	# if (actual_start > pg_start):
	# 	diff = actual_start - pg_start
	# 	pages = get_pages(fname, pg_start - diff, pg_start - 1) + pages

	# if (actual_end < pg_end):
	# 	diff = pg_end - actual_end
	# 	pages = pages + get_pages(fname, pg_end + 1, pg_end + diff)
	# nums = get_page_numbers(pages)
	# nums = [int(n) for n in nums]
	# parent_map = get_parent_map(pages)
	# #### now have all pages in an array that definitely includes all pages needed

	# for lineinfo in lines:
	# 	pg_num = lineinfo[0]
	# 	line_num = lineinfo[1]
	# 	pg_index = nums.index(pg_num)

	# 	d = get_feature_vec_pg(3, parent_map, pages, pg_index, line_num, freq_dict, dict_sum)
	# 	data.append(d)
	# 	str_pgnum = str(pg_num).zfill(4)
	# 	tags.append(book + '_' + str_pgnum + '_' + str(lineinfo[1]))

	save_data(data, tags, "../AllText/" + book + "_data", 'w')
	count += 1
	print "Book %d done after %.2f minutes" % (count, (time.time() - start)/60)