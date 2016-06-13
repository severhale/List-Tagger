import numpy as np
from poetryhelper import *

classification = open('classification', 'r')
lines = classification.readlines()
classification.close()
targets = [i.split(' ')[0].split('_') for i in lines]

arr = np.asarray(targets)
books = np.unique(arr[:,0])
data = []
tags = []
for book in books:
	#### get pages required
	fname = "../All Text/" + book + "_djvu.xml"
	lines = arr[arr[:,0]==book].astype(int)
	pg_start = min(lines[:,1])
	pg_end = max(lines[:,1])
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
	#### now have all pages in an array that definitely includes all pages needed

	for lineinfo in lines:
		pg_num = lineinfo[0]
		line_num = lineinfo[1]
		pg = pages[nums.index(pg_num)]
		data.append(get_feature_vec_pg(pg, line_num))
		tags.append('_'.join(lineinfo))

save_data(data, tags)