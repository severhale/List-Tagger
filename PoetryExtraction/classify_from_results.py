from poetryhelper import *
import numpy as np
import urllib
import random

fname = 'toclassify'

with open(fname, 'r') as f:
	lines = np.asarray(f.readlines())

linenum = random.randrange(len(lines))
book, page = lines[linenum].split()

#### Time to download the book
# for line in lines:
# 	book, page = line.split()
# 	url = 'http://archive.org/download/%s/%s_djvu.xml' % (book, book)
# 	urllib.urlretrieve(url, "../../DownloadedText/%s_djvu.xml" % (book))
pg_start = int(page)-5
pg_end = int(page)+5
print "../../DownloadedText/%s_djvu.xml" % (book), pg_start, pg_end

lines = np.delete(lines, linenum)
with open(fname, 'w') as f:
	for line in lines:
		f.write(line)