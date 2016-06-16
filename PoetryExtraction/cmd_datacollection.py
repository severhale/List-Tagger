import poetryhelper

def classify_line(book_name, pg_num, line_num, state):
	f = open('classification', 'a')
	f.write("%s_%s_%s %d\n" % (book_name, pg_num, line_num, state))
	f.close()

pages = poetryhelper.get_pages("../All Text/irresistiblecurr00loweiala_djvu.xml", 150, 152)
pg_nums = poetryhelper.get_page_numbers(pages)
name = poetryhelper.get_book_name(pages)
for i in range(len(pg_nums)):
	lines = poetryhelper.get_all_lines(pages[i])
	pg_dim = poetryhelper.get_page_dimensions(pages[i])
	num = pg_nums[i]
	for j in range(len(lines)):
		print '\n' + poetryhelper.get_line_text(lines[j])
		inp = raw_input("Poetry? ")
		state = 0
		if inp == 'y':
			state = 1
		classify_line(name, num, j, state)