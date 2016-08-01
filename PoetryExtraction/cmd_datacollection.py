## cmd_datacollection.py
## A command line interface for annotating data
## Usage: python cmd_datacollection.py <filename> [<startpage> [<endpage>]]
## startpage and endpage optional, can't have endpage without a startpage.
## If left out, use entire book.

import poetryhelper
import curses
import sys
import webbrowser

statedict = {
	' ':0,
	'w':1,
	'e':2,
	'r':3,
	't':4,
	'a':5
}

# Format classification into a string to be used in the classification file
def classify_line(book_name, pg_num, line_num, state):
	return "%s_%s_%s %d\n" % (book_name, pg_num, line_num, state)

# Append list of strings to classification file
def write_data(data):
	f = open('classification', 'a')
	for line in data:
		f.write(line)
	f.close()

# Wrapper function for annotating to be used within curses window
def gather(stdscr, pages, pg_nums, name):
	i=0
	data = []
	while i < len(pg_nums):
		if i!=0 and i%2==0:
			write_data(data)
			data = []
		lines = poetryhelper.get_all_lines(pages[i])
		# pg_dim = poetryhelper.get_page_dimensions(pages[i])
		num = pg_nums[i]
		j=0
		while j < len(lines):
			stdscr.clear()
			stdscr.move(0, 0)
			stdscr.addstr("Page %s, Line %d\n\r" % (num, j))
			stdscr.addstr(poetryhelper.get_line_text(lines[j]))
			inp = stdscr.getch()
			c = curses.keyname(inp)
			if c in statedict:
				state = statedict[c]
			elif inp==curses.KEY_BACKSPACE:
				if j>0:
					j -= 1
				elif i>0:
					oldi = i
					oldj = j
					foundline = False
					while i>0 and not foundline:
						i -= 1
						newlines = poetryhelper.get_all_lines(pages[i])
						# newpg_dim = poetryhelper.get_page_dimensions(pages[i])
						newnum = pg_nums[i]
						j = len(newlines) - 1
						if len(newlines)>0:
							foundline = True
							# pg_dim = newpg_dim
							lines = newlines
							num = newnum
					if not foundline: # first line in the book at oldi, oldj
						i = oldi
						j = oldj
				if len(data)>0:
					l = data[-1].split(' ')[0].split('_')
					# ONLY remove last piece of data if it's the prev line
					if l[0]==name and l[1]==num and int(l[2])==j:
						del data[-1]
				continue
			elif c=='q':
				i = len(pg_nums)
				break
			else:
				j += 1
				continue
			data.append(classify_line(name, num, j, state))
			j += 1
		i += 1
	write_data(data)


# fname = raw_input("Enter file name: ")
# pg_start = int(raw_input("Enter start page: "))
# pg_end = int(raw_input("Enter end page: "))
fname = sys.argv[1]
pg_start = 1
pg_end = sys.maxint
if len(sys.argv) > 2:
	pg_start = int(sys.argv[2])
if len(sys.argv) > 3:
	pg_end = int(sys.argv[3])
pages = poetryhelper.get_pages(fname, pg_start, pg_end)
pg_nums = poetryhelper.get_page_numbers(pages)
name = poetryhelper.get_book_name(pages)

url = "https://archive.org/stream/%s#page/n%d/mode/1up" % (name, int(pg_nums[0]))
webbrowser.open(url)
print "press q to quit"
print "w:\tbegin poem"
print "e:\tmiddle poem"
print "r:\tend poem"
print "t:\ttitle poem"
print "a:\tauthor"
print "<space>:\tnon-poem"
print "<backspace>:\treclassify last line"
print "<other key>:\tskip current line"
raw_input("Press enter when you're ready to begin.")
curses.wrapper(gather, pages, pg_nums, name)