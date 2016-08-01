## datacollection.py
## A failed attempt at visualizing text layout for easier classifiction

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageTk
from Tkinter import *
import numpy as np
import poetryhelper

def prompt_line(page, line_num, scale, panel):
	font_path = "Crimson-Roman.ttf"
	dim = poetryhelper.get_page_dimensions(page)
	dim = tuple(int(i/scale) for i in dim)
	img = PIL.Image.new('RGB', dim, (255, 255, 255))
	d = PIL.ImageDraw.Draw(img)
	words = page.findall(".//WORD")
	font_size = int(np.median([poetryhelper.get_word_size(i) for i in words])/scale)
	for word in page.findall(".//WORD"):
		box = tuple(int(i/scale) for i in poetryhelper.get_full_coords(word)[:4])
		pos = (box[0], box[1] - 1.2 * font_size)
		font = PIL.ImageFont.truetype(font_path, font_size)
		d.text(pos, word.text, fill=(0, 0, 0), font=font)
		# d.rectangle(box, outline=(0,255,0))
	line = page.findall(".//LINE")[line_num]
	word = line.find(".//WORD") # first word on line
	box = tuple(int(i/scale) for i in poetryhelper.get_full_coords(word)[:4])
	indicator = (box[0] - 100/scale, box[1], box[0], box[1] - 1.2 * font_size)
	d.rectangle(indicator, fill=(0,255,0))
	panel.configure(image=PIL.ImageTk.PhotoImage(img))
	

def classify_line(book_name, pg_num, line_num, state):
	f = open('classification', 'wa')
	f.write("%s_%s_%s %d" % (book_name, pg_num, line_num, state))
	f.close()

pages = poetryhelper.get_pages("../Poetry/anthologyofmagaz1917brai_djvu.xml", 50, 60)
pg_nums = poetryhelper.get_page_numbers(pages)
name = poetryhelper.get_book_name(pages)

root = Tk()
root.title('Classification')
dim = poetryhelper.get_page_dimensions(pages[1])
dim = tuple(int(i/3) for i in dim)
root.geometry("%dx%d+%d+%d" % (dim[0], dim[1], 0, 0))
panel = Label(root)
panel.pack(side=TOP, fill=BOTH, expand=YES)
root.mainloop()

for i in range(len(pages)):
	page = pages[i]
	num = pg_nums[i]
	line_count = len(page.findall(".//LINE"))
	for j in range(line_count):
		prompt_line(page, j, 3, panel)
		inp = raw_input("Poetry? ")
		state = 0
		if inp=='y':
			state = 1
		classify_line(name, num, j, state)
