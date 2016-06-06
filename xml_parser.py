import xml.etree.ElementTree as ET
import string

#this one commented out b/c is really big book; takes long time to run
#tree=ET.parse('betterfruit09wash_djvu.xml') #this line parses the xml file and makes it into a big tree
tree=ET.parse('heartsofthree.xml') #line to parse file
print type(tree)

#do I need this line?
root=tree.getroot() #gets root of tree and names it 'root'

pages=tree.findall(".//OBJECT") #this makes a list with all of the pages in the book (128 pages,0-127)
print type(pages)
print len(pages)

#COUNT NUMBER OF WORDS ON PAGE; just loop up to range(len(pages))
for i in range(10): #use for loop so don't have to do all 128 pages
	pg1=pages[i].findall(".//WORD") #finds # of words on a page
	# print i,len(pg1)

#playing with page 34 of book (lots of words, is ex of prose; pg about textured veg protein products)
pg=pages[5].findall(".//WORD")
print [i.text for i in pg]