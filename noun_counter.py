from __future__ import division
import nltk
import numpy as np

# Count proper nouns
def is_nounPOS(pos):
	return pos == "NNP"
	# return pos[:2] == "NN"

def proper_noun_count(text):
	tokenized_text = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(tokenized_text)
	tags = [i[1] for i in tagged]
	counts = np.array([tags.count(i) for i in np.unique(tags)])
	noun_count = np.sum(counts[np.array(map(is_nounPOS, np.unique(tags)))])
	total_count = np.sum(counts)
	return (100*noun_count/total_count)

# fname = raw_input('Enter file to analyze: ')
fname = 'sampletext'
f = open(fname, 'r')
text = f.read()
text = text.decode('utf-8')
print "%.2f" % (proper_noun_count(text)) + "%" + " proper nouns"
f.close()

