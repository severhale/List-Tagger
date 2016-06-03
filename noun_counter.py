from __future__ import division
import nltk
import numpy as np
import locale

def pos_count(text):
	tokenized_text = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(tokenized_text)
	tags = np.array([i[1] for i in tagged])
	unique, counts = np.unique(tags, return_counts=True)
	return unique, counts

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

fname = 'samplelist'
f = open(fname, 'r')
text = f.read()
text = text.decode('utf-8')
all_tags, all_data = pos_count(text)
feature_tags = ['Numbers', 'Determiners', 'Proper Nouns']
pos_tags = ['CD', 'DT', 'NNP']
data = []
size = np.sum(all_data)
for i in range(len(pos_tags)):
	index = np.argwhere(all_tags==pos_tags[i])
	if (len(index > 0) and len(index[0] > 0)):
		data.append(all_data[index[0][0]]/size)
	else:
		data.append(0)
print np.asarray((feature_tags, data)).T
f.close()