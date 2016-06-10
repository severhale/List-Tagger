from sklearn.externals import joblib
import datahelper
import numpy as np

target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])

# load model from file, test on xml
clf = joblib.load('Model/model.pkl')
pages = datahelper.get_pages('Mixed/famoussinglepoem00stevuoft_djvu.xml', 1, 360)
name = datahelper.get_book_name(pages)
nums, text = datahelper.tokenize_pages(pages)
data = datahelper.get_data(text, ['CD', 'DT', 'NNP'])
plength = datahelper.get_paragraph_length(pages)
X = np.hstack((data, plength))
prediction = clf.predict_proba(X)
for i in range(len(X)):
	j = np.argmax(prediction[i])
	print prediction[i]
	print "Page %s is %s with %.4f confidence" % (nums[i], target_names[j], prediction[i][j])