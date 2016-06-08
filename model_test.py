from sklearn.externals import joblib
import datahelper
import numpy as np

target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])

# load model from file, test on xml
clf = joblib.load('Model/model.pkl')
name, nums, pages = datahelper.tokenized_from_xml_range('Mixed/bookman37unkngoog_djvu.xml', 1, 50)
data = datahelper.get_data(pages, ['CD', 'DT', 'NNP'])
prediction = clf.predict_proba(data)
for i in range(len(data)):
	j = np.argmax(prediction[i])
	print data[i]
	print "Page %s is %s with %.4f confidence" % (nums[i], target_names[j], prediction[i][j])