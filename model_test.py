from sklearn.externals import joblib
import datahelper
clf = joblib.load('Model/model.pkl')
name, nums, pages = datahelper.tokenized_from_xml_range('Mixed/workmaterialsa17unit_djvu.xml', 1, 50)
data = datahelper.get_data(pages, ['CD', 'DT', 'NNP'])
prediction = clf.predict_proba(data)
for i in range(1, 50):
	print "Page %d -- List: %.4f Mixed: %.4f Prose: %.4f Poetry: %.4f" % (i, prediction[i][0], prediction[i][1], prediction[i][2], prediction[i][3])