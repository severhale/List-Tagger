import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix, andrews_curves
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import linalg
from sklearn import neighbors 
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

feature_names = ['Numbers', 'Determiners', 'Proper Nouns', 'Paragraph Length']
target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])
X, Y = load_svmlight_file("joined_data")
X = X.toarray()
Y = Y.astype(int)
XY = np.hstack((X,target_names[np.array(Y)][:,np.newaxis]))
df = pd.DataFrame(XY)
df.columns= feature_names + ["Class"]
df[feature_names] = df[feature_names].astype(float)

#Define plot colors and options
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#AFAFAF'])
labels=['sr','og','^b', 'd']
colors=['r','g','b', 'o']

#Define classifier decision boundary plotting function
def plot_classifier(x,y,clf,title):

    #Prepare grid for plotting decision surface
    gx1, gx2 = np.meshgrid(np.arange(min(x[:,0]), max(x[:,0]),(max(x[:,0])-min(x[:,0]))/200.0 ),
                         np.arange(min(x[:,1]), max(x[:,1]),(max(x[:,1])-min(x[:,1]))/200.0))
    gx1l = gx1.flatten()
    gx2l = gx2.flatten()
    gx   = np.vstack((gx1l,gx2l)).T

    #Compute a prediction for every point in the grid
    gyhat = clf.predict(gx)
    gyhat = gyhat.reshape(gx1.shape)

    #Plot the results
    for i in [0,1,2,3]:
      plt.plot(x[y==i,0],x[y==i,1],labels[i]);
    plt.xlabel(feature_names[2]);
    plt.ylabel(feature_names[3]);
    plt.pcolormesh(gx1,gx2,gyhat,cmap=cmap_light)
    plt.colorbar();
    plt.axis('tight');
    plt.title(title);

def sigma(x):
    return 1/(1+np.exp(-x))

def get_feature_importance():
    rf = RandomForestClassifier(n_estimators=1, criterion='entropy', max_features=2, max_depth=5, bootstrap=True, oob_score=True, n_jobs=2, random_state=33)
    rf = rf.fit(X2learn, Ylearn)
    return rf.feature_importances_

# for c in target_names:
#     thisdf = df[df["Class"]==c]
#     thisc=np.outer(sigma((thisdf[feature_names[0]])/2.5),1*colors[i])
#     if i==0: 
#         ax=thisdf.plot(kind='scatter', x=feature_names[0], y=feature_names[1], s=thisdf[feature_names[2]]*50, c=thisc, alpha=0.75)
#     else:
#         thisdf.plot(kind='scatter', x=feature_names[0], y=feature_names[1], s=thisdf[feature_names[2]]*50, c=thisc, alpha=0.75,ax=ax)    
#     plt.xlim(0, .4)
#     plt.ylim(0, .2)  
#     plt.hold(True)
#     i=i+1
# ax.legend(target_names)
# plt.show()

# Select K
#Define the parameter grid
param_grid = [{'C': [0.01,0.1,1, 10, 100], 'kernel': ['rbf'],'gamma': [0.1,1,10,100]}]

#Select just the first two features
# X2 = X[:,[2,3]]
X2 = X
Y2 = Y

#Creat a learning set/test set split
X2learn,X2test,Ylearn,Ytest = cross_validation.train_test_split(X2, Y2, test_size=0.25, random_state=22)

#Do search for optimal parameters using 
#5-fold cross validation on the learning set
clf = GridSearchCV(svm.SVC(C=1, probability=True), param_grid, cv=5)
clf.fit(X2learn, Ylearn)
# joblib.dump(clf, 'Model/model.pkl')


# plt.figure(1)
# plot_classifier(X2learn,Ylearn,clf,"SVM with Learning Set")

# #Plot the classification function with test set
# plt.figure(2)
# plot_classifier(X2test,Ytest,clf,"SVM with Test Set")
# plt.show()

#Print optimal parameter set
print "Optimal Parameters:", clf.best_params_

#Make predictions on the test set using optimal model
Yhat = clf.predict(X2test)

#Report the error rate
Err  = 1 - metrics.accuracy_score(Ytest, Yhat)
F1 = metrics.f1_score(Ytest, Yhat, average=None)
print "Predicted:",Yhat
print "Actual:",Ytest
print "F1: ",F1
print("Test Error Rate is: %.4f"%(Err,))

# # code to display data viz
# XY = np.hstack((X,target_names[Y][:,np.newaxis]))
# df = pd.DataFrame(XY)
# df.columns= feature_names + ["Class"]
# df[feature_names] = df[feature_names].astype(float)
# print df.corr()
# df.hist(bins=np.arange(0,1,0.005),sharex=True);
# scatter_matrix(df, alpha=0.2, figsize=(8, 8), diagonal='none');
# plt.figure()
# andrews_curves(df, 'Class')
# plt.show()