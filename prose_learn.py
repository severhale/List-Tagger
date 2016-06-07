import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import linalg
from sklearn import neighbors 
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file

feature_names = ['Numbers', 'Determiners', 'Proper Nouns']
target_names = np.array(['List', 'Mixed', 'Prose', 'Poetry'])
X, Y = load_svmlight_file("joined_data")
X = X.toarray()
Y = Y.astype(int)
XY = np.hstack((X,target_names[np.array(Y)][:,np.newaxis]))
df = pd.DataFrame(XY)
df.columns= feature_names + ["Class"]
df[feature_names] = df[feature_names].astype(float)


# colors=np.array([[1,0,0],[0,1,0],[0,0,1]])
# ["r","g","b"]
# i=0

# def sigma(x):
#     return 1/(1+np.exp(-x))

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
# X2 = X[:,[0,1]]
X2 = X
Y2 = Y

#Creat a learning set/test set split
X2learn,X2test,Ylearn,Ytest = cross_validation.train_test_split(X2, Y2, test_size=0.25, random_state=42)

#Do search for optimal parameters using 
#5-fold cross validation on the learning set
clf = GridSearchCV(svm.SVC(C=1), param_grid, cv=5)
clf.fit(X2learn, Ylearn)

#Print optimal parameter set
print "Optimal Parameters:", clf.best_params_

#Make predictions on the test set using optimal model
Yhat = clf.predict(X2test)

#Report the error rate
Err  = 1-metrics.accuracy_score(Yhat,Ytest)
print("Test Error Rate is: %.4f"%(Err,))