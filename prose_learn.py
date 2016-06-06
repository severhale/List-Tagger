import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import linalg
from sklearn import neighbors 
from sklearn import metrics
from sklearn import svm
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file

#Define plot colors and options
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
labels=['sr','og','^b']
colors=['r','g','b']
feature_names = ['Numbers', 'Determiners', 'Proper Nouns']

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
    for i in [0,1,2]:
      plt.plot(x[y==i,0],x[y==i,1],labels[i]);
    plt.xlabel(feature_names[0]);
    plt.ylabel(feature_names[1]);
    plt.pcolormesh(gx1,gx2,gyhat,cmap=cmap_light)
    plt.colorbar();
    plt.axis('tight');
    plt.title(title);
    plt.show()

X, Y = load_svmlight_file("joined_data")
X = X.toarray()
Y[Y==-1]=0
#Select K
for K in range(1, 3):

    #Select distance metric
    metric='euclidean'

    #Select first two features
    X2 = X[:,[0,1]]

    #Fit the classifier
    clf = neighbors.KNeighborsClassifier(K,metric=metric)
    clf.fit(X2, Y)

    plt.figure()
    #Plot the classification function
    plot_classifier(X2,Y,clf,"KNN with K=%d"%(K,))

    #Make predictions using model
    Yhat = clf.predict(X2)

    #Report the error rate
    Err  = 1-metrics.accuracy_score(Yhat,Y)
    print("Training Error Rate is: %.4f"%(Err,))