#importing required modules
from sklearn import datasets 
from sklearn.neighbors import KNeighborsClassifier

#loading iris dataset from sklearn
iris = datasets.load_iris()                         

#loading features from iris dataset
features = iris.data    
#loading labels form iris dataset
labels = iris.target                               

#print(features[0], labels[0])                      #printing 1st record from iris dataset

#creating a classifier
clf = KNeighborsClassifier()  
#training the classifier                      
clf.fit(features, labels)                            

#predicting results
pred = clf.predict([[9.1, 9.5, 6.4, 0.2]])          
print(pred)
