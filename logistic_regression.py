#importing reqd. modules
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

#loading dataset
iris = datasets.load_iris()

#slicing dataset
X = iris["data"][:, 3:]
Y = (iris["target"] == 2).astype(np.int)

#Training a logistic regression classifier
clf = LogisticRegression()
clf.fit(X,Y)

#prdicting result
pred = clf.predict([[2.6]])
print(pred)