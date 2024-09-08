#importing required modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()                 #loaded a diabetes dataset from sklearn

diabetes_X  =  diabetes.data[:,np.newaxis,2]        #taken out a feature coloumn from that diabetes dataset
diabetes_Y  =  diabetes.target                      #taken out label coloumn from diabetes dataset

diabetes_X_train = diabetes_X[:-30]                 #taken last 30 rows from col2 of diabetes for feature training
diabetes_X_test = diabetes_X[-30:]                  #taken first 30 rows from col2 of diabetes for feature testing

diabetes_Y_train = diabetes_Y[:-30]                 #taken last 30 rows from label set of diabetes for label training
diabetes_Y_test = diabetes_Y[-30:]                  #taken first 30 rows from label set of diabetes for label testing

model = linear_model.LinearRegression()             #created a linear regression model

model.fit(diabetes_X_train, diabetes_Y_train)       #machine is learning features n labels.......

diabetes_Y_predicted = model.predict(diabetes_X_test)   #imtehan of machine

mse = mean_squared_error(diabetes_Y_test, diabetes_Y_predicted) #taken mean squared error of test and predicted labels
print("mean squared error is: ", mse)

slope = model.coef_                                 #slope of linear regression model line
intercept = model.intercept_                        #intercept of linear regression model line
print("slope: ",slope, "intercept: ",intercept)

plt.scatter(diabetes_Y_train,diabetes_X_train)      #graphical repr. of linear regr. model points
plt.plot(diabetes_Y_train,diabetes_X_train)         #line
plt.show()