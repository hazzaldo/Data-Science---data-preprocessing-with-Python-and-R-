# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 01:30:59 2018

@author: hazzaldo
"""

# Importing the libraries:
# =========================
# numpy is a library that contains mathematical tools. 
# We will need this library for any mathematical operations. We created 'np' 
# as the shortcut so we don't have to reference the long name of the library.
import numpy as np
#From the matplotlib library we're importing the sub-library pyplot.
#This library help us plot charts. 
import matplotlib.pyplot as plt
#pandas library is the best library for importing and managing datasets.
import pandas as pd


# Importing the dataset:
# =========================
dataset = pd.read_csv('Data.csv')

# Create the Matrix Of Features (independant variables):
#Left of the first colon means how many rows we want
#to include from the imported dataset csv file.
#In this case no number specified means we want to
#include all rows.
#Right of the second colon the number specified
#indicate how many columns we want to include. 
#in this case all columns except the last one.
#.values means we want to get all the values in 
#the csv file. We assign all these values to the
#X variable as a matrix (or array of arrays) data type.
X = dataset.iloc[:, :-1].values

# Create the Dependant Variable Vector (dependant variables):
# In this case we just need to include the last column in our
#imported dataset. We start counting from column Country, starting
#with the count of 0 (with python you always count from index 0).
#So in this case column Purchased is column index 3. We don't include
#the second colon as we only want to include one column from the end.
#Not all column with an exception of the column at the end.
y = dataset.iloc[:, 3].values


# Taking care of missing data - calculate mean
# ============================================
# sklearn (or sikat learn) is a framework that contains amazing libraries
# to make machine learning models. From sklearn we import 'preprocessing'
# library that contains a lot of classes and methods to preprocess
# any data sets. We're then importing the Imputer class which takes
#care of those missing data.
from sklearn.preprocessing import Imputer
# NaN is what is represented when there's missing data in the matrix dataset.
# 'mean' is actually the default value for strategy param. So we normally 
# wouldn't need to include it, but we include it here just for readibility.
# axis param indicate whether we're calculating the mean of an entire column,
# or row to get the missing value. Of course in this case it's a column, so
# the value is 0 (otherwise 1 for row).
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# we use 'fit' method to pass which matrix we want to calculate missing data,
# in this case X. Then we specify which columns we want to include. First colon
# without any numbers means include all rows, and 1:3, means include the second
# (python starts with index 0) and 3  seems to mean column 4, but it's actually 
# column 3, because the upper bound or column 3 (4 column in reality) is not to
# be included. So if you say 0:1 it means include column from 0 to 1, but 1 is
# not included. Same 1:3 means 1 to 3 but exclusive of 3 (meaning 1 and 2). 
# we're assigning all of this to the same object imputer, to make X matrix fit
# in to the object.
imputer = imputer.fit(X[:, 1:3])
# replace the missing data with the mean
X[:, 1:3] = imputer.transform(X[:, 1:3]) 


# Encoding categoral data - changing string data to number categories for ease of mathematical modelling
# ============================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# an instance of LabelEncoder that represent the X matrix.
labelencoder_X = LabelEncoder()
# We fit the first column of matrix X (country column) to object labelencoder. 
# the fit_transform would return an encoded data of that column, i.e. in 
# numerical categories rather than string data. We then assign this to first
# column of matrix X to update the matrix with this change.
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Create the dummy data, as these are strings and there's no number order 
# relation between France, Germany, Spain ..etc. So category numbers should be
# dummy number data, i.e. 1 if country exist in the entry and 0 if it doesn't.
# First param we specify which column we want to One Hot Encode. In this case
# it's column 0 (country)
onehotencoder = OneHotEncoder(categorical_features = [0])
# now that column 0 is one-hot-encoded, we just fit it back to Matrix X.
X = onehotencoder.fit_transform(X).toarray()

# Encode the Dependant Variable Vector. As it is Dependant Variable Vector, the
# Machine Learning model will know automatically that it is a category and 
# that there's no order between the two data types. So we only need to 
# LabelEncode it. No need for OneHotEncode. 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into a Training set and a Test set
# ========================================================
from sklearn.model_selection import train_test_split
# X_train is the training part of the Matrix Of Features
# X_test is the test part of the Matrix Of Features
# y_train is the training part of the Dependant Variable Vector (that is associated to X_train, that means that we have the same indices for both with the same entries)
# y_test is the test part of the Dependant Variable Vector (associate to X_test)
# 1st param is the X and y matrices. 2nd param is the size of the test dataset.
# A good size for the test data would be between 20% to 30% of the entire 
# dataset. In this case we specified 20% (0.2).
# In our case we have 10 entries (or 10 observations) in our dataset, meaning
# 2 observations will be test data and the remaining are training data.
# 3rd param is just random number state generator. If you want the same result 
# as say another data-scientist or analyst, who's also training and testing the
# the data so that you can both compare and make sure the ML model outputs 
# the same consist result given the same input data, then you can both put the
# same number. In our case we simply chose 0. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 











