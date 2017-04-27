#!python3
#Perform a simple KNN algorithm on the popular MNIST dataset
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
import os
#import CSV data with pandas
test = pd.read_csv("../data/mnist_test.csv", header=None).as_matrix()
test_label = []
Xtest = []
train= pd.read_csv("../data/mnist_train.csv", header=None).as_matrix()
train_label = []
Xtrain = []
print('Done loading data...')
#define some constants
TESTLEN = 10000
TRAINLEN = 60000
#test is now 10,000 x 785. First num in each array is label, rest is pixel data
#same thing for train. parse to separate labels from pixel data.
for i in range(TESTLEN):
    test_label.append(test[i][0])
    Xtest.append(test[i][1:]) #10000x784
for i in range(TRAINLEN):
    train_label.append(train[i][0])
    Xtrain.append(train[i][1:]) #60000x784
print('Done parsing data...')
#train using svm w/ rbf kernel
clf = SVC(probability = False, kernel = "rbf", C = 2.8, gamma = .0073)
#train using svm w/ linear kernel
#clf = LinearSVC(dual=False)
clf.fit(Xtrain,train_label)
print('Done fitting training data...')
predicted = clf.predict(Xtest)
print('Done predicting test samples...')
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_label,predicted))
print("Accuracy: %0.4f" % metrics.accuracy_score(test_label,predicted))
