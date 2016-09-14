#!python3
#Perform a simple KNN algorithm on the popular MNIST dataset
import pandas as pd
import numpy as np
from scipy import stats
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
TESTLEN = 200
TRAINLEN = 5000
K = int(input("K-NN starting... please enter a value for K: "))
#test is now 10,000 x 785. First num in each array is label, rest is pixel data
#same thing for train. parse to separate labels from pixel data.
for i in range(TESTLEN):
    test_label.append(test[i][0])
    Xtest.append(test[i][1:]) #10000x784
for i in range(TRAINLEN):
    train_label.append(train[i][0])
    Xtrain.append(train[i][1:]) #60000x784
print('Done parsing data...')
#find distance from each test image to all training images
dists = np.ndarray((TESTLEN,TRAINLEN))
for i in range(TESTLEN):
    for j in range(TRAINLEN):
        dists[i][j] = np.linalg.norm(Xtest[i]-Xtrain[j])

print('Done analyzing distances...')
# predict based on k nearest neighbors
def kNN(d,k):
    values = []
    for i in range(TESTLEN):
        closest = d[i].argsort()[:k]
        nums = [train_label[i] for i in closest]
        mode = int(stats.mode(nums).mode)
        values.append(mode)
    print('Done predicting...')
    return values
predicted_label = kNN(dists,K)
errorCount = 0
for i in range(TESTLEN):
    if predicted_label[i] - test_label[i] != 0:
        errorCount += 1

output = "Error Rate: {}%".format(errorCount*100/1000)
print(output)
os.system('say "your program has finished"')
