"""
This files contains the functions used to make the baselines results
"""
import numpy as np
from sklearn import svm
from sklearn import neighbors
import torch
import dlc_bci as bci
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import time

def lousy_baselines(train_input, train_target,\
                     test_input, test_target):
    """
    takes torchs variables as inputs and compute some baselines
    for SVM and for KNN
    """

    # transorm data to appropriate format
    X = train_input.data.numpy()
    X = X.reshape(X.shape[0],-1)
    Y = train_target.data.numpy()
    test = test_input.data.numpy()
    test = test.reshape(test.shape[0],-1)
    Y_test = test_target.data.numpy()

    # make SVM_baseline
    print('--------------  SVM BASELINE -----------------')
    gamma = 0.00005
    C = 50.
    t0_SVM = time.clock()
    clf = svm.SVC(gamma=gamma, C=C)
    clf.fit(X[:-1],Y[:-1])

    # test the function
    nb_errors = 0
    for i,x in enumerate(clf.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
    t1_SVM = time.clock()
    print('Time for SVM with gamma:',gamma,', C:',C,' --- ',\
            t1_SVM-t0_SVM,' seconds')


    print('--------------  KNN BASELINE -----------------')
    k = 38
    t0_KNN = time.clock()
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto',\
                                         leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    knn.fit(X, Y)
    # test the function
    nb_errors = 0
    for i,x in enumerate(knn.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
    t1_KNN = time.clock()
    print('Time for KNN with k:',k,', metric:minkowski',' --- ',\
            t1_KNN-t0_KNN,' seconds')
######################################### end for baselines #########################################################
