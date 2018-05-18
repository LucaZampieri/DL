import dlc_bci as bci
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import time
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F


# from this same folder
import baseline
from nets import MyNet
from nets import MyNet2
from nets import MyNet3
import helpers


import numpy as np
from sklearn import svm
from sklearn import neighbors
import torch
import dlc_bci as bci
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.decomposition import PCA
# for cross validation:
from sklearn.model_selection import KFold

big_data = True
train_input, train_target = bci.load(root ="./data_bci", one_khz=big_data)
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size())
test_input, test_target = bci.load(root ="./data_bci", train = False, one_khz=True)
print(str(type(test_input)), test_input.size())
print(str(type(test_target)), test_target.size())

# normalize the data
train_input = (train_input - torch.mean(train_input,0,keepdim=True))/torch.std(train_input,0,True)
test_input = (test_input - torch.mean(test_input,0,keepdim=True))/torch.std(test_input,0,True)


train_input, train_target = helpers.augment_data(train_input, train_target)
test_input, test_target = helpers.augment_data(test_input, test_target)

gammas = [0.00005,0.001, 0.005,0.01,0.1,1,2]
Cs = [50,100,200,500,100]
gammas = [0.00005]
Cs = [40,50,80,100]
for gamma in gammas:
    for C in Cs:
        print('--------------  SVM BASELINE -----------------')
        print('gamma:',gamma,' C:',C)
        clf = svm.SVC(gamma=gamma, C=C)
        X = train_input.data.numpy()

        X = X.reshape(X.shape[0],-1)
        Y = train_target.data.numpy()

        clf.fit(X[:-1],Y[:-1])

        # test the function
        test = test_input.data.numpy()
        test = test.reshape(test.shape[0],-1)
        Y_test = test_target.data.numpy()

        nb_errors = 0
        for i,x in enumerate(clf.predict(test)):
            if x!= Y_test[i]:
                nb_errors += 1
        print(nb_errors/test.shape[0]*100,'%\n')


# end of file
