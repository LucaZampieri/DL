import dlc_bci as bci
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import time
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

# fix seeds
torch.manual_seed(432)
random.seed(452)
np.random.seed(5687)

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

big_data = False
train_input, train_target = bci.load(root ="./data_bci", one_khz=big_data)
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size())
test_input, test_target = bci.load(root ="./data_bci", train = False, one_khz=False)
print(str(type(test_input)), test_input.size())
print(str(type(test_target)), test_target.size())

# normalize the data
train_input = (train_input - torch.mean(train_input,0,keepdim=True))/torch.std(train_input,0,True)
test_input = (test_input - torch.mean(test_input,0,keepdim=True))/torch.std(test_input,0,True)


#train_input, train_target = helpers.augment_data(train_input, train_target)
print(train_input.shape)
train_input = torch.cat((train_input.mean(1,keepdim=True),train_input.std(1,keepdim=True)),1)
test_input = torch.cat((test_input.mean(1,keepdim=True),test_input.std(1,keepdim=True)),1)
print(train_input.shape)
# test the function
X_test = test_input.data.numpy()
X_test = X_test.reshape(X_test.shape[0],-1)
Y_test = test_target.data.numpy()

X = train_input.data.numpy()
X = X.reshape(X.shape[0],-1)
Y = train_target.data.numpy()

ks = range(1,50)
for k in ks:
    print('--------------  KNN BASELINE -----------------')
    print('k:',k)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto',\
                                         p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    knn.fit(X, Y)
    nb_errors = 0
    for i,x in enumerate(knn.predict(X_test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/X_test.shape[0]*100,'%')


# end of file
