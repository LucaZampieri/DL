


import numpy as np
from sklearn import svm
from sklearn import neighbors
import torch
import dlc_bci as bci
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.decomposition import PCA


# transform into one-hot-encoding
def convert_to_one_hot(data):
    return np.eye(2)[data]




def get_data_as_numpy(HD_SIGNAL, CRIT):
    

    train_input, train_target = bci.load ( root = './data_bci',train=True, download=False, one_khz=HD_SIGNAL)
    print( str( type(train_input) ), train_input.size() )
    print( str( type(train_target) ), train_target.size() )

    test_input , test_target = bci.load ( root = './data_bci' , train = False, download=False, one_khz=HD_SIGNAL)
    print( str( type(test_input) ) , test_input.size() )
    print( str( type(test_target) ) , test_target.size() )

    print('train --- mean {:0.2f}, std: {:0.2f}'.format(train_input.mean(),train_input.std()),\
         'target mean: ',train_target.numpy().mean())
    print('test --- mean {:0.2f}, std: {:0.2f}'.format(test_input.mean(),test_input.std()),\
         'target mean: ',test_target.numpy().mean())
       
    train_input  = train_input.numpy()
    train_target = train_target.numpy()
    test_input   = test_input.numpy()
    test_target  = test_target.numpy()
    
    #train_target = one_hot_targets
    if CRIT=='MSE':
        train_target = convert_to_one_hot(train_target)
        test_target = convert_to_one_hot(test_target)

    return train_input, test_input, train_target, test_target

def from_numpy_to_torch(my_input, my_target, CRIT):
    my_input = Variable(torch.Tensor(my_input))
    if CRIT == 'MSE':
        my_target_target = Variable(torch.Tensor(my_target))
    elif CRIT == 'cross':
        my_target = Variable(torch.LongTensor(my_target))
    return my_input, my_target



###################### for baselines ####################################################################################
def my_pca_channels(X, nb_components_channels):
    nb_components = nb_components_channels
    XX = np.zeros((X.shape[0], nb_components,X.shape[2] ))

    for i in range(X.shape[0]):
        #print(X[i].shape)
        pca = PCA(n_components=nb_components)
        pca.fit(X[i])
        XX[i] = pca.components_
    return XX

def my_pca_samples(X, nb_components_channels):
    nb_components = nb_components_channels
    XX = np.zeros((X.shape[0], X.shape[1],nb_components ))

    for i in range(X.shape[0]):
        #print(X[i].shape)
        pca = PCA(n_components=nb_components)
        pca.fit(X[i])
        XX[i] = pca.components_
    return XX

def crappy_baselines(train_input, train_target,\
                     test_input, test_target,\
                     with_pca = False, nb_channels_pca = 10):
    
    print('--------------  SVM BASELINE -----------------')
    clf = svm.SVC(gamma=0.001, C=100.)
    X = train_input.data.numpy()
    
    if with_pca:
        X = my_pca_channels(X=X, nb_components_channels=nb_channels_pca)
    X = X.reshape(X.shape[0],-1)
    Y = train_target.data.numpy()
    print(X.shape,Y.shape)

    clf.fit(X[:-1],Y[:-1]) 

    # test the function
    test = test_input.data.numpy()
    if with_pca:
        test = my_pca_channels(X=test, nb_components_channels=nb_channels_pca)
    test = test.reshape(test.shape[0],-1)
    Y_test = test_target.data.numpy()
    print(test.shape,Y_test.shape)
    nb_errors = 0
    for i,x in enumerate(clf.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
    
    
    print('--------------  KNN BASELINE -----------------')
    knn = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto',\
                                         leaf_size=300, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    knn.fit(X, Y) 
    print(knn.predict(test))
    nb_errors = 0
    for i,x in enumerate(knn.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
######################################### end for baselines #########################################################