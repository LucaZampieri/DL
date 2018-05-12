


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


# transform into one-hot-encoding
def convert_to_one_hot(data):
    return np.eye(2)[data]


def get_data_as_numpy(hd_signal, criterion):
    
    train_input, train_target = bci.load ( root = './data_bci',train=True, download=False, one_khz=hd_signal)
    print( str( type(train_input) ), train_input.size() )
    print( str( type(train_target) ), train_target.size() )

    test_input , test_target = bci.load ( root = './data_bci' , train = False, download=False, one_khz=hd_signal)
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
    if criterion=='MSE':
        train_target = convert_to_one_hot(train_target)
        test_target = convert_to_one_hot(test_target)

    return train_input, test_input, train_target, test_target

def from_numpy_to_torch(my_input, my_target, criterion):
    my_input = Variable(torch.Tensor(my_input))
    if criterion == 'MSE':
        my_target_target = Variable(torch.Tensor(my_target))
    elif criterion == 'cross':
        my_target = Variable(torch.LongTensor(my_target))
    return my_input, my_target

def shuffle_data(input_, target_):
    my_randperm =torch.randperm(input_.size(0))
    return input_[my_randperm], target_[my_randperm]

def normalize_data(dataset):
    mu  = np.mean(dataset,axis=0)
    std = np.std(dataset,axis=0)
    return (dataset-mu)/std

def noise_DA(my_input, my_target, std=1/2):
    """
    input: a numpy array and its corresponding target
    output: the noised data augmented numpy array, and corresponding target
    """
    mean = 0.0
    noised_signals = np.zeros(my_input.shape)
    for i in range(my_input.shape[0]):
        noised_signals[i] = my_input[i] + np.random.normal(mean, std, my_input[i].shape)
    noised_DA_input = np.concatenate((my_input, noised_signals), 0)
    corresponding_target = np.concatenate((my_target, my_target), 0)
    return noised_DA_input, corresponding_target

"""def noise_DA_tensor(my_input, my_target, std=1/2):
    mean = 0.0
    noised_signals = torch.zeros(my_input.shape)
    torch.zeros(my_input.shape)
    for i in range(my_input.shape[0]):
        noised_signals[i] = my_input[i] + np.random.normal(mean, std, my_input[i].shape)
    noised_DA_input = np.concatenate((my_input, noised_signals), 0)
    corresponding_target = np.concatenate((my_target, my_target), 0)
    return noised_DA_input, corresponding_target
"""

def subsampling_DA(train_input, train_target, subsampling_list=range(1,10)):
    """
    input: input tensor and target tensor
    prints the size of the DA tensors
    output: the subsampled DataAungmented tensors
    """
    big_data_set = train_input[:,:,0::10]
    big_target = train_target.type(torch.FloatTensor)
    for i in subsampling_list:
        
        shifted_data = train_input[:,:,i::10]
        big_data_set = torch.cat( (big_data_set,shifted_data),  0)
        big_target = torch.cat((big_target,train_target.type(torch.FloatTensor)),0)

    train_input = big_data_set
    train_target = big_target.type(torch.LongTensor)
    return train_input, train_target

def cross_validation(train_input, train_target,n_splits_K_fold):
    """
    input: train_input, train_target
    output: train_indices and test indices
    """
    X = train_input
    y = train_target
    kf = KFold(n_splits=n_splits_K_fold)
    kf.get_n_splits(X) 
    train_indices, test_indices = kf.split(X)
    print('KFold with K=',len(train_indices))
    return train_indices, test_indices

def final_check(train_input,train_target,test_input,test_target):
    """
    prints the size and type of our data
    """
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print( 'train_input: ', str( type(train_input) ), train_input.size() )
    print( 'train_target: ',str( type(train_target) ), train_target.size() )
    #print(train_input.mean())
    print ('\n----------------------------------------------------------------')
    print('train --- mean {:0.5f}, std: {:0.2f}'.format(train_input.data.mean(),train_input.data.std()),\
         'target mean: ',train_target.data.numpy().mean())
    print('test --- mean {:0.5f}, std: {:0.2f}'.format(test_input.data.mean(),test_input.data.std()),\
         'target mean: ',test_target.data.numpy().mean())


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

def lousy_baselines(train_input, train_target,\
                     test_input, test_target,\
                     with_pca = False, nb_channels_pca = 10):
    """
    takes torhs variables as inputs and compute some baselines
    """
    
    print('--------------  SVM BASELINE -----------------')
    clf = svm.SVC(gamma=0.001, C=100.)
    X = train_input.data.numpy()
    
    if with_pca:
        X = my_pca_channels(X=X, nb_components_channels=nb_channels_pca)
    X = X.reshape(X.shape[0],-1)
    Y = train_target.data.numpy()

    clf.fit(X[:-1],Y[:-1]) 

    # test the function
    test = test_input.data.numpy()
    if with_pca:
        test = my_pca_channels(X=test, nb_components_channels=nb_channels_pca)
    test = test.reshape(test.shape[0],-1)
    Y_test = test_target.data.numpy()
    
    nb_errors = 0
    for i,x in enumerate(clf.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
    
    
    print('--------------  KNN BASELINE -----------------')
    knn = neighbors.KNeighborsClassifier(n_neighbors=4, weights='uniform', algorithm='auto',\
                                         leaf_size=300, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    knn.fit(X, Y) 
    #print(knn.predict(test))
    nb_errors = 0
    for i,x in enumerate(knn.predict(test)):
        if x!= Y_test[i]:
            nb_errors += 1
    print(nb_errors/test.shape[0]*100,'%')
######################################### end for baselines #########################################################

############### for weights initilaisation
def print_info(m):
    print(m.weight.data.norm())
    print(m.bias.data.norm())
    
    
def weights_init(m):
    classname = m.__class__.__name__
    print(m)
    if classname.find('Conv') != -1:
        print_info(m)
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0.0, 0.02)
        print_info(m)
        #xavier(m.weight.data)
        #xavier(m.bias.data)
    if classname.find('Linear') != -1:
        print_info(m)
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
        print_info(m)
    elif classname.find('BatchNorm') != -1:
        print_info(m)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        print_info(m)
    print('\n\n')
    

    
