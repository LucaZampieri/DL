"""
This files contains functions used to plot or make some small changes in the data
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import time
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)



def augment_data(train_X, train_Y):

    big_data_set = torch.Tensor(0,28,50)
    big_target = torch.Tensor(0)
    for i in range(10):
        shifted_data = train_X[:,:,i::10]
        big_data_set = torch.cat((big_data_set,shifted_data),0)
        big_target = torch.cat((big_target,train_Y.type(torch.FloatTensor)),0)

    return big_data_set, big_target.type(torch.LongTensor)

def visualize_data(train_input, train_target, test_input, test_target):
    n_train_samples = train_target.size()[0]
    print('Number of training samples = {}'.format(n_train_samples))
    n_test_samples = test_target.size()[0]
    print('Number of testing samples = {}'.format(n_test_samples))
    n_channels = train_input.size()[1]
    print('Number of channels = {}'.format(n_channels))
    n_readings = train_input.size()[2]
    print('Number of readings per channel = {}'.format(n_readings))
    freq = 2*n_readings
    tt = np.arange(0,0.5,1/freq)
    for i in range(n_channels):
        plt.plot(tt,train_input[0,i,:].numpy())
    plt.show()
    plt.plot(tt,train_input[10:27,8,:].numpy().transpose())
    plt.title('')
    plt.show()

def prepare_test(test_input,test_target, DA = False):
    """reform the test set to be fed to the network.
    if DA = True, do data augmentation on the test set as well"""
    if DA == True:
        test_x, test_y = augment_data(test_input, test_target)
    else:
        test_x, test_y = test_input, test_target
    # unsqueeze to get the channels as second dimension of the data i.e. adds one dimension
    test_X = Variable(torch.unsqueeze(test_x,1))
    test_Y = Variable(test_y)
    print("Test shapes \n" ,test_X.shape, test_Y.shape)
    return test_X, test_Y

def prepare_train_with_validation(train_input,train_target,n_val_samples, DA = True):
    """separates the training data into training and validation.
    if DA is true, then"""
    n_train_samples = train_input.shape[0]-n_val_samples
    val_idx = random.sample(range(n_train_samples), int(n_val_samples)) # pick n_validation_samples high resolution samples
    train_idx = np.delete(np.arange(len(train_input)),val_idx) # deletes indices used in validation

    # Prepare validation data
    if DA == True: # Data augment the validation set
        val_x, val_y = augment_data(train_input[torch.LongTensor(val_idx)],(train_target[torch.LongTensor(val_idx)]))
    else :
        val_x, val_y = (train_input[torch.LongTensor(val_idx)],(train_target[torch.LongTensor(val_idx)]) )
    # unsqueeze to treat channels as a 2nd dimenstion of data
    val_X = Variable(torch.unsqueeze(val_x,1),requires_grad=True)
    val_Y = Variable(val_y)
    print("Validation shapes \n", val_X.shape, val_Y.shape)

    # Prepare training data
    if DA == True: # data augment the training set
        train_x, train_y = augment_data(train_input[torch.LongTensor(train_idx)], train_target[torch.LongTensor(train_idx)])
    else :
        train_x, train_y = (train_input[torch.LongTensor(train_idx)], train_target[torch.LongTensor(train_idx)])
    # unsqueeze to treat channels as a 2nd dimenstion of data
    train_X = Variable(torch.unsqueeze(train_x,1))
    train_Y = Variable(train_y)
    print("Train shapes \n", train_X.shape, train_Y.shape)
    return train_X, train_Y, val_X, val_Y

def shuffle_data(input_, target_):
    """makes random permutations in the data
    :input: input and target torch Variable
    :output: randomly permutated input and target Variables"""
    my_randperm =torch.randperm(input_.size(0))
    return input_[my_randperm], target_[my_randperm]


def N_true(Y_pred, Y_true):
    """Computes th number of correctly labeled Y_pred with groudtruth Y_true"""
    return torch.sum(torch.round(Y_pred).squeeze()==Y_true.float()).type(torch.FloatTensor)

def BCELoss(Y_true,Y_pred):
    """ Binary Cross Entropy Personalized. """
    l = Variable(torch.Tensor(len(Y_true)))
    for i in range(len(Y_true)):
        if(Y_true[i]==1).data.numpy():
            l[i] = - Y_pred[i].log()
        else:
            l[i] = - (1-Y_pred[i]).log()

    return torch.mean(l)
