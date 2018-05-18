"""Main file for project 1 for EPFL EE-559 given by Francois fleuret
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

import random
import time

# fix seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# imports from this same folder
import dlc_bci as bci
import baseline
from nets import MyNet
from nets import MyNet2
from nets import MyNet3
import helpers
from helpers import BCELoss
from helpers import N_true

print('Version of PyTorch:', torch.__version__,', and we used: 0.4.0')


##### Import data
big_data = True # takes the hight resolution signal
train_input, train_target = bci.load(root ="./data_bci",download=False, one_khz=big_data)
# do the predictions on the low resolution signal
test_input, test_target = bci.load(root ="./data_bci", train = False, download=False, one_khz=False)
# prints infos about the shapes of the datasets
print(str(type(train_input)), train_input.size())
print(str(type(train_target)), train_target.size())
print(str(type(test_input)), test_input.size())
print(str(type(test_target)), test_target.size())


##### normalize the data
train_input = (train_input - torch.mean(train_input,0,keepdim=True))/torch.std(train_input,0,True)
test_input = (test_input - torch.mean(test_input,0,keepdim=True))/torch.std(test_input,0,True)

##### Make non-neural-network baselines
print('---------- BASELINES ----------')
train_input_baselines, train_target_baselines = helpers.augment_data(train_input,train_target)
baseline.lousy_baselines(train_input_baselines, train_target_baselines,test_input, test_target)


###### Prepare everything for training

# set parameters and initialises modules
net = MyNet3()
#optimizer = optim.Adam(net.parameters(), lr=0.00005)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.4)

# set some parameters
batch_size = 15
n_max_epochs = 150
n_val_samples = 16

# transform the dataset such that they can be fed to the network
print('\n---------- Shapes before net ----------')
train_X, train_Y, val_X, val_Y = helpers.prepare_train_with_validation(train_input,\
                                                                       train_target,\
                                                                       n_val_samples,\
                                                                       DA=True)
test_X, test_Y = helpers.prepare_test(test_input,test_target)

# initialize lists
train_losses = []
val_losses = []
val_acc = []
train_acc = []
VERBOSE = False

##### Train and evaluate the model
t0=time.clock() # save the current time

for epoch in range(n_max_epochs):  # loop over the dataset multiple times

    net.train() # sets the net in training mode --> dropout activated
    train_loss = 0.0
    num_correct_per_epoch = 0.0
    # train_X, train_Y = helpers.shuffle_data(input_=train_X, target_=train_Y) # shuffles the training between epochs
    for b in range(0, train_X.shape[0], batch_size):   #Train batch by batch

        # take care of the last batch (which size could not match with the remaining)
        end = batch_size
        if((b+end)>len(train_X)): end=b+end-len(train_X)

        # make predictions and make an optimization step
        output = net(train_X[b:b+end])
        loss = BCELoss(train_Y[b:b+end],output.squeeze())
        loss.backward()
        optimizer.step()
        net.zero_grad()

        # update loss and num of correct prediction per batch
        train_loss += loss.data
        num_correct_per_batch = N_true(output, train_Y[b:b+end])
        num_correct_per_epoch += num_correct_per_batch

    # evaluate validation
    net.eval() # set the net in eval mode --> dropout deactivated
    val_pred = net(val_X)

    # append values to lists to keep track of the evolution of loss and accuracy
    train_losses.append(train_loss) #add train loss
    train_acc.append(num_correct_per_epoch/train_X.shape[0]) # add train accuracy
    val_losses.append(BCELoss(val_Y,val_pred).data) #add validation loss
    val_acc.append(N_true(val_pred,val_Y)/val_X.shape[0]) #add validation accuracy
    if VERBOSE == True:
        print("\nEpoch ", epoch)
        print("classification accuracy on training set: {}".format(train_acc[epoch].data.numpy()))
        print("classification accuracy on validation set: {}".format(val_acc[epoch].data.numpy()))



t1=time.clock() # save the current time
# make predictions on test set
test_pred = net(Variable(torch.unsqueeze(test_input,1)))
test_acc = (N_true(net(Variable(torch.unsqueeze(test_input,1))),Variable(test_target)))*100/test_input.shape[0]

# prints info about the network
print('\n\n---------- RESULTS ----------')
print("time for training : ", t1-t0, ' seconds')
print("classification accuracy on test set: {}%".format(test_acc))

# plot training accuracy and final result
fig = plt.figure()
plt.plot(np.asarray(train_acc)*100,'b', label='train')
plt.plot(range(len(train_acc)),torch.ones(len(train_acc)).numpy()*test_acc.numpy(),label='final test')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy in %')
plt.xlim(0,n_max_epochs)
plt.grid(), plt.legend()
plt.show()
