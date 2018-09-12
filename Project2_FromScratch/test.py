"""
main executable for project 2 of the DeepLearning course given at EPFL by
Francois Fleuret (EE-559).
"""


########## imports ##########
import torch
import math
from torch import Tensor
import matplotlib.pyplot as plt
# imports from this same folder
import loadData
import baseline
import optim
import modules
import trainAndTest

# print and compare the versions
print('Pytorch version: ',torch.__version__, '.  We used 0.3.1.post2')

# Set seed for reproducibility
torch.manual_seed(404)


########## get the data ##########

# Generate data with target 0 if outside the circle or 1 if inside
train_input, train_target = loadData.generate_disc_set(1000)
test_input, test_target = loadData.generate_disc_set(1000)

# see if data are close to be balanced and plot the dataset
loadData.plot_data(train_input, train_target, show_plot = False)

# check what accuracy we could expect with a simple baseline
# e.g. with a linear classifier after having added the L2 norm of the points
baseline.baseline_linear_model(train_input,train_target,test_input,test_target)

# convert targets to one-hot encoding
train_target = loadData.convert_to_one_hot(train_target)
test_target  = loadData.convert_to_one_hot(test_target)
print('Targets converted to one-hot')

# Normalize inplace the data
loadData.normalize_data(train_input)
loadData.normalize_data(test_input)


########## modules and model #########

# define optimizers and losses as list, to be able to juggle with them
optimizers = [optim.SGDOptimizer, optim.SGDmomOptimizer, optim.AdamOptimizer, optim.BFGSOptimizer]
losses = [modules.LossMSE]

# define layers and activations
Lin1 = modules.Linear(2,25)
Lin2 = modules.Linear(25,25)
Lin3 = modules.Linear(25,2)
act1 = modules.ReLU()
act2 = modules.ReLU()
act3 = modules.Tanh()
#act4 = modules.Sigmoid()

# combine the layers together
layers = [
    Lin1,
    act1,
    Lin2,
    act2,
    Lin3,
    act3]

# set parameters for the run
lr = 0.005 # learning rate, for BFGS multiply by 10 to 100
epochs = 250 # epochs for the run
mini_batch_size = 50 # mini_batch_size for the run

# initialize loss, model and optimizer
loss = losses[0]() # [0] for MSELoss
model = modules.Sequential(layers, loss)
optimizer = optimizers[2](model, lr) # [2] for Adam optimizer

# train the model
print('\n----------------------------------------')
print('          TRAINING THE MODEL          \n\n')
loss_list, train_acc = trainAndTest.train(model, optimizer, loss, train_input, train_target,\
                                          epochs, mini_batch_size, verbose = False)

# print results
print('\n----------------------------------------')
print('          RESULTS          \n')
print('learning rate: ', lr, '  ||  epochs: ', epochs, '  ||  mini_batch_size: ', mini_batch_size)
print('optimizer: ', optimizer.name)
print('minimum training loss: ',min(loss_list))
print('maximum training accuracy on train: ',max(train_acc)/train_input.size(0)*100,'%')
print('On test accuracy: ', trainAndTest.test(model,test_input,test_target)/test_input.size(0)*100,'%' )

# plot training curves
print('\n----------------------------------------')
print('          PLOTTING          \n')
trainAndTest.plot_loss_accuracy(loss_list, train_acc)

# plot end of file
print('END OF FILE')

# END OF FILE
