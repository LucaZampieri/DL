"""
File contains the functions to train and test the models
and to do the corresponding plots
"""

import torch
from torch import Tensor
import math
import tqdm
from tqdm import trange
import matplotlib.pyplot as plt


def train(model, optimizer, loss, train_input, train_target, epochs, mini_batch_size, verbose = True):
    """Trains the model, returns information about the training
    :inputs: their name is explicit
    :output: one list with the loss and one with the number of correct prediction
    at each epoch"""
    loss_list = []
    train_acc = []
    if verbose == True:
        my_range = range(epochs)
    else:
        my_range = trange(epochs) # nice visualisation

    for epoch in my_range:
        train_loss = 0
        nb_corrects = 0
        optimizer.zero_grad()
        train_input, train_target = shuffle_data(train_input, train_target)
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input[b:b+mini_batch_size,:])
            batch_loss = loss.forward(output, train_target[b:b+mini_batch_size,:])
            train_loss += batch_loss
            model.backward()
            # optimize:
            optimizer.step()
            optimizer.zero_grad()
            # check accuracy:
            _, pred = torch.max(output,1)
            _, target = torch.max(train_target[b:b+mini_batch_size],1)
            nb_corrects += torch.sum(pred==target)

        # append to the output lists
        loss_list.append(train_loss)
        train_acc.append(nb_corrects)
        if verbose == True:
            print("\n Epoch", epoch)
            print("\nTraining accuracy : ", nb_corrects/train_target.size(0)*100)
            print("\nTraining loss : ", train_loss)
    return loss_list, train_acc

def test(model, test_input, test_target):
    """make predictions on the test set"""
    output = model.forward(test_input)
    _, pred = torch.max(output,1)
    _, target = torch.max(test_target,1)
    test_accuracy = torch.sum(pred==target)
    return test_accuracy

def shuffle_data(input_, target_):
    """Shuffle data in the same way for input and for target
    :input: input and target to shuffle
    :output: shuffled input and target"""
    my_randperm =torch.randperm(input_.size(0))
    return input_[my_randperm], target_[my_randperm]

def plot_loss_accuracy(loss_list, train_acc, save=False):
    plt.figure(5, figsize=(15,6))
    plt.subplot(121)
    plt.plot(loss_list, label='loss function')
    plt.title('Loss per epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(train_acc, label='# correct samples')
    plt.title('Number of correct samples')
    plt.grid(True)
    plt.legend()

    if save == True:
        plt.savefig('loss and accuracy')
    plt.show()

# End of file
