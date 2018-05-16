"""
file containing functions to load data and plot it
"""
import torch
from torch import Tensor
import math
import matplotlib.pyplot as plt

def generate_disc_set(nb):
    """
    Generates a number 'nb' of samples with their label.
    """
    input_ = Tensor(nb, 2).uniform_(0,1)
    disk_center = Tensor(nb, 2).fill_(0.5)
    R = 1/math.sqrt(2*math.pi) # Radius of the disk
    target = (R - (disk_center - input_).pow(2).sum(1).sqrt()).sign()#.long()
    target.add_(1).div_(2) # to transform [-1,1] into [0,1]
    return input_, target



def plot_data(input_, target_, figure_size = 6, show_plot = True):
    """
    Checks if the classes are balanced and plots the dataset.
    if show_plot it True, it shows the plot.
    """
    input_true = torch.Tensor(0,2)
    input_false = torch.Tensor(0,2)
    for i,x in enumerate(input_):
        if target_[i] == 0 :
            input_false = torch.cat((input_false, input_[i,:].view(-1,2)),0 )
        else :
            input_true = torch.cat( (input_true, input_[i,:].view(-1,2)),0 )
    print ('#samples:       ',input_.size())
    print ('Are the classes balanced?')
    print ('#true_samples:  ',input_true.size())
    print ('#false_samples: ',input_false.size())
    p1 = plt.figure(1,figsize=(figure_size,figure_size))
    plt.plot(input_true[:,0].numpy(),input_true[:,1].numpy(),'r.',label='within circle')
    plt.plot(input_false[:,0].numpy(),input_false[:,1].numpy(),'b.',label='outside circle')
    plt.xlim(0,1), plt.ylim(0,1)
    plt.legend(fontsize='x-large')
    plt.title('Distribution of generated data')
    if show_plot == True:
        plt.show()

def convert_to_one_hot(data_target):
    """convert data target to one-hot encoding"""
    return torch.cat((1-data_target.unsqueeze(1), data_target.unsqueeze(1)),1)

def normalize_data(data):
    mu, std = data.mean(),data.std()
    data.sub_(mu).div_(std)
