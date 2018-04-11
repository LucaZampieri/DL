import torch
import math

from torch import Tensor
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self,nb_hidden,nb_init_filters = 64, nb_convs=2, kernel_size=5, length_signal = 50):
        super(Net, self).__init__()
        # parameters
        self.nb_init_filters = nb_init_filters # nb of filters conv1
        self.nb_convs = nb_convs # nb of additional conv layers
        self.kernel_size = kernel_size # size of the kernels
        self.signal_length = length_signal
        # size of the final filter and of the flattened layer
        self.final_signal_size = (self.signal_length-self.nb_convs*2*(self.kernel_size//2))
        self.final_nb_filters = self.nb_init_filters*pow(2,self.nb_convs-1)
        self.flattened_size = self.final_signal_size*self.final_nb_filters
        
        # layers
        self.conv1 = nn.Conv1d(28, self.nb_init_filters, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv1d(self.nb_init_filters, self.final_nb_filters, kernel_size=self.kernel_size)
        self.fc1 = nn.Linear(self.flattened_size, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)
        #self.param = Parameter ( Tensor (123 , 456) 
        
    def my_conv(self,x,i):
        tmp_conv = nn.Conv1d(self.nb_init_filters*pow(2,i), self.nb_init_filters*pow(2,i+1),
                         kernel_size=self.kernel_size)
        return tmp_conv(x)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        for i in range(0, self.nb_convs-1):
            x = F.relu(self.my_conv(x,i))
            
        x = F.relu(self.fc1(x.view(-1, self.flattened_size)))
        x = self.fc2(x)
        return x