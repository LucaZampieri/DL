"""
This file contains the differents neural networks implemented
"""

import torch
from torch import nn
from torch.nn import functional as F



class MyNet(nn.Module):
    """Our optimal network, best accuracy but longer training """
    def __init__(self):
        super(MyNet, self).__init__()

        # Layer 1
        self.conv1a = nn.Conv2d(1, 8, (1, 5), dilation=2) # Apply convolution on each channels
        self.batchnorm1a = nn.BatchNorm2d(8, False) # Normalize
        # Layer 2
        self.conv1b = nn.Conv2d(8, 16, (1, 5)) # Apply convolution on each channels
        self.batchnorm1b = nn.BatchNorm2d(16, False) # Normalize
        # Layer 3
        self.conv2 = nn.Conv2d(16,32,(28, 1)) # Apply one convolution on each timestamp
        self.batchnorm2 = nn.BatchNorm2d(32, False) # Normalize

        # Fully Connected Layers
        self.fc1 = nn.Linear(1216,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,1)

    def forward(self, x):
        x = self.conv1a(x)
        x = self.batchnorm1a(x)
        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = self.conv2(x)
        x = F.elu(self.batchnorm2(x))
        x = F.dropout2d(x,0.5)
        s = x.shape
        a = s[1]*s[2]*s[3] #s[1]*s[2]*s[3] = 32*1*42
        x = x.view(-1,a)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.dropout(x,0.5)
        x = F.elu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x


class MyNet2(nn.Module):
    """Very simple network made with two fully connected layers"""
    def __init__(self):
        super(MyNet2, self).__init__()
        self.fc1 = nn.Linear(28*50,128)
        self.fc2 = nn.Linear(128,1)

    def forward(self,x):
        x = x.view(-1,28*50)
        x = F.dropout(x,0.5)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class MyNet3(nn.Module):
    """Simple convolutional net with no dropout between convolutions"""
    def __init__(self):
        super(MyNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (1, 5), dilation=2)
        self.batchnorm1 = nn.BatchNorm2d(4,False)
        self.conv2 = nn.Conv2d(4,4,(28,1))
        self.batchnorm2 = nn.BatchNorm2d(4, False) # Normalize
        self.fc1 = nn.Linear(4*42,64)
        self.fc2 = nn.Linear(64,1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = x.view(-1,4*42)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.6)
        x = F.sigmoid(self.fc2(x))
        return x

class DANet3(nn.Module):
    """Convolutional net with data augmentation done by the first layer. It simulates
    subsampling more or less as done by the data augmentation used by some other models
    The train input and test for this model should be of high resolution i.e. have lenght 500 """
    def __init__(self):
        super().__init__()
        self.last_lenght = 46
        self.conv1 = nn.Conv2d(1, 4, (1, 5),stride=(1,10), dilation=10)
        self.batchnorm0 = nn.BatchNorm2d(1,False)
        self.batchnorm1 = nn.BatchNorm2d(4,False)
        self.conv2 = nn.Conv2d(4,4,(28,1))
        self.batchnorm2 = nn.BatchNorm2d(4, False) # Normalize

        self.fc1 = nn.Linear(4*self.last_lenght,64)
        self.fc2 = nn.Linear(64,1)


    def forward(self,x):
        # subsample the signal and add them together
        x0 = self.conv1(x[:,:,:,0:])
        for i in range(1,10):
            x0 = x0.add(self.conv1(x[:,:,:,i:]))

        x = self.batchnorm1(x0)
        x = F.dropout2d(x,0.5)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.dropout2d(x,0.5)
        x = x.view(-1,4*self.last_lenght)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.6)
        x = F.sigmoid(self.fc2(x))

        return x
