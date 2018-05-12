import dlc_bci as bci
import numpy as np
import torch
import random

from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F

class MyNet(nn.Module):
    def __init__(self, big_data):
        super(MyNet, self).__init__()
        
        # Layer 1
        self.conv1a = nn.Conv2d(1, 8, (1, 5), dilation=2) # Apply convolution on each channels
        self.batchnorm1a = nn.BatchNorm2d(8, affine = False) # Normalize

        self.conv1b = nn.Conv2d(8, 16, (1, 5)) # Apply convolution on each channels
        self.batchnorm1b = nn.BatchNorm2d(16, affine = False) # Normalize
        # Layer 2
        self.conv2 = nn.Conv2d(16,32,(28, 1)) # Apply one convolution on each timestamp
        self.batchnorm2 = nn.BatchNorm2d(32,affine = False) # Normalize
        
        self.dropout2d = torch.nn.Dropout2d(p=0.5, inplace=False)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
             
        # Fully Connected Layers
        self.fc1 = nn.Linear(1216,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,1)
    
    def forward(self, x):

        print("input shape : {}".format(x.size()))
        x = self.conv1a(x)
        print("Shape after self.conv1a(x) : {}".format(x.size()))
        x = self.batchnorm1a(x)
        #print("Shape after self.batchnorm1a(x) : {}".format(x.shape))
        x = self.conv1b(x)
        #print("Shape after self.conv1b(x) : {}".format(x.shape))
        x = self.batchnorm1b(x)
        #print("Shape after self.batchnorm1b(x) : {}".format(x.shape))
        x = self.conv2(x)
        x = F.elu(self.batchnorm2(x))
        #print("Shape after secondConvolution : {}".format(x.shape))
        x = self.dropout2d(x)
        #x.Dropout2d(0.5,inplace=True)
        s = x.size()
        print(s)
        a = s[1]*s[2]*s[3] #s[1]*s[2]*s[3] = 32*1*42
        print(a)
        x = x.view(-1,a) 
        #print("Flatten shape for FC : {}".format(x.shape))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        #print("Output format : {}".format(x.shape))
        
        return x
    
    
class MyNetDA(nn.Module):
    def __init__(self, big_data):
        super().__init__()
        
        # Layer 1
        self.conv1a = nn.Conv2d(1, 8, (1, 5), dilation=20) # Apply convolution on each channels
        self.batchnorm1a = nn.BatchNorm2d(8, affine = False) # Normalize

        self.conv1b = nn.Conv2d(8, 16, (1, 5)) # Apply convolution on each channels
        self.batchnorm1b = nn.BatchNorm2d(16, affine = False) # Normalize
        # Layer 2
        self.conv2 = nn.Conv2d(16,32,(28, 1)) # Apply one convolution on each timestamp
        self.batchnorm2 = nn.BatchNorm2d(32,affine = False) # Normalize
        
        self.dropout2d = torch.nn.Dropout2d(p=0.5, inplace=False)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
             
        # Fully Connected Layers
        self.final_length = 32*416
        self.fc1 = nn.Linear(self.final_length,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,2)
    
    def forward(self, x):

        #print("input shape : {}".format(x.size()))
        x = self.conv1a(x)
        #print("Shape after self.conv1a(x) : {}".format(x.size()))
        x = self.batchnorm1a(x)
        #print("Shape after self.batchnorm1a(x) : {}".format(x.size()))
        x = self.conv1b(x)
        #print("Shape after self.conv1b(x) : {}".format(x.size()))
        x = self.batchnorm1b(x)
        #print("Shape after self.batchnorm1b(x) : {}".format(x.size()))
        x = self.conv2(x)
        x = F.elu(self.batchnorm2(x))
        #print("Shape after secondConvolution : {}".format(x.size()))
        x = self.dropout2d(x)
        #x.Dropout2d(0.5,inplace=True)
        s = x.size()
        #print('x.size before linear',s)
        a = s[1]*s[2]*s[3] #s[1]*s[2]*s[3] = 32*1*42
        
        x = x.view(-1,self.final_length) 
        #print("Flatten shape for FC : {}".format(x.shape))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.elu(self.fc3(x))
        #x = F.sigmoid(self.fc4(x))
        x = self.fc4(x)#.view(-1) 
        #print("Output format : {}".format(x.shape))
        
        return x

class MyNet2(nn.Module):
    def __init__(self, big_data):
        super(MyNet2, self).__init__()
        
        self.fc1 = nn.Linear(28*50,128)
        self.fc2 = nn.Linear(128,1)
        
    def forward(self,x):
        
        #print("input shape : {}".format(x.shape))
        x = x.view(-1,28*50)
        #print("Flatten shape for FC : {}".format(x.shape))
        x = F.dropout(x,0.5)
        x = F.relu(self.fc1(x))
        #print(self.fc2(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

class MyNet3(nn.Module):
    def __init__(self, big_data):
        super(MyNet3, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 4, (1, 5), dilation=2)
        self.batchnorm1 = nn.BatchNorm2d(4,False)
        self.conv2 = nn.Conv2d(4,4,(28,1))
        self.batchnorm2 = nn.BatchNorm2d(4, False) # Normalize
        self.fc1 = nn.Linear(4*26,64)    
        self.fc2 = nn.Linear(64,1)
        
    def forward(self,x):
        
        #print("input shape : {}".format(x.shape))
        x = self.conv1(x)
        #print("Shape after self.conv1(x) : {}".format(x.shape))
        x = self.batchnorm1(x)
        #print("Shape after self.batchnorm1(x) : {}".format(x.shape))
        x = self.conv2(x)
        #print("Shape after self.conv2(x) : {}".format(x.shape))
        x = self.batchnorm2(x)
        #print("Shape after self.batchnorm2(x) : {}".format(x.shape))
        x = x.view(-1,4*42)
        #print("Flatten shape for FC : {}".format(x.shape))
        x = F.relu(self.fc1(x))
        x = F.dropout(x,0.6)
        #print("after fc1 : {}".format(x.shape))      
        x = F.sigmoid(self.fc2(x))
        
        return x