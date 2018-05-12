import torch
import math

from torch import Tensor
from torch import optim
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn





"""def create_model_1():
    return Net(nb_hidden=100, nb_init_filters = 16, nb_convs=2, kernel_size=5, length_signal = SIGNAL_LENGTH )"""
class DilatedNet(nn.Module):

    def __init__(self, hidden = 100, num_classes=2, nl='relu',iterations = 0,length_signal = 500 ):
        
        super().__init__()
        #self.dropping_prob = 0.5
        if nl == 'leaky': f_non_linearity = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        elif nl == 'tanh' : f_non_linearity =  nn.Tanh()
        elif nl == 'relu': f_non_linearity = nn.ReLU(inplace=True)
            
        filter_size = 3
        dilation = 1
        my_dropout = nn.Dropout(p=0.8)
        ################### layers ################
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        f_layers = []
        
        # my dil_10 layer
        loss_of_signal_dil10 = 0
        if dil_10 == True:
            ker_size_dil_10 = 50
            stride_first = 1
            f_layers.append(nn.Conv1d(28, 28, kernel_size=ker_size_dil_10,stride=1, dilation=10))
            apparent_dil10_size = (ker_size_dil_10-1)*dilation+1
            loss_of_signal_dil10 = 450 #(apparent_dil10_size//2)*2
                
        if stride10 == True:
            ker_size_stride = 1
            stride_first = 10
            f_layers.append(nn.Conv1d(28, 28, kernel_size=ker_size_stride,stride=stride_first, dilation=1))
            if ker_size_dil_10==1:
                apparent_dil10_size = 0
                loss_of_signal_dil10 = 450
            else :
                apparent_dil10_size = (ker_size_stride-1)*dilation+1
                loss_of_signal_dil10 = (apparent_dil10_size//2)*2

        
        
        f_layers.append(nn.Conv1d(28, 8, kernel_size=filter_size, dilation=dilation)) # 7 
        f_layers.append(f_non_linearity)
        
        f_layers.append(nn.Conv1d(8, 8, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(8, affine=True))
        f_layers.append(my_dropout)
        #f_layers.append(nn.MaxPool1d(kernel_size=2))
        
        
        for i in range (iterations):
            f_layers.append(nn.Conv1d(4, 4, kernel_size=3, dilation=dilation))
            f_layers.append(f_non_linearity)
            f_layers.append(my_dropout)
        
        f_layers.append(nn.Conv1d(8, 16, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(16, affine=True))
        f_layers.append(my_dropout)
        
        self.features = nn.Sequential(*f_layers)
        
        # signal size
        apparent_filter_size = (filter_size-1)*dilation+1 # because of dilation
        self.final_filters = length_signal - 3*(apparent_filter_size//2)*2 - loss_of_signal_dil10 #-iterations*2
        self.final_length = 16
        
        c_layers = [] 
        c_layers.append(nn.Linear(self.final_filters*self.final_length, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        #c_layers.append(nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=False))
        #c_layers.append(nn.Dropout(inplace=True))
        """c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(my_dropout)
        c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))"""
        c_layers.append(nn.Linear(hidden, num_classes))
        #c_layers.append(nn.Softmax(dim=1))
        
        self.classifier = nn.Sequential(* c_layers)
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.final_filters*self.final_length)
        x = self.classifier(x)
        return x

"""def create_model_1():
    return Net(nb_hidden=100, nb_init_filters = 16, nb_convs=2, kernel_size=5, length_signal = SIGNAL_LENGTH )"""
class DilatedNet2(nn.Module):

    def __init__(self, hidden = 200, num_classes=2, nl='relu',iterations = 0,length_signal = 500 ):
        
        super().__init__()
        #self.dropping_prob = 0.5
        if nl == 'leaky': f_non_linearity = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        elif nl == 'tanh' : f_non_linearity =  nn.Tanh()
        elif nl == 'relu': f_non_linearity = nn.ReLU(inplace=True)
            
        # 
        filters_size = [3,3,3]
        self.dilations    = [8,2,2]
        
        my_dropout = nn.Dropout(p=0.7)
        ################### layers ################
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # torch.nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=False)
        f_layers = []
        

        #### Conv1
        f_layers.append(nn.Conv1d(28, 8, kernel_size=filters_size[0], dilation=self.dilations[0])) 
        f_layers.append(f_non_linearity)
        #f_layers.append(nn.BatchNorm1d(8, affine=True))
        
        #### Conv2
        f_layers.append(nn.Conv1d(8, 8, kernel_size=filters_size[1], dilation=self.dilations[1]))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(8, affine=True))
        f_layers.append(my_dropout)
  
        #### Conv3
        f_layers.append(nn.Conv1d(8, 16, kernel_size=filters_size[2], dilation=self.dilations[2]))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(16, affine=True))
        f_layers.append(my_dropout)
        
        #### add all the convs to self.features
        self.features = nn.Sequential(*f_layers)
        
        #### set the signal size tha will be used for the fcl
        length_signal -= (filters_size[0]-1)*self.dilations[0]
        length_signal -= (filters_size[1]-1)*self.dilations[1]
        length_signal -= (filters_size[2]-1)*self.dilations[2]
        
        self.final_filters = length_signal 
        self.final_length = 16
        
        ####  Linear layers:
        c_layers = [] 
        c_layers.append(nn.Linear(self.final_filters*self.final_length, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(nn.Dropout(0.5))
        
        """c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(nn.Dropout(0.5))"""
        
        c_layers.append(nn.Linear(hidden, num_classes))
        
        self.classifier = nn.Sequential(* c_layers)
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.final_filters*self.final_length)
        x = self.classifier(x)
        return x

class DilatedNet3(nn.Module):

    def __init__(self, hidden = 300, num_classes=2, nl='relu',iterations = 0,length_signal = 500 ):
        
        super().__init__()
        #self.dropping_prob = 0.5
        if nl == 'leaky': f_non_linearity = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        elif nl == 'tanh' : f_non_linearity =  nn.Tanh()
        elif nl == 'relu': f_non_linearity = nn.ReLU(inplace=True)
            
        filter_size = 3
        dilation = 1
        my_dropout = nn.Dropout(p=0.3)
        ################### layers ################
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        f_layers = []
        #self.init
        self.DA_layer = nn.Conv1d(28, 28, kernel_size=1,stride=10, dilation=1)
        self.test = nn.Conv1d(28, 10, kernel_size=3,stride=10, dilation=1)
        
        f_layers.append(nn.Conv1d(28, 8, kernel_size=filter_size, dilation=dilation)) # 7 
        f_layers.append(f_non_linearity)
        
        f_layers.append(nn.Conv1d(8, 8, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(8, affine=True))
        f_layers.append(my_dropout)
        #f_layers.append(nn.MaxPool1d(kernel_size=2))
        
        
        for i in range (iterations):
            f_layers.append(nn.Conv1d(4, 4, kernel_size=3, dilation=dilation))
            f_layers.append(f_non_linearity)
            f_layers.append(my_dropout)
        
        f_layers.append(nn.Conv1d(8, 16, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(16, affine=True))
        f_layers.append(my_dropout)
        
        self.features = nn.Sequential(*f_layers)
        
        # signal size
        apparent_filter_size = (filter_size-1)*dilation+1 # because of dilation
        first_kernel_loss = 450
        self.final_filters = length_signal - 3*(apparent_filter_size//2)*2 - first_kernel_loss #-iterations*2
        self.final_length = 16
        
        c_layers = [] 
        c_layers.append(nn.Linear(self.final_filters*self.final_length, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(nn.Dropout(0.5))
        #c_layers.append(nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=False))
        #c_layers.append(nn.Dropout(inplace=True))
        #c_layers.append(nn.Linear(hidden, hidden))
        #c_layers.append(nn.ReLU(inplace=True))
        #c_layers.append(nn.Dropout(0.5))
        """c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))"""
        c_layers.append(nn.Linear(hidden, num_classes))
        #c_layers.append(nn.Softmax(dim=1))
        
        self.classifier = nn.Sequential(* c_layers)
        
        
    def forward(self, x):
        x0 = self.DA_layer(x[:,:,0:])
        for i in range(1,10):
            x0 = x0.add(self.DA_layer(x[:,:,i:]))
        x = self.features(x0)
        x = x.view(-1, self.final_filters*self.final_length)
        x = self.classifier(x)
        return x

"""def create_model_1():
    return Net(nb_hidden=100, nb_init_filters = 16, nb_convs=2, kernel_size=5, length_signal = SIGNAL_LENGTH )"""
class DilatedNet4(nn.Module):

    def __init__(self, hidden = 300, num_classes=2, nl='relu',iterations = 0,length_signal = 500 ):
        
        super().__init__()
        #self.dropping_prob = 0.5
        if nl == 'leaky': f_non_linearity = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        elif nl == 'tanh' : f_non_linearity =  nn.Tanh()
        elif nl == 'relu': f_non_linearity = nn.ReLU(inplace=True)
            
        filter_size = 3
        dilation = 2
        my_dropout = nn.Dropout(p=0.6)
        ################### layers ################
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        f_layers = []
        
        #torch.nn.ConstantPad1d(padding, value)
        f_layers.append(nn.Conv1d(28, 28, kernel_size=201, dilation=1, padding=100))
        
        f_layers.append(nn.Conv1d(28, 8, kernel_size=3, dilation=10)) # 7 
        f_layers.append(f_non_linearity)
        
        f_layers.append(nn.Conv1d(8, 8, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(8, affine=True))
        f_layers.append(my_dropout)
        #f_layers.append(nn.MaxPool1d(kernel_size=2))
        
        
        for i in range (iterations):
            f_layers.append(nn.Conv1d(4, 4, kernel_size=3, dilation=dilation))
            f_layers.append(f_non_linearity)
            f_layers.append(my_dropout)
        
        f_layers.append(nn.Conv1d(8, 16, kernel_size=filter_size, dilation=dilation))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(16, affine=True))
        f_layers.append(my_dropout)
        
        self.features = nn.Sequential(*f_layers)
        
        # signal size
        apparent_filter_size = (filter_size-1)*dilation+1 # because of dilation
        first_kernel_loss = 20
        self.final_filters = length_signal - 2*(apparent_filter_size//2)*2 - 20 #-iterations*2
        self.final_length = 16
        
        c_layers = [] 
        c_layers.append(nn.Linear(self.final_filters*self.final_length, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(nn.Dropout(0.5))
        #c_layers.append(nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=False))
        #c_layers.append(nn.Dropout(inplace=True))
        c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(nn.Dropout(0.5))
        """c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))"""
        c_layers.append(nn.Linear(hidden, num_classes))
        #c_layers.append(nn.Softmax(dim=1))
        
        self.classifier = nn.Sequential(* c_layers)
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.final_filters*self.final_length)
        x = self.classifier(x)
        return x

class Net_seq(nn.Module):

    def __init__(self, hidden = 100, num_classes=2, nl='relu',iterations = 0,length_signal = 500 ):
        
        super().__init__()
        #self.dropping_prob = 0.5
        if nl == 'leaky': f_non_linearity = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        elif nl == 'tanh' : f_non_linearity =  nn.Tanh()
        elif nl == 'relu': f_non_linearity = nn.ReLU(inplace=True)
            
        filter_size = 3
        my_dropout = nn.Dropout(p=0.8)
        ################### layers ################
        
        f_layers = []
        f_layers.append(nn.Conv1d(28, 8, kernel_size=filter_size)) # 7 
        f_layers.append(f_non_linearity)
        
        f_layers.append(nn.Conv1d(8, 8, kernel_size=filter_size))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(8, affine=True))
        f_layers.append(my_dropout)
        #f_layers.append(nn.MaxPool1d(kernel_size=2))
        
        
        for i in range (iterations):
            f_layers.append(nn.Conv1d(4, 4, kernel_size=3))
            f_layers.append(f_non_linearity)
            f_layers.append(my_dropout)
        
        f_layers.append(nn.Conv1d(8, 16, kernel_size=filter_size))
        f_layers.append(f_non_linearity)
        f_layers.append(nn.BatchNorm1d(16, affine=True))
        f_layers.append(my_dropout)
        
        self.features = nn.Sequential(*f_layers)
        
        self.final_filters = length_signal - 3*(filter_size//2)*2 #-iterations*2
        self.final_length = 16
        
        c_layers = [] 
        c_layers.append(nn.Linear(self.final_filters*self.final_length, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        #c_layers.append(nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=False))
        #c_layers.append(nn.Dropout(inplace=True))
        """c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))
        c_layers.append(my_dropout)
        c_layers.append(nn.Linear(hidden, hidden))
        c_layers.append(nn.ReLU(inplace=True))"""
        c_layers.append(nn.Linear(hidden, num_classes))
        #c_layers.append(nn.Softmax(dim=1))
        
        self.classifier = nn.Sequential(* c_layers)
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.final_filters*self.final_length)
        x = self.classifier(x)
        return x
        

# 14% with just 2 conv layers 26,16 filter 5 ;; 16,32, filter 5, 2 fcl with hidden = 100

