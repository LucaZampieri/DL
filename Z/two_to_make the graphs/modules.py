"""
Contains the implementations of various modules
"""

import torch
from torch import Tensor
import math


class Module ( object ) :
    def forward ( self , * input ) :
        raise NotImplementedError

    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError

    def param ( self ) :
        return []

    def zero_grad ( self ) :
        pass

    def reset_params( self ) :
        pass

class Linear(Module):
    """ implementation of a standard layer"""
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = torch.Tensor(out_features, in_features).normal_(mean=0, std = 1)
        self.bias = torch.Tensor(out_features).uniform_(0,0)

        self.dl_dw = torch.Tensor(out_features, in_features)
        self.dl_db = torch.Tensor(out_features)
        #self.reset_parameters()
        #self.reset_parameters_xavier() # if we prefer to use xavier
        self.zero_grad()

        self.params = [(self.weights, self.bias),(self.dl_dw, self.dl_db)]

    def forward(self,x):
        self.x = x
        if(x.size()[1]!=self.in_features):
            raise TypeError('Size of x should correspond to size of linear module')
        return torch.mm(x,self.weights.t()) + self.bias.expand(x.size()[0], self.out_features)

    def backward(self, d_dx):
        self.dl_db = torch.mean(d_dx,0)
        self.dl_dw = torch.mm(d_dx.t(), self.x)
        return torch.mm(d_dx,self.weights)

    def param(self):
        self.params = [(self.weights, self.bias),(self.dl_dw, self.dl_db)]
        return self.params

    def zero_grad(self):
        self.dl_db.zero_()
        self.dl_dw.zero_()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1)) / math.sqrt(3) #
        self.weights.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        size =self.weights.size()
        stdv = math.sqrt(2.0/(size[0] + size[1]))
        self.weights.normal_(-stdv, stdv)
        self.bias.uniform_(0,0)


class Sequential(Module):
    """implementation of the sequential modules.
    """
    def __init__(self, modules, loss):
        """:input: list of modules, loss module"""
        super().__init__()

        self.modules = modules
        self.loss = loss
        self.params = []
        self.param()

    def add(self, new_modules):
        """function to add new modules"""
        self.modules.append(new_modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self):
        d_dx = self.loss.backward()
        for module in reversed(self.modules):
            d_dx = module.backward(d_dx)
        self.param()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def param(self):
        self.params = []
        for module in self.modules:
            if(module.params):
                self.params.append(module.param())

    def reset_params(self):
        for module in self.modules:
            module.reset_params()


class LeakyReLU(Module):
    def __init__(self, alpha = 0.1):
        super().__init__()

        self.params = []
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return x*((x>0).type(torch.FloatTensor)+self.alpha*(x<0).type(torch.FloatTensor))

    def backward(self, d_dx):
        return d_dx*((self.x>0).type(torch.FloatTensor)+self.alpha*(self.x<0).type(torch.FloatTensor))

    def param(self):
        return self.params

class ReLU(Module):
    def __init__(self):
        super().__init__()

        self.params = []

    def forward(self, x):
        self.x = x
        return x*(x>0).type(torch.FloatTensor)

    def backward(self, d_dx):
        return d_dx*(self.x>0).type(torch.FloatTensor)

    def param(self):
        return self.params


class Tanh(Module):
    """Rectified Tanh, since we predict betwee 0 and 1"""
    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, x):
        self.x = x
        return 0.5*(1+x.tanh())

    def backward(self, d_dx):
        return 0.5*d_dx*(1-torch.tanh(self.x)**2)

    def param(self):
        return self.params

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, x):
        self.x = x
        return torch.sigmoid(x)

    def backward(self, d_dx):
        return d_dx * (torch.sigmoid(self.x*(1-torch.sigmoid(self.x))))

    def param(self):
        return self.params


class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, y, t):
        self.y = y
        self.t = t
        return torch.dist(y,t,p=2) # L2 norm

    def backward(self):
        return 2*(self.y-self.t)

    def param(self):
        return self.params



class LossBCE(Module):
    def __init__(self):
        super().__init__()
        self.params = []

    def forward(self, y_pred, t):
        self.t = t
        self.y_pred = y_pred
        true_target, true_index = torch.max(t.view(1,-1),1)
        y_true = y_pred[true_index]
        y_false = y_pred[1-true_index]
        self.true_target = true_target

        self.y_true = y_true
        self.y_false = y_false
        # old way to do it
        L_true = -torch.log(y_true)
        L_false = -torch.log(1-y_false)
        L = torch.cat((L_true,L_false),0)
        #L = - (true_target*torch.log(y_pred) + (1-true_target)*torch.log(1-y_pred))
        return torch.mean(L)

    def backward(self):
        # old way
        dL_true = -1/self.y_true
        dL_false = -1/self.y_false
        dL = torch.cat((dL_true,dL_false),0)
        #dL = -(self.true_target*1/self.y_pred + (1-self.true_target)*1/(1-self.y_pred))
        return torch.mean(dL)

    def param(self):
        return self.params
