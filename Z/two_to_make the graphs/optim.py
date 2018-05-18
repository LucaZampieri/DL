""""
File that contains the optimizers for the project two of DeepLearning.
"""

import torch
from torch import Tensor
import math


class Optimizer(object):
    def step(self, model, lr, *input):
        raise NotImplementedError


class SGDOptimizer(Optimizer):
    """implementation of the Stochastic Gradient Descent optimizer"""
    def __init__(self, model, lr, L1_regularisation = 0):
        super().__init__()
        self.model = model
        self.params = self.model.params
        self.lr = lr # learning rate
        self.name = 'SGD'
        self.lambda_ = L1_regularisation # for regularisation

    def zero_grad(self):
        """ set the model gradients to zero"""
        self.model.zero_grad()

    def step(self):
        """Do the optimization step"""
        for param in self.model.params:
            param[0][0].add_(- self.lr * param[1][0])
            param[0][1].add_(- self.lr * param[1][1])
            # .add_(-param[0][0].sign()*param[0][0].abs().clamp(max = self.lambda_ )) # add this for regularisation


class SGDmomOptimizer(Optimizer):
    """implementation of SGD with momentum"""
    def __init__(self, model, lr, gamma=0.2):
        super().__init__()
        self.model = model
        self.lr = lr # learning rate
        self.gamma = gamma # momentum
        self.name = 'SGDmom'
        self._initialize_u()

    def zero_grad(self):
        """ set the model gradients to zero"""
        self.model.zero_grad()

    def _initialize_u(self):
        self.u = []
        for param in self.model.params:
            self.u.append(torch.zeros_like(param[0][0]))
            self.u.append(torch.zeros_like(param[0][1]))

    def step(self):
        """Do the optimization step"""
        for i,param in enumerate(self.model.params):
            for j in range(len(param)): # iterate for biases and weights
                param[0][j].sub_(self.lr * (self.gamma * self.u[2*i+j] +  param[1][j]))
                self.u[2*i+j] = param[0][j]



class AdamOptimizer(Optimizer):
    """implementation of the Adam optimizer. The default values for b1 and b2
    are values often used in the literature"""
    def __init__(self, model, lr, b1=0.9, b2=0.999):
        super().__init__()
        self.model = model
        self.params = self.model.params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.name = 'Adam'
        self._initialize_m()
        self._initialize_v()

    def zero_grad(self):
        """ set the model gradients to zero"""
        self.model.zero_grad()

    def _initialize_m(self):
        self.m = []
        for param in self.model.params:
            self.m.append(torch.zeros_like(param[0][0]))
            self.m.append(torch.zeros_like(param[0][1]))

    def _initialize_v(self):
        self.v = []
        for param in self.model.params:
            self.v.append(torch.zeros_like(param[0][0]))
            self.v.append(torch.zeros_like(param[0][1]))

    def step(self):
        """Do the optimization step"""
        for i,param in enumerate(self.model.params):
            for j in range(len(param)): # iterate for biases and weights
                self.m[2*i+j] = self.b1*self.m[2*i+j] + (1-self.b1)*param[1][j]
                self.v[2*i+j] = self.b2*self.v[2*i+j] + (1-self.b2)*param[1][j]**2

                mhat = (1/(1-self.b1))*self.m[2*i+j]
                vhat = (1/(1-self.b2))*self.v[2*i+j]

                div = torch.sqrt(vhat)+torch.Tensor(vhat.size()).normal_(mean = 0, std = 1e-6)

                param[0][j].sub_(self.lr * mhat/div)


class BFGSOptimizer(Optimizer):
    """implement the BFGS optimizer for weights and SGD for biases"""
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.params = self.model.params
        self.step_size = lr
        self.name = 'BFGS'

        self.grads = []
        self.hessians_inv = []
        self.old_grads = []
        self.directions = []
        self.w_diff = []
        self.grad_diff = []
        self.w = []
        self.w_old = []

        self._initialize_parameters()

    def zero_grad(self):
        """ set the model gradients to zero"""
        self.model.zero_grad()

    def _initialize_parameters(self):
        """initialize the parameters of the optimizer"""
        for param in self.params:
            my_ones = torch.ones(param[1][0].size())
            my_zeros = torch.zeros(param[1][0].size())
            self.grads.append(my_zeros.clone())
            self.hessians_inv.append(my_ones.clone())
            self.old_grads.append(my_zeros.clone())
            self.w_diff.append(my_zeros.clone())
            self.w_old.append(my_zeros.clone())
            self.w.append(my_zeros.clone())
            self.grad_diff.append(my_zeros.clone())
            self.directions.append(param[1][0].clone())

    def _compute_direction(self):
        for i, param in enumerate(self.params):
            self.directions[i] = self.hessians_inv[i]*self.grads[i]

    def _quasi_update(self):
        """update the inverse of the hessian"""
        for i, inv_hess in enumerate(self.hessians_inv):
            rho = 1.0/(self.grad_diff[i]*self.w_diff[i]+0.00001)
            tmp1 = 1.0 - rho * self.grad_diff[i] * self.w_diff[i]
            tmp2 = rho * self.w_diff[i] * self.w_diff[i]
            self.hessians_inv[i] = tmp1 * inv_hess * tmp1 + tmp2  # for BFGS update
            #self.hessians_inv[i] = inv_hess # for "SGD" update

    def _compute_step_size(self):
        """update the step size. We could for example use line-search algorythms.
        or leave the original learning rate as step size"""
        # in this version we keep the original learning rate
        #self.step_size = self.step_size
        if self.step_size == 0: # check that the step_size is not 0
            print('optimizer step size is 0')

    def _get_new_parameters(self):
        """update parameters needed to run the step"""
        for i,param in enumerate(self.params):
            self.grads[i] = param[1][0].clone()
            self.w[i] = param[0][0].clone()

    def _update_parameters(self):
        """update parameters needed for the next step"""
        for i, param in enumerate(self.params):
            self.w_diff[i] = self.w[i] - self.w_old[i]
            self.grad_diff[i] = self.grads[i] - self.old_grads[i]
            self.w_old[i] = self.w[i].clone()
            self.old_grads[i] = self.grads[i].clone()

    def step(self):
        """Do the optimization step"""
        self._get_new_parameters() # update the dradients and weights stored in the optimizer
        self._compute_direction() # update direction
        self._compute_step_size() # update step size

        # do the step
        for i,param in enumerate(self.model.params):
            param[0][0].add_(- self.step_size * self.directions[i])
            param[0][1].add_(- self.step_size * param[1][1])

        # update optimizer parameters and do quasi-update
        self._update_parameters()
        self._quasi_update()




# End of file
