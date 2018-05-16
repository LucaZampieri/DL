""""
File that contains the optimizers for the project two of DeepLearning.
"""

import torch
from torch import Tensor
import math


class SGDOptimizer():
    """implementation of the Stochastic Gradient Descent optimizer"""
    def __init__(self, Sequential, lr):
        super().__init__()

        self.Sequential = Sequential
        self.lr = lr # learning rate

    def step(self):
        for param in self.Sequential.params:
            param[0][0].add_(- self.lr * param[1][0])
            param[0][1].add_(- self.lr * param[1][1])

        #self.Sequential.zero_grad()




class BFGSOptimizer():
    """implement the BFGS optimizer for weights and SGD for biases"""
    def __init__(self, Sequential, lr):
        super().__init__()
        self.Sequential = Sequential
        self.params = self.Sequential.params # 1 per layer
        self.step_size = lr

        self.grads = []
        self.hessians_inv = []
        self.old_grads = []
        self.directions = []
        self.w_diff = []
        self.grad_diff = []
        self.w = []
        self.w_old = []

        self._initialize_parameters()

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
        for i, inv_hess in enumerate(self.hessians_inv):
            rho = 1.0/(self.grad_diff[i]*self.w_diff[i]+0.00001)
            tmp1 = 1.0 - rho * self.grad_diff[i] * self.w_diff[i]
            tmp2 = rho * self.w_diff[i] * self.w_diff[i]
            self.hessians_inv[i] = tmp1 * inv_hess * tmp1 + tmp2  # for BFGS update
            #self.hessians_inv[i] = inv_hess # for "SGD"

    def _compute_step_size(self):
        #self.step_size = self.step_size
        if self.step_size == 0:
            print('optimizer step size is 0')

    def _get_new_parameters(self):
        for i,param in enumerate(self.params):
            self.grads[i] = param[1][0].clone()
            self.w[i] = param[0][0].clone()

    def _update_parameters(self):
        for i, param in enumerate(self.params):
            self.w_diff[i] = self.w[i] - self.w_old[i]
            self.grad_diff[i] = self.grads[i] - self.old_grads[i]
            self.w_old[i] = self.w[i].clone()
            self.old_grads[i] = self.grads[i].clone()

    def step(self):
        self._get_new_parameters()
        self._compute_direction() # update direction
        self._compute_step_size() # update step size

        for i,param in enumerate(self.Sequential.params):
            param[0][0].add_(- self.step_size * self.directions[i])
            param[0][1].add_(- self.step_size * param[1][1])

        # update optimizer parameters and do quasi-update
        self._update_parameters()
        self._quasi_update()

        #self.Sequential.zero_grad()
