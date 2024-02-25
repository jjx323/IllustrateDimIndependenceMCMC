#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:56:03 2022

@author: Junxiong Jia
"""

import os
import numpy as np
from scipy.special import logsumexp

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from core.optimizer import GradientDescent, NewtonCG


###################################################################
class MCMCBase:
    def __init__(self, model, reduce_chain=None, save_path=None):
        assert hasattr(model, "prior")
        self.model = model
        self.prior = model.prior
        self.reduce_chain = reduce_chain
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.chain = []
        self.acc_rate = 0.0
        self.index = 0

    def save_local(self):
        ## reduce_chain is an interger number, we will withdraw the reduce_chain number of samples
        ## to save the memory; if the path is specified, we save the chain.
        if self.reduce_chain is not None:
            if self.save_path is not None:
                if np.int64(len(self.chain)) == np.int64(self.reduce_chain):
                    np.save(self.save_path / ('sample_' + str(np.int64(self.index))), self.chain)
                    # tmp = self.chain[-1].copy()
                    # self.chain = [tmp]
                    self.chain = []
                    self.index += 1
            elif self.save_path is None:
                if np.int64(len(self.chain)) == np.int64(self.reduce_chain):
                    # tmp = self.chain[-1].copy()
                    # self.chain = [tmp]
                    self.chain = []
                    self.index += 1  ## here the index has not meaning

    def save_all(self):
        if self.save_path is not None:
            if self.reduce_chain is None:
                np.save(self.save_path / ('samples_all', self.chain))
        # else:
        #     print('\033[1;31m', end='')
        #     print("Not specify the save_path!")
        #     print('\033[0m', end='')

    def sampling(self, len_chain, callback=None, u0=None, index=None, **kwargs):
        raise NotImplementedError


class VanillaMCMC(MCMCBase):
    '''
    Ref: S. L. Cotter, G. O. Roberts, A. M. Stuart and D. White,
    MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science, 2013.
    See Subsection 4.2 of the above article.
    '''
    def __init__(self, model, beta, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs
        assert hasattr(model.prior, "eval_CM_inner")
        self.prior = model.prior

        self.beta = beta
        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)

    def rho(self, x_info, y_info):
        val = 0.5*self.prior.eval_CM_inner(x_info[0])
        # print(val)
        return x_info[1] + val

    def proposal(self, x_info):
        # coef = np.sqrt(2*self.dt)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return ans1 + self.beta * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_info = [x, self.loss(x)]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_info = [y, self.loss(y)]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(tem_acc)
            if np.random.uniform() < tem_acc:
                x_info = y_info
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()

class pCN(MCMCBase):
    '''
    M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems,
    Hankbook of Uncertainty Quantification, 2017 [Sections 5.1 and 5.2]
    '''
    def __init__(self, model, beta, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.dim = model.num_dofs

        tmp = np.sqrt(1 - beta**2)
        self.dt = (2 - 2*tmp)/(1 + tmp)
        self.beta = beta

    def rho(self, x_info, y_info):
        return x_info[1]

    def proposal(self, x_info):
        # dt = self.dt
        coef1 = np.sqrt(1-self.beta*self.beta)
        ans1 = x_info[0]
        ans2 = self.prior.generate_sample()
        return coef1 * ans1 + self.beta * ans2

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        x_loss = self.loss(x)
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            y_loss = self.loss(y)
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(self.rho(x_info, y_info), self.rho(y_info, x_info), y_loss, x_loss)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()


class pCNL(MCMCBase):
    '''
    Ref: S. L. Cotter, G. O. Roberts, A. M. Stuart and D. White,
    MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science, 2013.

    Remark: The current program may not work well, which may contains some unknown buggs.
    '''
    def __init__(self, model, beta, grad_smooth_degree=1e-2, reduce_chain=None, save_path=None):
        super().__init__(model, reduce_chain, save_path)
        self.model.smoother.set_degree(grad_smooth_degree)  ## prepare for evaluating optimization
        self.prior = model.prior
        self.M, self.K = model.prior.M, model.prior.K
        self.loss = model.loss_res
        self.grad = model.eval_grad_res
        self.dim = model.num_dofs

        tmp = np.sqrt(1 - beta ** 2)
        self.dt = (2 - 2 * tmp) / (1 + tmp)

    def rho(self, x_info, y_info):
        x, grad_x, loss_x = x_info
        y = y_info[0]
        coef1, coef2, coef3, coef4 = 1, 1/2, self.dt/4, self.dt/4
        ans1 = loss_x
        ans2 = grad_x@self.M@(y-x)
        ans3 = grad_x@self.M@(x+y)
        ans4 = self.prior.eval_C(grad_x)@(self.M@grad_x)
        return coef1*ans1+coef2*ans2+coef3*ans3+coef4*ans4

    def proposal(self, x_info):
        dt = self.dt
        x, grad_x, loss_x = x_info

        coef1 = (2-dt)/(2+dt)
        coef2 = -2*dt/(2+dt)
        coef3 = np.sqrt(8*dt)/(2+dt)
        ans1 = x
        ans2 = self.prior.eval_C(grad_x)
        ans3 = self.prior.generate_sample()
        return coef1*ans1+coef2*ans2+coef3*ans3

    def sampling(self, len_chain=1e5, callback=None, u0=None, index=None):
        if u0 is None:
            x = self.prior.generate_sample()
        else:
            x = u0.copy()
        assert x.shape[0] == self.dim
        assert x.ndim == 1
        data_info = np.finfo(x[0].dtype)
        max_value = data_info.max
        min_value = data_info.min
        self.chain = [x]
        acc_num = 0
        ## index is the block of the chain saved on the hard disk
        if index == None:
            self.index = 0
        else:
            self.index = index
        grad_x = self.grad(x)
        grad_x = self.model.smoother.smoothing(grad_x)
        x_loss = self.loss()
        if np.isnan(x_loss):
            x_loss = max_value
        x_info = [x, grad_x, x_loss]
        i = 1
        while i <= len_chain:
            y = self.proposal(x_info)
            grad_y = self.grad(y)
            grad_y = self.model.smoother.smoothing(grad_y)
            y_loss = self.loss()
            if np.isnan(y_loss):
                y_loss = max_value
            y_info = [y, grad_y, y_loss]
            tem = self.rho(x_info, y_info) - self.rho(y_info, x_info)
            tem_acc = np.exp(min(0, tem))
            # print(tem_acc)
            if np.random.uniform() < tem_acc:
                x_info = [y_info[0], y_info[1], y_info[2]]
                acc_num += 1
            self.acc_rate = acc_num / i
            self.chain.append(x_info[0])
            i += 1
            self.save_local()
            if callback is not None:
                callback([x_info, i, self.acc_rate])

        self.save_all()


