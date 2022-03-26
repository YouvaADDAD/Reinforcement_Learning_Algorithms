from torch.utils.data import DataLoader
import torch.nn as nn
import enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns
from utils import *


class AffineFlow(FlowModule):
    def __init__(self, in_features, dtype = torch.double):
        #in_features : la dimension des données
        #On est en batchs
        super().__init__()
        self.s = nn.Parameter(torch.randn(in_features, dtype = dtype, requires_grad=True))#Broadcast x * s-> (batch, in_features)
        self.t = nn.Parameter(torch.randn(in_features, dtype = dtype, requires_grad=True))#Broadcast x + s-> (batch, in_features)
        
    def f(self, x):
        y = x * torch.exp(self.s) + self.t
        logdet = torch.sum(self.s , dim = -1) #exp est toujours positive et log(exp(x)) = x
        return y, logdet
    
    def invf(self, y):
        x = (y - self.t) * torch.exp(- self.s) #f^{-1}(y)
        logdet = - torch.sum(self.s, dim = -1)
        return x, logdet
    
    def check(self, x):
        return self.invf(self.f(x)[0])[0].allclose(x)

#On hérite de AffineFlow afin d'avoir le checker et réduire le code
#Le broadcast sera toujours fait si (dim1, dim2) (op) (dim2) = (dim1, dim2) (op) (1, dim2)
class ActNorm(AffineFlow):
    def __init__(self, in_features, dtype = torch.double):
        super().__init__(in_features, dtype = dtype)
        self.first_init = False
    
    def invf(self, y):
        if not self.first_init:
            self.s.data.copy_(torch.log(y.std(dim = 0)).data)
            self.t.data.copy_(y.mean(dim = 0).data)
            self.first_init = True
        return super().invf(y)

class AffineCouplingLayer(FlowModule):
    def __init__(self, in_features, hidden_dim = 64):
        #Soit x la valeur actuelle de dimension 2×l
        #2xl = in_features
        #l = in_features/2
        super().__init__()
        assert in_features%2 == 0, 'Must be divisible by 2'
        self.s = MLP(in_features // 2, in_features // 2, hidden_dim)
        self.t = MLP(in_features // 2, in_features // 2, hidden_dim)
    
    def f(self, x):
        #x : (batch, in_features)
        assert x.shape[1]%2 ==0, 'Must be divisible by 2' 
        x_1, x_2 = torch.chunk(x, 2, dim = 1)
        s = self.s(x_1)
        t = self.t(x_1)
        y_1 = x_1
        y_2 = x_2 * torch.exp(s) + t
        y = torch.cat((y_1, y_2), dim = 1)
        logdet = torch.sum(s , dim = 1)
        return y, logdet
    
    def invf(self, y):
        assert y.shape[1]%2 ==0, 'Must be divisible by 2'
        y_1, y_2 = torch.chunk(y, 2, dim = 1)
        x_1 = y_1
        s = self.s(y_1)
        t = self.t(y_1)
        x_2 = (y_2 - t) * torch.exp(- s)
        x = torch.cat((x_1, x_2), dim = 1)
        logdet =  - torch.sum(s, dim = 1)
        return x, logdet

class Convolution1x1(FlowModule):
    def __init__(self, in_features, dtype = torch.double):
        super().__init__()
        self.in_features = in_features
        W = torch.nn.init.orthogonal_(torch.randn(in_features, in_features, dtype = dtype))
        W_LU, pivots = W.lu()
        W_P, W_L, W_U = torch.lu_unpack(W_LU, pivots)
        self.P = W_P #Matrice Permutation rien a apprendre
        self.L = nn.Parameter(W_L, requires_grad=True)
        self.U = nn.Parameter(W_U, requires_grad=True)
        ###################################################################################
        S = torch.diag(W_U)
        self.sign = torch.sign(S)
        self.logS = nn.Parameter(torch.log(torch.abs(S)), requires_grad=True) #Pour plus de stabilité

    def f(self, x):
        L = torch.tril(self.L, diagonal = -1) + torch.eye(self.in_features)
        U = torch.triu(self.U, diagonal = 1) + torch.diag(self.sign * torch.exp(self.logS))   
        W = self.P @ L @ U
        y = x @ W
        logdet = torch.sum(self.logS)
        return y, logdet
    
    
    def invf(self, y):
        L = torch.tril(self.L, diagonal = -1) + torch.eye(self.in_features)
        U = torch.triu(self.U, diagonal = 1) + torch.diag(self.sign * torch.exp(self.logS))   
        W = self.P @ L @ U
        x = y @ torch.inverse(W)
        logdet = - torch.sum(self.logS)
        return x, logdet



    

