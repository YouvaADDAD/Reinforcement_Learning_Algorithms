from Flow import *
from torch.utils.data import DataLoader
import torch.nn as nn
import enum
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns
from torch.autograd import grad
import scipy.stats as stats
from sklearn import datasets
from utils import *

if __name__ =="__main__":

    dim = 2
    dtype = torch.double
    hidden_dim = 32
    modules = []
    L = 8
    for _ in range(L):
        modules.append(ActNorm(dim, dtype = dtype))
        modules.append(Convolution1x1(dim, dtype = dtype))
        modules.append(AffineCouplingLayer(dim).to(dtype))
    #################################################################################################################
    batchsize = 128
    lr = 1e-5
    wd = 1e-2
    nb_epochs = 50000
    mu = torch.zeros(dim, dtype = dtype)
    s = torch.ones(dim, dtype = dtype)
    prior = torch.distributions.independent.Independent(torch.distributions.normal.Normal(mu, s),1)
    model = FlowModel(prior, *modules).to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
    #################################################################################################################
    for i in range(nb_epochs):
            x, _ = datasets.make_circles(n_samples = batchsize, factor=0.5, noise=0.05, random_state=0)
            x = torch.as_tensor(x).to(dtype)
            z_0_logprob, zs, logdet = model.invf(x)
            logprob = z_0_logprob + logdet
            negative_log_likelihood = - torch.sum(logprob)
            model.zero_grad()
            negative_log_likelihood.backward()
            optimizer.step()
            print(f'loss: {negative_log_likelihood.item()} at epoch:{i}')
    #################################################################################################################
    x_prior = prior.sample(torch.Size([1024,]))
    data, _ = datasets.make_circles(n_samples = 128, factor=0.5, noise=0.05, random_state=0)
    data = torch.as_tensor(data)
    samples = []
    samples.append(("data", data))
    zs = model.f(x_prior)[0]
    layer = []
    layer.append("dist")
    layer.extend(model.modulenames())
    samples.extend(list(zip(layer,zs)))
    scatterplots(samples)
    plt.savefig('Flows.png')
    #################################################################################################################
    x,_ = datasets.make_circles(n_samples = 128, factor=0.5, noise=0.05, random_state=0)
    x = torch.as_tensor(x, dtype = dtype)
    _, zs, _ = model.invf(x)
    z = zs[-1]

    x = x.detach().numpy()
    z = z.detach().numpy()
    p = model.prior.sample([128])
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(p[:,0], p[:,1], c='g', s=5)
    plt.scatter(z[:,0], z[:,1], c='r', s=5)
    plt.scatter(x[:,0], x[:,1], c='b', s=5)
    plt.legend(['dist', 'from data to dist', 'data'])
    plt.axis('scaled')
    plt.title('from data to dist')
    plt.savefig('All.png')
    
    p = model.prior.sample([1024])
    zs, _ = model.f(p)
    z = zs[-1]
    z = z.detach().numpy()
    plt.subplot(122)
    plt.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
    plt.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
    plt.legend(['data', 'from dist to data'])
    plt.axis('scaled')
    plt.title('from dist to data')
    plt.savefig('comparaison.png')