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
from utils import *


if __name__ == "__main__":

    dim = 2
    number_affine = 4
    mu_x = 12
    s_x = 4
    batchsize = 64
    lr = 0.001
    nb_epochs = 100000
    #################################################################################################
    mu = torch.zeros(dim)
    s = torch.ones(dim)
    #################################################################################################
    prior = torch.distributions.independent.Independent(torch.distributions.normal.Normal(mu, s),1)
    x_distribution = torch.distributions.independent.Independent(torch.distributions.normal.Normal(torch.ones(dim) * mu_x, torch.ones(dim) * s_x),1)
    #################################################################################################
    modules = [AffineFlow(dim) for _ in range(number_affine)]
    model = FlowModel(prior, *modules)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
    #################################################################################################
    x_prior = prior.sample(torch.Size([1024,]))
    X_init = model.f(x_prior)[0]
    X_init = X_init[-1].detach().numpy()
    #################################################################################################
    for i in range(nb_epochs):
        x = x_distribution.sample(sample_shape=torch.Size([batchsize]))
        z_0_logprob, zs, logdet = model.invf(x)
        logprob = z_0_logprob + logdet
        negative_log_likelihood = - torch.mean(logprob)
        model.zero_grad()
        negative_log_likelihood.backward()
        optimizer.step()
        print(negative_log_likelihood.item())
    #################################################################################################
    x_prior = prior.sample(torch.Size([1024,]))
    X = x_distribution.sample(torch.Size([1024,])).detach().numpy()
    X_flow = model.f(x_prior)[0]
    X_flow = X_flow[-1].detach().numpy()
    plt.scatter(X_init[:,0], X_init[:,1], c='r', s=5)
    plt.scatter(X[:,0], X[:,1], c='g', s=5)
    plt.scatter(X_flow[:,0], X_flow[:,1], c='b', s=5)
    plt.legend(['FlowsIntial','DistributionGoal', 'FlowsFinal'])
    plt.savefig(f'DistributionGoal{number_affine}Affine.png', dpi=2048) 
    #################################################################################################
    plt.subplot(1, 2, 1)
    sns.distplot(X[:,0], hist=False, kde=True,
                bins=None,color='red',
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2},
                label='data')
    sns.distplot(X_flow[:,0], hist=False, kde=True,
                bins=None, color='blue',
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2},
                label='flow')
    plt.title(r'$p(x_1)$')
    plt.subplot(1, 2, 2)
    sns.distplot(X[:,1], hist=False, kde=True,
                bins=None,color='red',
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2},
                label='data')
    sns.distplot(X_flow[:,1], hist=False, kde=True,
                bins=None, color='blue',
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2},
                label='flow')
    plt.title(r'$p(x_2)$')
    plt.legend(['Target','Result'])
    plt.savefig(f'DistributionComparison{number_affine}Affine.png', dpi=200) 
    #################################################################################################