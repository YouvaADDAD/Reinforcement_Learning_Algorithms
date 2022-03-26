import torch
from torch import nn
from torch.distributions import Categorical
from utils import *
from Expert_Load import *


class NN(nn.Module):
    def __init__(self,inSize,outSize,layers=[100,100],activation=nn.ReLU,finalActivation=None,dropout=0.0):
        super(NN,self).__init__()
        layer=[]
        for x in layers:
            layer.append(nn.Linear(inSize,x))
            layer.append(activation())
            if dropout>0.0:
                layer.append(nn.Dropout())
            inSize=x
        layer.append(nn.Linear(inSize,outSize))

        if finalActivation is not None:
            classname = finalActivation().__class__.__name__
            if classname.find('Softmax')!= -1:
                layer.append(finalActivation(dim=-1))
            else:
                layer.append(finalActivation())

        self.model=nn.Sequential(*layer)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,n_state,n_action,layers=[100,100],activation=nn.Tanh, finalActivation = nn.Sigmoid, sigma_noise =0.01):
        super(Discriminator,self).__init__()
        inSize=n_state+n_action
        self.sigma_noise = sigma_noise
        self.model=NN(inSize,1,layers=layers, activation=activation, finalActivation = finalActivation)
    
    def forward(self,obs,action):
        state_action = torch.cat([obs, action.float()], dim = -1)
        state_action = state_action + (torch.rand_like(state_action) * self.sigma_noise)
        return self.model(state_action)

class ActorCritic(nn.Module):
    def __init__(self,n_state,n_action,layers=[30,30],activation=nn.ReLU,finalActivation=nn.Softmax,dropout=0.0):
        super(ActorCritic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.actor=NN(self.n_state,self.n_action,layers=layers,activation=activation,finalActivation=finalActivation,dropout=dropout)
        self.critic=NN(self.n_state,1,layers=layers,activation=activation,dropout=dropout)
    
    def forward(self,observation):
        policy_dist = Categorical(self.actor(observation))
        value = self.critic(observation)
        return value,policy_dist




