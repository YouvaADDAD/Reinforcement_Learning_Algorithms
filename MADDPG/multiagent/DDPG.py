import argparse
from readline import parse_and_bind
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import torch
from multiagent.utils import *
from multiagent.core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt, use
import yaml
from datetime import datetime
import torch.nn as nn
import torch
import time
from copy import deepcopy

#########Creation des modéles afin de de pouvoir rajouter la Batch Normalization
class Actor(nn.Module):
    def __init__(self,n_state,n_action,act_limit=1,layers=[30,30],activation=nn.ReLU,finalActivation=None,dropout=0.0,use_batch_norm=False):
        super(Actor,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.act_limit=act_limit  #-> No need here since we have actions between -1 and 1
        ##################################################
        layer = nn.ModuleList([])
        inSize =n_state
        for x in layers:
            layer.append(nn.Linear(inSize, x))
            if use_batch_norm:
                layer.append(nn.BatchNorm1d(x))
            layer.append(activation())
            if dropout > 0:
                layer.append(nn.Dropout(dropout)) 
            inSize=x  
        layer.append(nn.Linear(inSize, n_action))
        layer.append(nn.Tanh())
        if finalActivation:
            layer.append(finalActivation())
        self.actor=nn.Sequential(*layer)
        ##################################################

    def forward(self,obs):
        #Current observation for one agent
        #No need for act limit action between -1 and 1
        action = self.act_limit * self.actor(obs)
        return action

class Critic(nn.Module):
    def __init__(self,n_state,n_action,layers=[30,30],activation=nn.ReLU,finalActivation=None,dropout=0.0,use_batch_norm=False):
        super(Critic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        layer = nn.ModuleList([])
        inSize =n_state+n_action
        for x in layers:
            layer.append(nn.Linear(inSize, x))
            if use_batch_norm:
                layer.append(nn.BatchNorm1d(num_features=x))
            layer.append(activation())
            if dropout > 0:
                layer.append(nn.Dropout(dropout)) 
            inSize = x
        layer.append(nn.Linear(inSize, 1))
        if finalActivation:
            layer.append(finalActivation())
        self.critic=nn.Sequential(*layer)
    
    def forward(self,obs,action):
        return self.critic(torch.cat([obs,action],dim=-1)).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self,n_state,n_action,all_states,n_agent,act_limit,layers=[30,30],activation=nn.ReLU,dropout=0.0,use_batch_norm=False):
        super(ActorCritic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.act_limit=act_limit
        self.all_states=all_states

        #Integrer par défaut pour le tanh
        self.policy=Actor(self.n_state,self.n_action,self.act_limit,layers=layers,activation=activation,dropout=dropout,use_batch_norm=use_batch_norm)
        self.q=Critic(self.all_states,self.n_action*n_agent,layers=layers,activation=activation,dropout=dropout,use_batch_norm=use_batch_norm)

class DDPG(object):
    def __init__(self,n_state,n_action,all_states,n_agent,opt,high=1,low=-1,
                 layers=[30,30],activation=nn.LeakyReLU):
        
        #Environment 
        self.n_state=n_state
        self.n_action = n_action
        self.all_states=all_states
        self.n_agent=n_agent
        self.high=high
        self.low=low
        
        #Parameters
        self.use_batch_norm=opt.use_batch_norm
        self.gamma=opt.gamma
        self.ru=opt.ru

        #Uhlenbeck & Ornstein, 1930
        self.N=Orn_Uhlen(self.n_action,sigma=opt.sigma)
        self.sigma=opt.sigma
        #Initialize target network Q′ and μ′ with weights θQ′ ← θQ, θμ′ ← θμ
        self.model=ActorCritic(self.n_state,self.n_action,self.all_states,self.n_agent,self.high,layers=layers,activation=activation,use_batch_norm=self.use_batch_norm)
        self.target=deepcopy(self.model)

        #Freeze target network pour ne pas le mettre a jour
        for param in self.target.parameters():
            param.requires_grad = False

        #Optimiseur & Loss
        self.loss=nn.SmoothL1Loss()
        self.policy_optim=torch.optim.Adam(self.model.policy.parameters(),weight_decay=0.0,lr=opt.lr_pi)
        self.q_optim=torch.optim.Adam(self.model.q.parameters(),weight_decay=0.0,lr=opt.lr_q)
    
    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass
    
    def act(self,obs):
        with torch.no_grad():
            self.model.policy.eval()
            obs=torch.as_tensor(obs,dtype=torch.float).unsqueeze(0)
            action=self.model.policy(obs)+self.N.sample()
            self.model.policy.train()
        return torch.clamp(action,min=self.low,max=self.high).squeeze(0).numpy()
        
    def learn(self,obs,actions,rewards,next_obs,dones,obs_for_agent,next_obs_for_agent):
        #All Parameters are Tensors
        # To DO
        pass

                                      
    def update_parameters(self):
        for param, param_target in zip(self.model.parameters(), self.target.parameters()):
            param_target.data.mul_(self.ru)
            param_target.data.add_((1 - self.ru) * param.data)
            
