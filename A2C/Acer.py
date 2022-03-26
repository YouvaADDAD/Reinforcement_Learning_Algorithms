import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from core import *
import torch.nn as nn
import torch
import time
from torch.distributions import Categorical
from copy import deepcopy
from memory import Memory

class ActorCritic(nn.Module):
    def __init__(self,n_state,n_action,layers=[30,30],activation=torch.tanh,finalActivation=nn.Softmax(dim=-1),dropout=0.0):
        super(ActorCritic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.actor=NN(self.n_state,self.n_action,layers=layers,activation=activation,finalActivation=finalActivation,dropout=dropout)
        self.critic=NN(self.n_state,self.n_action,layers=layers,activation=activation,dropout=dropout)
    
    def forward(self,observation):
        policy= self.actor(observation)
        value = self.critic(observation)
        return policy,value

class ACER(object):
    def __init__(self, env,opt,layers=[30,30]): 
        #Environment 
        self.env=env
        self.opt=opt
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False

        #Parameters
        self.gamma=opt.gamma
        self.Lambda=opt.Lambda
        self.prior=opt.prior

        #Buffer
        self.capacity=opt.capacity
        self.events=Memory(mem_size=self.capacity,prior=self.prior)
        self.ratio_replay=opt.ratio_replay
        self.batch_size=opt.batch_size
        
        #Compteur
        self.nbEvents=0

        #A2C models
        self.model=ActorCritic(self.env.observation_space.shape[0],self.action_space.n , layers=layers,finalActivation=nn.Softmax(dim=1), activation=torch.tanh,dropout=0.0)
        self.target=deepcopy(self.model)

        #Histoire d'etre sur de ne pas mettre a jour le target
        for param in self.target.parameters():
            param.requires_grad=False

        #Optimizer
        self.optim=torch.optim.RMSprop(params=self.model.parameters(),weight_decay=0.0,lr=opt.lr,alpha=0.99)
        
        #Loss
        self.loss=nn.SmoothL1Loss()
    
    # sauvegarde du modèle
    def save(self,outputDir):
        pass
    
    # chargement du modèle.
    def load(self,inputDir):
        pass

    def act(self,obs):
        with torch.no_grad():
            obs=torch.as_tensor(obs)
            policy,_=self.model(obs)
            action = Categorical(policy).sample().item()
            return action.item()

    def store(self,ob,action,reward,mu,done,it): 
        if not self.test:
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            self.events.store((ob,action,reward,mu,done))

    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

    def learn(self,on_policy):
        if self.test:
            pass
        else:
            #Get sample if on_policy or not
            if not on_policy:
                _,_,l = self.events.sample(self.batch_size)
                states,actions,rewards,mus,dones=map(list,zip(*l))
                states =torch.FloatTensor(states).view(self.batch_size,-1)
                actions=torch.FloatTensor(actions).view(self.batch_size)
                rewards=torch.FloatTensor(rewards).view(self.batch_size)
                dones  =torch.FloatTensor(dones).view(self.batch_size)
                mus    =torch.FloatTensor(mus).view(self.batch_size,-1)
            else:
                states,actions,rewards,mus,dones=self.events.getData(-1)
                states =torch.FloatTensor(states).view(1,-1)
                actions=torch.FloatTensor(actions).view(1)
                rewards=torch.FloatTensor(rewards).view(1)
                dones  =torch.FloatTensor(dones).view(1)
                mus    =torch.FloatTensor(mus).view(1,-1)
            ##########################################################################
            
        
