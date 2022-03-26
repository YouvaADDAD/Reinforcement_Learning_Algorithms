import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
import numpy as np
from datetime import datetime
from core import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from memory import Memory
import time
from copy import deepcopy
from torch.distributions import Normal
from itertools import chain


class PolicyNN(nn.Module):
    def __init__(self, n_observation, n_action,act_limit, layers=[256,256], activation=F.relu,dropout=0.0):
        super(PolicyNN,self).__init__()
        self.n_observation=n_observation
        self.n_action=n_action
        self.act_limit=act_limit

        #On met une finalActivation TanH afin non linearise la sortie de model sachant qu'on va faire rentrer le resultat du forward dans le NN mu_ et std_
        self.model=NN(self.n_observation,layers[-1] , layers=layers, activation=activation,finalActivation=activation,dropout=dropout)
        self.mu_ = nn.Linear(layers[-1], self.n_action) #Pour une sortie de la mean
        self.std_ = nn.Linear(layers[-1], self.n_action) #Pour une sortie du sigma
    
    def forward(self,obs):
        output= self.model(obs)
        mu = self.mu_(output)
        #Pour avoir le std entre positive et pas tres eloigner
        std = torch.clamp(self.std_(output), 2e-10, 10)
        policy_distribution = Normal(mu, std)
        probs_action=policy_distribution.rsample() #Reparametrization trick
        action = torch.tanh(probs_action)
        #Version plus stable, reecrire tanh avec l'exponential, et softplus(x)=log(1+exp(x)), sofplus elle meme utilise une stabilité avec un threshold
        logprob_policy = (policy_distribution.log_prob(probs_action)-(2*np.log(2)-2*probs_action-2*F.softplus(-2*probs_action))).sum(dim=-1)
        action = self.act_limit * action
        return action, logprob_policy

class ActorCriticNN(nn.Module):
    def __init__(self, n_observation, n_action,act_limit, layers=[256,256], activation=torch.tanh,dropout=0.0):
        super(ActorCriticNN,self).__init__()
        self.n_observation=n_observation
        self.n_action=n_action
        self.act_limit=act_limit

        #La politique
        self.policy=PolicyNN(n_observation, n_action,act_limit, layers=layers, activation=activation,dropout=dropout)
        #Fonction de Quality
        self.Q_1=NN(self.n_observation+self.n_action,1, layers=layers, activation=activation,dropout=0.0)
        self.Q_2=NN(self.n_observation+self.n_action,1, layers=layers, activation=activation,dropout=0.0)


    def getAction(self, obs):
        with torch.no_grad():
            action, logprobs = self.policy(obs)
            return action.numpy(),logprobs

class SAC(object):
    def __init__(self, env,opt,layers=[256,256]):    
        #Load yaml and env 
        self.opt=opt
        self.env=env

        #Buffer
        self.batch_size=opt.batch_size
        self.capacity=opt.capacity
        self.nbEvents=0
        self.start_after=10000

        #Hyperparameters
        self.gamma=opt.gamma
        self.ru=opt.ru

        #Optimization step
        self.test=False
        self.update_target=opt.update_target
        self.lr_pi=opt.lr_pi
        self.lr_q=opt.lr_q
        self.update_many=self.opt.update_many
        self.alpha=opt.alpha #For entropy

        #From environment
        self.n_observation=env.observation_space.shape[0]
        self.n_action=env.action_space.shape[0]
        self.act_limit=env.action_space.high[0]

        #Memory
        self.events=Memory(self.capacity,prior=False)

        #Models
        self.model=ActorCriticNN(self.n_observation, self.n_action, self.act_limit,layers=layers, activation=F.leaky_relu,dropout=0.0)
        self.target=deepcopy(self.model)

        #Histoire d'etre sur qu'on mettent pas a jour le target
        for param in self.target.parameters():
            param.requires_grad = False
        
        #Optimizer
        self.Q_optim=torch.optim.Adam(chain(self.model.Q_1.parameters(),self.model.Q_2.parameters()), lr=self.lr_q)
        self.policy_optim = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr_pi)

        #Loss
        self.loss_fn_Q1=nn.SmoothL1Loss()
        self.loss_fn_Q2=nn.SmoothL1Loss()
        
     # sauvegarde du modèle
    def save(self,outputDir):
        pass
    
    def act(self,obs):
        obs=torch.as_tensor(obs)
        action, _= self.model.getAction(obs)
        return action

    # chargement du modèle.
    def load(self,inputDir):
        pass

    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done) #(st, at, rt, st+1)
            self.events.store(tr)
    
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents>self.opt.freqOptim and self.nbEvents%self.update_many==0

    def learn(self):
        if self.test:
            pass
        else:
             for _ in range(self.update_many):
                _,_,l = self.events.sample(self.batch_size)
                ls0, la, lr, ls1, ld=map(list,zip(*l))
                ls0=torch.FloatTensor(ls0).view(self.batch_size,-1)
                la = torch.FloatTensor(la).view(self.batch_size,-1)
                lr = torch.FloatTensor(lr).view(self.batch_size)
                ls1=torch.FloatTensor(ls1).view(self.batch_size,-1)
                ld =torch.FloatTensor(ld).view(self.batch_size)

                #Compute loss and update for quality
                #################################################################################################
                with torch.no_grad():
                    actions_tilde, logprobs_tilde = self.model.policy(ls1) #Action et logprob du next state
                    states_acts_next=torch.cat([ls1, actions_tilde], dim=-1)
                    q_target=torch.min(self.target.Q_1(states_acts_next).squeeze(-1),self.target.Q_2(states_acts_next).squeeze(-1))
                    y=lr + self.gamma*(1-ld)*(q_target-self.alpha*logprobs_tilde)
           
                states_acts=torch.cat([ls0, la], dim=-1)

                q1=self.model.Q_1(states_acts).squeeze(-1)
                q2=self.model.Q_2(states_acts).squeeze(-1)

                loss_quality=self.loss_fn_Q1(q1,y)+self.loss_fn_Q2(q2,y)
                logger.direct_write("loss_quality", loss_quality, self.nbEvents)
                self.Q_optim.zero_grad()
                loss_quality.backward()
                self.Q_optim.step()
                #################################################################################################
                #Compute loss and update for policy
                #On a la differentiation
                actions_tilde, logprobs_tilde=self.model.policy(ls0)

                states_acts=torch.cat([ls0, actions_tilde], dim=-1)
                q=torch.min(self.model.Q_1(states_acts).squeeze(-1),self.model.Q_2(states_acts).squeeze(-1))

                loss_policy=(-q+self.alpha*logprobs_tilde).mean()
                logger.direct_write("loss_policy", loss_policy, self.nbEvents)
                self.policy_optim.zero_grad()
                loss_policy.backward()
                self.policy_optim.step()
                #################################################################################################
                with torch.no_grad():
                    #Sachant qu'on utilise pas policy target on peut directement mettre a jour
                    for param,param_target in zip(self.model.parameters(), self.target.parameters()):
                        param_target.data.mul_(self.ru)
                        param_target.data.add_((1-self.ru)*param.data)

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config-Pendulum.yaml', "SAC")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    agent = SAC(env,config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()


        while True:
            if verbose:
                env.render()
            
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward
            ob=new_ob
            if agent.timeToLearn(done):
                agent.learn()
            
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break
    env.close()
