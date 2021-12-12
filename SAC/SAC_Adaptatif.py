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
import torch.nn as nn
import torch.nn.functional as F
import torch
from memory import Memory
import time
from copy import deepcopy
from torch.distributions import Normal
from itertools import chain


#Plus complete utilisant la batchNormalisation ou le layerNormalisation ou le Dropout
class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[], finalActivation=None, activation=nn.Tanh,dropout=0.0,use_batch_norm=False,use_layer_norm=False):
        super(NN,self).__init__()
        self.inSize=inSize
        self.outSize=outSize
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize,x))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(num_features=x))
            if use_layer_norm:
                self.layers.append(nn.LayerNorm(x))
            if dropout>0.0:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(activation())
            inSize=x
        self.layers.append(nn.Linear(x,outSize))
        if finalActivation is not None:
            self.layers.append(finalActivation())
        self.model=nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.model(x)

    def setcuda(self, device):
        self.cuda(device=device)

class PolicyNN(nn.Module):
    def __init__(self, n_observation, n_action,act_limit, layers=[256,256], activation=nn.ReLU,dropout=0.0,use_batch_norm=False,use_layer_norm=False):
        super(PolicyNN,self).__init__()
        self.n_observation=n_observation
        self.n_action=n_action
        self.act_limit=act_limit

        #On met une finalActivation TanH afin non linearise la sortie de model sachant qu'on va faire rentrer le resultat du forward dans le NN mu_ et std_
        self.model=NN(self.n_observation,layers[-1] , layers=layers, activation=activation,finalActivation=activation,dropout=dropout,use_batch_norm=use_batch_norm,use_layer_norm=use_layer_norm)
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
#On crée un Nouveau model de Critic histoire de ne pas faire de squeeze a chaque fois et de ne pas faire de cat
class CriticNN(nn.Module):
    def __init__(self, n_observation, n_action, layers=[256,256],finalActivation=None, activation=nn.ReLU,dropout=0.0,use_batch_norm=False,use_layer_norm=False):
        super(CriticNN,self).__init__()
        self.model=NN(n_observation+n_action,1,layers=layers,dropout=dropout,activation=activation,finalActivation=finalActivation,use_batch_norm=use_batch_norm,use_layer_norm=use_layer_norm)
    
    def forward(self,obs,act):
        states_actions=torch.cat([obs,act],dim=-1)
        return self.model(states_actions).squeeze(-1)

class ActorCriticNN(nn.Module):
    def __init__(self, n_observation, n_action,act_limit, layers=[256,256], activation=nn.ReLU,dropout=0.0,use_batch_norm=False,use_layer_norm=False):
        super(ActorCriticNN,self).__init__()
        self.n_observation=n_observation
        self.n_action=n_action
        self.act_limit=act_limit

        #La politique
        self.policy=PolicyNN(n_observation, n_action,act_limit, layers=layers, activation=activation,dropout=dropout,use_batch_norm=use_batch_norm,use_layer_norm=use_layer_norm)

        #Fonction de Quality
        self.Q_1=CriticNN(self.n_observation,self.n_action, layers=layers, activation=activation,dropout=dropout,use_batch_norm=use_batch_norm,use_layer_norm=use_layer_norm)
        self.Q_2=CriticNN(self.n_observation,self.n_action, layers=layers, activation=activation,dropout=dropout,use_batch_norm=use_batch_norm,use_layer_norm=use_layer_norm)
    
    def getAction(self, obs):
        with torch.no_grad():
            action, logprobs = self.policy(obs)
            return action.numpy(),logprobs

class SAC_Adaptatif(object):
    def __init__(self,env,opt,layers=[256,256], activation=nn.ReLU,dropout=0.0,use_batch_norm=False,use_layer_norm=False) -> None:
        super().__init__()
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
        self.entropy_0=torch.tensor([opt.entropy_0])

        #Optimization step
        self.test=False
        self.update_target=opt.update_target
        self.lr_pi=opt.lr_pi
        self.lr_q=opt.lr_q
        self.lr_alpha=opt.lr_alpha
        self.update_many=self.opt.update_many
        #self.alpha=opt.alpha #For entropy

        #From environment
        self.n_observation=env.observation_space.shape[0]
        self.n_action=env.action_space.shape[0]
        self.act_limit=env.action_space.high[0]

        #Memory
        self.events=Memory(self.capacity,prior=False)

        #Models
        self.model=ActorCriticNN(self.n_observation, self.n_action, self.act_limit,layers=layers ,activation=activation,dropout=dropout,use_layer_norm=use_layer_norm,use_batch_norm=use_batch_norm)
        self.target=deepcopy(self.model)

        #Histoire d'etre sur qu'on mettent pas a jour le target
        for param in self.target.parameters():
            param.requires_grad = False
        
        #Optimizer
        self.alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=self.lr_alpha)
        self.policy_optim = torch.optim.Adam(self.model.policy.parameters(), lr=self.lr_pi)
        self.Q_optim=torch.optim.Adam(chain(self.model.Q_1.parameters(),self.model.Q_2.parameters()), lr=self.lr_q)

        #Loss
        self.loss_fn_Q1=nn.SmoothL1Loss()
        self.loss_fn_Q2=nn.SmoothL1Loss()

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

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
    
    def act(self,obs):
        obs=torch.as_tensor(obs).unsqueeze(0)
        self.model.policy.eval()
        action, _= self.model.getAction(obs)
        self.model.policy.train()
        return action.squeeze(0)
    
    def learn(self):
        if self.test:
            pass
        else:
             for _ in range(self.update_many):
                _,_,l = self.events.sample(self.batch_size)
                ls0, la, lr, ls1, ld=map(list,zip(*l))
                ls0= torch.FloatTensor(ls0).view(self.batch_size,-1)
                la = torch.FloatTensor(la).view(self.batch_size,-1)
                lr = torch.FloatTensor(lr).view(self.batch_size)
                ls1= torch.FloatTensor(ls1).view(self.batch_size,-1)
                ld = torch.FloatTensor(ld).view(self.batch_size)

                #Compute loss and update for quality
                #################################################################################################
                self.model.eval()
                self.target.eval()
                with torch.no_grad():
                    actions_tilde, logprobs_tilde = self.model.policy(ls1) #Action et logprob du next state
                    q_target=torch.min(self.target.Q_1(ls1,actions_tilde),self.target.Q_2(ls1,actions_tilde))
                    y=lr + self.gamma*(1-ld)*(q_target-self.alpha*logprobs_tilde)
                q1=self.model.Q_1(ls0,la)
                q2=self.model.Q_2(ls0,la)
                self.model.Q_1.train()
                self.model.Q_2.train()
                loss_quality=self.loss_fn_Q1(q1,y)+self.loss_fn_Q2(q2,y)
                logger.direct_write("loss_quality", loss_quality, self.nbEvents)
                self.Q_optim.zero_grad()
                loss_quality.backward()
                self.Q_optim.step()
                self.model.Q_1.eval()
                self.model.Q_2.eval()
                #################################################################################################
                #Compute loss and update for policy
                #On a la differentiation
                actions_tilde, logprobs_tilde=self.model.policy(ls0)
                q=torch.min(self.model.Q_1(ls0, actions_tilde),self.model.Q_2(ls0, actions_tilde))
                self.model.policy.train()
                loss_policy=(-q+self.alpha*logprobs_tilde).mean()
                logger.direct_write("loss_policy", loss_policy, self.nbEvents)
                self.policy_optim.zero_grad()
                loss_policy.backward()
                self.policy_optim.step()
                self.model.policy.eval()
                #################################################################################################
                loss_alpha=-torch.mean(self.alpha*(logprobs_tilde+self.entropy_0).detach())
                self.alpha_optim.zero_grad()
                loss_alpha.backward()
                self.alpha_optim.zero_grad()
                #################################################################################################
                with torch.no_grad():
                    #Sachant qu'on utilise pas policy target on peut directement mettre a jour
                    for param,param_target in zip(self.model.parameters(), self.target.parameters()):
                        param_target.data.mul_(self.ru)
                        param_target.data.add_((1-self.ru)*param.data)


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config-lunarLundar.yaml', "SAC_Adaptatif")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    agent = SAC_Adaptatif(env,config,layers=[256,256], activation=nn.LeakyReLU,dropout=0.0,use_batch_norm=True,use_layer_norm=False)

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
