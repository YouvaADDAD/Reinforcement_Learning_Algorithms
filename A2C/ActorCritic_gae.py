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
from collections import deque
from copy import deepcopy
import random


class ActorCritic(nn.Module):
    def __init__(self,n_state,n_action,layers=[30,30],activation=torch.tanh,finalActivation=nn.Softmax(dim=1),dropout=0.0):
        super(ActorCritic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.actor=NN(self.n_state,self.n_action,layers=layers,activation=activation,finalActivation=finalActivation,dropout=dropout)
        self.critic=NN(self.n_state,1,layers=layers,activation=activation,dropout=dropout)
    
    def forward(self,observation):
        policy_dist=Categorical(self.actor(observation))
        value = self.critic(observation)
        return value,policy_dist

class A2C(object):
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

        #Buffer
        self.capacity=opt.capacity
        self.events=deque([],maxlen=self.capacity)
        
        #Compteur
        self.nbEvents=0

        #A2C models
        self.model=ActorCritic(self.env.observation_space.shape[0],self.action_space.n , layers=layers,finalActivation=nn.Softmax(dim=1), activation=torch.tanh,dropout=0.0)
        self.target=deepcopy(self.model)

        #Optimizer
        #self.optim=torch.optim.Adam(params=self.model.parameters(),weight_decay=0.0,lr=opt.lr)
        self.optim=torch.optim.RMSprop(params=self.model.parameters(),weight_decay=0.0,lr=opt.lr,alpha=0.99)

        #Loss
        self.loss=nn.SmoothL1Loss()

    def act(self,obs):
        with torch.no_grad():
            obs=torch.as_tensor(obs)     
            _,policy_dist=self.model(obs)
            action=policy_dist.sample()
            return action.item()

        # sauvegarde du modèle
    
    # sauvegarde du modèle
    def save(self,outputDir):
        pass
    
    # chargement du modèle.
    def load(self,inputDir):
        pass

    def store(self,obs,action,reward,new_obs,done,it): 
        if not self.test:
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            self.events.append((obs,action,reward,new_obs,done))
    
    
    def discount_rewards_(self,rewards,dones,values):
        with torch.no_grad():
            next_value=0
            gae=0.
            returns=[]
            for step in reversed(range(len(rewards))):
                TD=rewards[step]+self.gamma*next_value*(1-dones[step])-values[step]
                gae=TD+self.gamma*self.Lambda*(1-dones[step])*gae
                next_value= values[step]
                returns.append(gae+values[step])
            return torch.tensor(returns[::-1])


    def learn(self):
        if self.test:
            pass
        else:
            #ls0,la,lr,_,ld,=map(list,zip(*random.sample(self.events,len(self.events))))
            ls0,la,lr,_,ld,=map(list,zip(*self.events))
            ls0=torch.FloatTensor(ls0).view(len(self.events),-1)
            la=torch.FloatTensor(la).view(len(self.events))
            lr=torch.FloatTensor(lr).view(len(self.events))
            ld  =torch.FloatTensor(ld).view(len(self.events))
            values,policy_dist=self.model(ls0)
            logprobs=policy_dist.log_prob(la).view(len(self.events),-1)
            ##################################################################################################
            returns=self.discount_rewards_(lr,ld,values.detach()).view(len(self.events),-1)
            advantage=returns-values
            ##################################################################################################
            actor_loss=-torch.mean(logprobs*advantage.detach())
            critic_loss=self.loss(values,returns)
            error=actor_loss+critic_loss
            self.optim.zero_grad()
            error.backward()
            self.optim.step()
            self.events.clear()
            ##################################################################################################

if __name__ == '__main__':
    #config_random_gridworld.yaml
    #config_random_cartpole.yaml
    #config_random_lunar.yaml
    env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "A2C_gae") 
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = A2C(env,config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False

    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
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

        new_ob = agent.featureExtractor.getFeatures(ob)

        while True:
            if verbose:
                env.render()
            
            ob = new_ob  
            action= agent.act(ob) #Tensor
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)
            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob,action,reward,new_ob,done,j)
            rsum += reward

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                break
        agent.learn()
        

    env.close()