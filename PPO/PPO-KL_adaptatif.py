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
from torch.distributions.kl import kl_divergence
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

class Buffer(object):

    def __init__(self):
        self.observations=[]
        self.actions=[]
        self.rewards=[]
        self.dones=[]
        self.values=[]
        self.log_probs=[]

    def store(self,ob, action,reward,done,value,log_prob):
        self.observations.append(ob) 
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def reset(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def sample(self):
        return self.observations,self.actions,self.rewards,self.dones,self.values,self.log_probs
        
    def __len__(self):
        return len(self.observations)

class PPO(object):
    def __init__(self, env,opt,layers=[30,30],K=30,beta=1,delta=0.5):
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
        self.events=Buffer()
        
        #Compteur
        self.nbEvents=0

        #Parameters PPO
        self.K=K
        self.beta=beta
        self.delta=delta
    
        #Actor Critic
        self.model=ActorCritic(self.env.observation_space.shape[0],self.action_space.n , layers=layers,finalActivation=nn.Softmax(dim=1), activation=torch.tanh,dropout=0.0)

        #Optimiseur & Loss
        self.loss=nn.SmoothL1Loss()
        self.kl=nn.KLDivLoss(log_target=True)
        self.policy_optim=torch.optim.Adam(self.model.actor.parameters(),weight_decay=0.0,lr=opt.lr_pi)
        self.critic_optim=torch.optim.Adam(self.model.critic.parameters(),weight_decay=0.0,lr=opt.lr_v)
    
    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    def store(self,ob, action,reward,done,value,log_prob,it): 
        if not self.test:
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            self.events.store(ob, action,reward,done,value,log_prob)

    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

    def act(self,obs):
        with torch.no_grad():
            obs=torch.as_tensor(obs)
            values,dist = self.model(obs)
            action = dist.sample()
        return action.item(),values,dist.log_prob(action)

    def discount_rewards_(self,rewards,dones,values):
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
            old_states,actions,rewards,dones,old_values,old_logprobs=self.events.sample()
            old_states=torch.FloatTensor(old_states).squeeze(1)
            actions=torch.FloatTensor(actions)
            rewards=torch.FloatTensor(rewards)
            dones=torch.FloatTensor(dones)
            old_values=torch.FloatTensor(old_values)
            old_logprobs=torch.FloatTensor(old_logprobs)
            returns=self.discount_rewards_(rewards,dones,old_values.detach())
            advantage=returns-old_values
            ###############################################################################
            for _ in range(self.K):
                new_values,new_policy_dist = self.model(old_states)    
                new_logprobs=new_policy_dist.log_prob(actions)
                new_values=new_values.squeeze(-1) 
                #with torch.no_grad(): 
                #    advantage=returns-new_values
                ########################################
                ratio=torch.exp(new_logprobs-old_logprobs)
                divergence=torch.nn.functional.kl_div(new_logprobs,torch.exp(old_logprobs))
                logger.direct_write("divergence", divergence, self.nbEvents)
                ########################################
                actor_loss=-torch.mean(ratio*advantage)+self.beta*(divergence)
                self.policy_optim.zero_grad()
                actor_loss.backward()
                self.policy_optim.step()
                #########################################
                #critic_loss=self.loss(advantage,new_values)
                #self.critic_optim.zero_grad()
                #critic_loss.backward()
                #self.critic_optim.step()
            ########################################
            if divergence>=1.5*self.delta:
                self.beta=2*self.beta
            if divergence<= self.delta/1.5:
                self.beta=0.5*self.beta
            ########################################
            new_values,new_policy_dist = self.model(old_states) 
            critic_loss=self.loss(returns,new_values)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.events.reset()


if __name__ == '__main__':
    #config_random_gridworld.yaml
    #config_random_cartpole.yaml
    #config_random_lunar.yaml
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "PPO_KL")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = PPO(env,config)


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

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action,value,log_prob= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action,reward,done,value,log_prob,j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                break

    env.close()




