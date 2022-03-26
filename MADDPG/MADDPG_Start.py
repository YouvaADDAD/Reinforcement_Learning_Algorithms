import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import multiagent
import multiagent.scenarios
import multiagent.scenarios.simple_tag as simple_tag
import multiagent.scenarios.simple_tag as simple_spread
import multiagent.scenarios.simple_tag as simple_adversary
from multiagent.environment import MultiAgentEnv
from torch.utils.tensorboard import SummaryWriter
import multiagent.scenarios as scenarios
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import logging
import json
import subprocess
from collections import namedtuple,defaultdict
from utils import *
from core import *
from datetime import datetime
import random
from collections import deque
from Actor_Critic import *

class Agent(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)

    # agent_type: 0 = random, 1 = DDPG, 2 = MADDPG
    def __init__(self,env,opt,id,action_size,obs_shapes,agent_type = 2, nb_subPolicies = 1, optimizer = torch.optim.Adam,lr = 0.1, lrq = 0.1):


        self.id=id
        self.opt=opt


        self.agent_type = agent_type
        self.obs_shapes = obs_shapes
        self.env = env

        self.action_size=action_size
        self.nb_subPolicies=nb_subPolicies
        self.policies = []
        self.targetPolicies = []
        self.q = None
        self.qtarget = None


        if agent_type==0: #random
            self.nb_subPolicies = 0

        else:
            #TODO votre agent ici
            self.policies=[Actor(obs_n[id],action_size,layers=opt.layers,activation=opt.activation,finalActivation=opt.finalActivation) for _ in range (nb_subPolicies)]
            self.targetPolicies=copy.deepcopy(self.policies)

            self.q=Critic(sum(obs_n),action_size*env.n,layers=opt.layers,activation=opt.activation)
            self.qtarget=copy.deepcopy(self.q)

            for net in self.targetPolicies:
                for param in net.parameters():
                    param.requires_grad=False

            for param in self.qtarget.parameters():
                param.requires_grad=False

            self.polyakP = self.opt.polyakP
            self.polyakQ = self.opt.polyakQ

        self.currentPolicy=0
        self.events = [deque(maxlen=self.opt.capacity) for i in range(nb_subPolicies)]
        self.batchsize = self.opt.batchsize

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)


        # Creation optimiseurs
        if agent_type > 0:  # not random
            wdq = self.opt.wdq   # weight decay for q
            wdp = self.opt.wdp  # weight decay for pi
            self.qtarget.load_state_dict(self.q.state_dict())
            for i in range(nb_subPolicies):
                self.targetPolicies[i].load_state_dict(self.policies[i].state_dict())

            self.optimizerP = [optimizer([{"params": self.policies[i].parameters()}], weight_decay=wdp, lr=lr) for i in
                                   range(nb_subPolicies)]
            self.optimizerQ = optimizer([{"params": self.q.parameters()}], weight_decay=wdq, lr=lrq)

    def act(self,obs):

        if self.agent_type == 0:
            a = self.floatTensor.new((np.random.rand(self.action_size) - 0.5) * 2).view(-1)
            return a
        return self.policies[self.currentPolicy](obs)

    def getTargetAct(self,obs):
        if self.agent_type == 0:
            a=self.floatTensor.new((np.random.rand(obs.shape[0],self.action_size) - 0.5) * 2).view(-1,self.action_size)
            return a
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        return self.targetPolicies[i](obs)

    def addEvent(self,event):
        if self.agent_type == 0:
            return
        i=self.currentPolicy
        self.events[i].append(event)
    
    def sample(self):
        if self.agent_type == 0:
            return
        events=self.events[self.currentPolicy]
        idx=np.random.randint(0,high=len(events),size=self.batchsize)
        return [events[i] for i in idx]

    def selectPolicy(self):
        if self.agent_type==0:
            return 0
        i = np.random.randint(0, self.nb_subPolicies, 1)[0]
        self.currentPolicy = i

    def eval(self):
        if self.q is not None:
            for p in self.policies:
                p.eval()
            self.q.eval()
    
    def train(self):
        if self.q is not None:
            for p in self.policies:
                p.train()
            self.q.train()

    def soft_update(self):
        if self.q is not None:

            for i in range(len(self.policies)):
                for target, src in zip(self.targetPolicies[i].parameters(), self.policies[i].parameters()):
                    target.data.copy_(target.data * self.polyakP + src.data * (1-self.polyakP))

            for target, src in zip(self.qtarget.parameters(), self.q.parameters()):
                target.data.copy_(target.data * self.polyakQ + src.data * (1 - self.polyakQ))

    def setcuda(self,device):
        Agent.floatTensor = torch.cuda.FloatTensor(1, device=device)
        Agent.longTensor = torch.cuda.LongTensor(1, device=device)
        if self.q is not None:
            for p in self.policies:
                p.setcuda(device)
            for p in self.targetPolicies:
                p.setcuda(device)
            self.q.setcuda(device)
            self.qtarget.setcuda(device)

    def save(self,file):

        if self.q is not None:
            for x in range(self.nb_subPolicies):
                torch.save(self.policies[x].state_dict(), file + "_policy_" + str(self.id) + "_" + str(x) + ".txt")
            torch.save(self.q.state_dict(), file + "_value_"+str(self.id)+".txt")

    def load(self,file):
        if self.q is not None:
            for x in range(self.nb_subPolicies):
                self.policies[x].load_state_dict(torch.load(file +"_policy_"+str(self.id)+"_"+str(x)+".txt"))
                self.q.load_state_dict(torch.load(file + "_value_"+str(self.id)+".txt"))


class MADDPG(object):
    floatTensor = torch.FloatTensor(1)
    longTensor = torch.LongTensor(1)
    verbose = 0


    def __init__(self,env,opt,action_size,obs_shapes,noise,noiseTest):
        super(MADDPG, self).__init__()

        self.action_size = action_size
        self.env=env
        self.opt=opt
        self.gamma=self.opt.gamma


        agent_types = self.opt.agent_types
        nb_subPolicies =  self.opt.nb_subPolicies
        lr = self.opt.lr
        lrq = self.opt.lrq
        device = self.opt.device


        nbSteps = self.opt.nbSteps
        freqOptim = self.opt.freqOptim
        optimizer=torch.optim.Adam
        seed=self.opt.seed



        self.freqOptim=freqOptim
        self.nbSteps=nbSteps
        self.noise=noise
        self.noiseTest = noiseTest
        self.startEvents=self.opt.startEvents
        self.test=False

        self.nbEvts=0
        self.nbEvents=0


        self.polyakP=self.opt.polyakP
        self.polyakQ = self.opt.polyakQ
        self.nbOptim=0

        self.nbRuns=0


        self.agents=[]

        for i in range(env.n):
            a=Agent(env,opt,i,action_size,obs_shapes,agent_types[i],nb_subPolicies[i],optimizer,lr=lr[i],lrq=lrq[i])
            self.agents.append(a)

        self.nb=0

        self.sumDiff=0
        prs("lr",lr)


        self.current=[]
        self.batchsize=self.opt.batchsize

        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)

        if device>=0:
            cudnn.benchmark = True
            torch.cuda.device(device)
            torch.cuda.manual_seed(seed)
            MADDPG.floatTensor = torch.cuda.FloatTensor(1, device=device)
            MADDPG.longTensor = torch.cuda.LongTensor(1, device=device)
            for i in range(env.n):
                self.agents[i].setcuda(device)

    def save(self,file):
        for agent in self.agents:
            agent.save(file)

    def load(self,file):
        for agent in self.agents:
            agent.load(file)

    def store(self,ob,action,new_ob,rewards,done,it):
        d=done[0]
        if it == self.opt.maxLengthTrain:
            print("undone")
            d = False

        for a in self.agents:
            a.addEvent((ob, action, rewards, new_ob, d))
        self.nbEvts += 1

    def act(self, obs):

        for a in self.agents:
            a.eval()

        if self.nbEvts>self.startEvents:
            with torch.no_grad():

                actions = torch.cat([self.agents[i].act(self.floatTensor.new(obs[i])).view(-1) for i in range(self.env.n)],dim=-1).cpu()
                #print(actions)
                #print
                if (not self.test) or self.opt.sigma_noiseTest>0:
                    noise=self.noise
                    if self.test:
                        noise=self.noiseTest
                    e=torch.cat([x.sample() for x in noise],dim=-1)
                    actions=actions+e.view(-1)
                actions=actions.numpy()

        else:
            actions =np.concatenate([x.sample().numpy() for x in self.noise],axis=-1)


        return actions.reshape((self.env.n,-1))

    def endsEpisode(self):
        for i in range(self.env.n):
            self.noise[i].reset()
            self.noiseTest[i].reset()
            self.agents[i].selectPolicy()

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        #print(self.nbEvents,self.opt.freqOptim,self.opt.startEvents)
        if self.nbEvents % self.opt.freqOptim == 0 and self.nbEvents > self.opt.startEvents:
            print("Time to Learn! ")
            return True
        return False

    def learn(self):
        self.nbOptim+=1
        for a in self.agents:
            a.train()
        sl=torch.nn.MSELoss() #SmoothL1Loss()

        for j in range(self.nbSteps):
            for x in range(self.env.n):
                agent=self.agents[x]
                currentPolicy=agent.currentPolicy
                if agent.agent_type==0:
                    continue
                #TODO learn agent
                batch = agent.sample()
                obs, actions, rewards, next_obs, dones = map(list,zip(*batch))
                obs=torch.FloatTensor(obs) #batch_size,n_agent,obs_shape
                actions=torch.FloatTensor(actions)#batch,n_agent,act_shape
                rewards=torch.FloatTensor(rewards)#batch,n_agent
                next_obs=torch.FloatTensor(next_obs)#batch_size,n_agent,obs_shape
                dones=torch.FloatTensor(dones)#batch
                rewards=torch.clamp(rewards,min=-100,max=100)
                #---------------------------------------------------------------
                current_reward=rewards[:,x]
                all_obs = obs.view(self.batchsize,-1)
                all_next_obs = next_obs.view(self.batchsize,-1)
                target_actions = torch.cat([agent2.getTargetAct(next_obs[:,i,:]) for i,agent2 in enumerate(self.agents)],dim=-1)
                current_all_action = actions.view(self.batchsize,-1)
                #---------------------------------------------------------------
                with torch.no_grad():
                    y = current_reward + self.gamma * (1-dones) * agent.qtarget(all_next_obs,target_actions)
                q=agent.q(all_obs,current_all_action)
                #----------------------------------------------------------------
                #Le CRITIC
                agent.optimizerQ.zero_grad() #Zero risque
                critic_loss = sl(q, y)
                critic_loss.backward()
                logger.direct_write('Loss/critic',critic_loss,self.nbEvents)
                agent.optimizerQ.step()
                #----------------------------------------------------------------
                #L'ACTOR
                agent.optimizerP[currentPolicy].zero_grad()
                obs_agent = obs[:,x,:]
                act_agent = agent.policies[currentPolicy](obs_agent)
                actions_not_agent=actions.clone()
                actions_not_agent[:,x,:] = act_agent

                actions_cat = actions_not_agent.view(self.batchsize,-1)
                actor_loss = - agent.q(all_obs, actions_cat).mean()
                logger.direct_write('Loss/actor',actor_loss,self.nbEvents)
                actor_loss.backward()
                agent.optimizerP[currentPolicy].step()

                


        for x in range(self.env.n):
            self.agents[x].soft_update()








"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    print("env ok!")
    return env #,scenario,world

def padObs(obs,size):
    return([np.concatenate((o,np.zeros(size-o.shape[0]))) if o.shape[0]<size else o for o in obs])

if __name__ == '__main__':

    env, config, outdir, logger = init('./configs/config_maddpgStart_simple_spread.yaml', "MADDPG")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    noise = [config["noise"](config["action_size"], sigma=config["sigma_noise"]) for i in range(env.n)]
    noiseTest = [config["noise"](config["action_size"], sigma=config["sigma_noiseTest"]) for i in range(env.n)]
    #
    #
    obs = env.reset()
    obs_n = [o.shape[0] for o in obs]
    mshape=max(obs_n)
    obs_n = [mshape for o in obs]

    agent = MADDPG(env, config, config["action_size"], obs_n, noise, noiseTest)
    verbose=True
    rsum = np.zeros(env.n)
    mean = np.zeros(env.n)
    itest = 0
    ntest = 0
    ntrain = 0
    rewards = np.zeros(env.n)
    done = [False]
    ended=False
    agent.nbEvents = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = np.zeros(env.n)

        obs = env.reset()
        obs = padObs(obs, mshape)
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = np.zeros(env.n)
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            #logger.direct_write("rewardTest", mean / nbTest, itest)
            for k in range(env.n):
                logger.direct_write("rewardTest/raw_" + str(k), mean[k]/nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()


        new_obs = np.array(obs)

        ended = False

        while True:

            if verbose:
                env.render(mode="none")

            obs = new_obs
            actions = agent.act(obs)

            j += 1
            new_obs, rewards, done, _ = env.step(actions)
            new_obs = padObs(new_obs, mshape)
            if ((not agent.test) and j >= int(config["maxLengthTrain"])) or (j>=int(config["maxLengthTest"])) :
                ended=True
            new_obs = np.array(new_obs)
            rewards = np.clip(np.array(rewards),-100,100)
            if (not agent.test):

                agent.store(obs, actions, new_obs, rewards, done,j)

                if agent.timeToLearn(ended):
                    agent.learn()

            rsum += rewards

            if done[0] or ended:
                agent.endsEpisode()
                if not agent.test:
                    ntrain += 1
                    print("Train:",str(ntrain) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                    for k in range(len(rsum)):
                        logger.direct_write("reward/raw_"+str(k), rsum[k], ntrain)
                else:
                    ntest += 1
                    print("Test:", str(ntest) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")


                mean = mean + rsum
                rsum = np.zeros(env.n)

                break


    env.close()

