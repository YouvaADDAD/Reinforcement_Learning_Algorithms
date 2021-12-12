import argparse
import sys
import matplotlib
matplotlib.use("TkAgg")
import gym
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt, use
import yaml
from datetime import datetime
import torch.nn as nn
import torch
import time
from copy import deepcopy
from memory import Memory

#########Creation des modéles afin de de pouvoir rajouter la Batch Normalization

class Actor(nn.Module):
    def __init__(self,n_state,n_action,act_limit,layers=[30,30],activation=nn.ReLU,finalActivation=None,dropout=0.0,use_batch_norm=False):
        super(Actor,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.act_limit=act_limit
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
        action= self.act_limit*self.actor(obs)
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
    def __init__(self,n_state,n_action,act_limit,layers=[30,30],activation=nn.ReLU,dropout=0.0,use_batch_norm=False):
        super(ActorCritic,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.act_limit=act_limit
        #Integrer par défaut pour le tanh
        self.policy=Actor(self.n_state,self.n_action,self.act_limit,layers=layers,activation=activation,dropout=dropout,use_batch_norm=use_batch_norm)
        self.q=Critic(self.n_state,self.n_action,layers=layers,activation=activation,dropout=dropout,use_batch_norm=use_batch_norm)

class DDPG(object):
    def __init__(self, env,opt,layers=[30,30],sigma=0.15,activation=nn.LeakyReLU,use_batch_norm=False):
        #Environment 
        self.env=env
        self.opt=opt
        self.action_space = env.action_space
        self.test=False
        self.n_state=self.env.observation_space.shape[0]
        self.n_action = self.action_space.shape[0]
        self.high=self.action_space.high[0]
        self.low=self.action_space.low[0]

        #Parameters
        self.gamma=opt.gamma
        self.ru=opt.ru
        self.optim_step=opt.optimStep

        #Buffer
        self.batch_size=opt.batch_size
        self.capacity=opt.capacity
        self.events=Memory(self.capacity,prior=False)

        #Compteur
        self.nbEvents=0

        #Uhlenbeck & Ornstein, 1930
        self.N=Orn_Uhlen(self.n_action,sigma=sigma)
        self.sigma=sigma
        #Initialize target network Q′ and μ′ with weights θQ′ ← θQ, θμ′ ← θμ
        self.model=ActorCritic(self.n_state,self.n_action,self.high,layers=layers,activation=activation,use_batch_norm=use_batch_norm)
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
            obs=torch.as_tensor(obs).unsqueeze(0)
            action=self.model.policy(obs)+self.N.sample()
            #action=self.model.policy(obs)
            #action+=self.sigma*torch.randn(self.n_action)
            self.model.policy.train()
        return torch.clamp(action,min=self.low,max=self.high).squeeze(0).numpy()
    
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
        return self.nbEvents%self.opt.freqOptim==0 #and  self.nbEvents%self.optim_step==0

    def learn(self):
        if self.test:
            pass
        else:
            for _ in range(self.optim_step):
                self.target.policy.eval()
                self.target.q.eval()
                self.model.q.eval()
                _,_,batch = self.events.sample(self.batch_size)
                obs, actions, rewards, next_obs, dones = map(list,zip(*batch))
                obs=torch.FloatTensor(obs)
                actions=torch.FloatTensor(actions)
                rewards=torch.FloatTensor(rewards)
                next_obs=torch.FloatTensor(next_obs)
                dones=torch.FloatTensor(dones)

                #####################################################
                with torch.no_grad():
                    y=rewards+self.gamma*(1-dones)*self.target.q(next_obs,self.target.policy(next_obs))

                q=self.model.q(obs,actions)
                self.model.q.train()
                critic_loss =self.loss(q,y)
                logger.direct_write('Loss/critic',critic_loss,self.nbEvents)
                self.q_optim.zero_grad()
                critic_loss.backward()
                self.q_optim.step()
                #####################################################
                self.model.q.eval()
                mu=self.model.policy(obs)
                self.model.policy.train()
                actor_loss=-self.model.q(obs,mu).mean()
                logger.direct_write('Loss/actor',actor_loss,self.nbEvents)
                self.policy_optim.zero_grad()
                actor_loss.backward()
                self.policy_optim.step()
                #####################################################
                with torch.no_grad():
                    for param, param_target in zip(self.model.parameters(), self.target.parameters()):
                        param_target.data.mul_(self.ru)
                        param_target.data.add_((1 - self.ru) * param.data)
            

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_mountainCar.yaml', "DDPG")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DDPG(env,config,use_batch_norm=True,layers=[64,64,32,32])


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
            

            action = agent.act(ob)
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
                agent.N.reset()
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break

    env.close()

