import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from memory import Memory
from randomAgent import *
from core import *
import torch.nn as nn
from copy import deepcopy
import torch

class DQN(object):
    def __init__(self, env,opt,layers=[200],activation=torch.tanh,dropout=0.0):
        super().__init__()
        #Les hyperparametres
        self.opt=opt
        self.env=env
        self.C=opt.C
        self.gamma=opt.gamma
        self.batch_size=opt.batch_size
        self.capacity=opt.capacity
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.use_target=opt.target
        self.prior=opt.prior

        #Exploration
        self.decay=opt.decay
        self.explo=opt.explo
        self.max_explo=opt.explo
        self.min_explo=0.0001

        #Compteur
        self.nbEvents=0
        self.events=Memory(self.capacity,prior=self.prior)

        #Model
        self.Q=NN(self.env.observation_space.shape[0],self.action_space.n , layers=layers, activation=activation,dropout=dropout)
        self.Qhat=deepcopy(self.Q)
        self.loss=nn.SmoothL1Loss()
        self.optim=torch.optim.Adam(self.Q.layers.parameters(),weight_decay=0.0,lr=opt.lr)


    def act(self, obs):
        with torch.no_grad():
            n = np.random.random()
            if (n < self.explo):
                return self.action_space.sample()
            else:
                obs=torch.FloatTensor(obs)
                return self.Q(obs).argmax(dim=-1).item()
    

    def decays_eps(self,):
        #self.explo=np.clip(self.explo,self.min_explo,self.max_explo)
        #self.explo=max(self.max_explo * np.exp(-t * self.decay),self.min_explo)
        self.explo=max(self.explo*self.decay,self.min_explo)


    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.events.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0
    
        # sauvegarde du modèle
    
    def save(self,outputDir):
        pass 
    # chargement du modèle.
    def load(self,inputDir):
        pass

    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        else:
            #Mini Batch 
            _,_,l = self.events.sample(self.batch_size)
            ls0, la, lr, ls1, ld=map(list,zip(*l))
            ls0= torch.FloatTensor(ls0).view(self.batch_size,-1)
            la = torch.LongTensor(la).view(-1)
            lr = torch.FloatTensor(lr).view(-1)
            ls1= torch.FloatTensor(ls1).view(self.batch_size,-1)
            ld = torch.FloatTensor(ld).view(-1)
            ###########################################################################
            q=self.Q(ls0)[range(self.batch_size),la]
            with torch.no_grad():
                if self.use_target:
                    qhat=self.Qhat(ls1)
                else:
                    qhat=self.Q(ls1)
                #Prendre les valeurs max
                vmax=qhat.max(dim=-1)[0]
                y=lr + self.gamma * vmax * (1-ld)
            ###########################################################################
            error = self.loss(q,y)
            self.optim.zero_grad()
            error.backward()
            self.optim.step()
            ###########################################################################
            if self.nbEvents%self.C==0:
                self.Qhat.load_state_dict(self.Q.state_dict())
            
   
if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "DQN")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = DQN(env,config,activation=torch.tanh,dropout=0.0)


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
            action= agent.act(ob)
            
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward
            if agent.timeToLearn(done):
                agent.learn()
            
            #Decay a chaque evenement 
            if not agent.test:
                agent.decays_eps()

            if done :
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if not agent.test:
                    logger.direct_write("rewardTrain", rsum, i)
                mean += rsum
                rsum = 0
                break

    env.close()

    
    

