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
from torch.nn import init as Init
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init= 0.017, bias = True):
        super(NoisyLinear,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias
        
        #Init mu and sigma weight
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        
        #Init bias mu and sigma
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        #Enregister les epsilon comme etant pas defferentiable
        self.register_buffer("weight_epsilon", torch.FloatTensor((out_features, in_features)))
        if bias:
            self.register_buffer("bias_epsilon", torch.FloatTensor((out_features)))
        ########################################################################
        bound=np.sqrt(3/self.in_features)
        Init.uniform_(self.weight_mu ,-bound, bound)
        if self.bias_mu is not None:
            Init.uniform_(self.weight_mu ,-bound, bound)
        ########################################################################
        self.weight_sigma.data.fill_(sigma_init)
        if self.bias_sigma is not None:
            self.bias_sigma.data.fill_(sigma_init)
        self._noise()

    def _noise(self):
        self.weight_epsilon = torch.randn(size=(self.out_features,self.in_features))
        if self.bias:
            self.bias_epsilon = torch.randn(size=[self.out_features])
    
    def forward(self,input):
        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        return F.linear(input, weight, bias)

class NoisyNN(nn.Module):
    def __init__(self, inSize, outSize,layers=[200], activation=nn.Tanh,dropout=0.0):
        super(NoisyNN,self).__init__()
        self.layers=nn.ModuleList([])
        for x in layers:
            self.layers.append(NoisyLinear(inSize,x))
            self.layers.append(activation())
            if dropout>0.0:
                self.layers.append(nn.Dropout(dropout))
            inSize=x
        self.layers.append(NoisyLinear(inSize,outSize))
        self.model=nn.Sequential(*self.layers)

    def reset_noise(self):
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer._noise()
    
    def forward(self,input):
        return self.model(input)

class NoisyDQN(object):
    def __init__(self, env,opt,layers=[200],activation=nn.Tanh,dropout=0.0):
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
        self.prior=opt.prior

        #Compteur
        self.nbEvents=0
        self.events=Memory(self.capacity,prior=self.prior)

        #Model
        self.Q=NoisyNN(self.env.observation_space.shape[0],self.action_space.n , layers=layers, activation=activation,dropout=dropout)
        self.Qhat=deepcopy(self.Q)
        self.loss=nn.SmoothL1Loss()
        self.optim=torch.optim.Adam(self.Q.layers.parameters(),weight_decay=0.0,lr=opt.lr)


    def act(self, obs):
        with torch.no_grad():
            obs=torch.FloatTensor(obs)
            return self.Q(obs).argmax(dim=-1).item()
    
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
                qhat=self.Qhat(ls1)
                #Prendre les valeurs max
                vmax=qhat.max(dim=-1)[0]
                y=lr + self.gamma * vmax * (1-ld)
            ###########################################################################
            error = self.loss(q,y)
            self.optim.zero_grad()
            error.backward()
            self.optim.step()
            ###########################################################################
            self.Q.reset_noise()
            self.Qhat.reset_noise()
            if self.nbEvents%self.C==0:
                self.Qhat.load_state_dict(self.Q.state_dict())
            
   
if __name__ == '__main__':

    env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "Noisy_DQN_lr_0.0003")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = NoisyDQN(env,config,activation=nn.Tanh,dropout=0.0)


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

            if done :
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if not agent.test:
                    logger.direct_write("rewardTrain", rsum, i)
                mean += rsum
                rsum = 0
                break
    env.close()

    

        

        




    

