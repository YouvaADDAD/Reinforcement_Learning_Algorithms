import torch
from torch import nn
from torch.distributions import Categorical
from models import NN
import datetime
import yaml
import gym
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils import *
from Expert_Load import *
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BehavioralCloning(nn.Module):
    def __init__(self, env, opt):
        super(BehavioralCloning, self).__init__()
        self.env = env
        self.opt = opt
        self.n_state = env.observation_space.shape[0]
        self.n_action  = env.action_space.n
        self.dataset    = DatasetExpertPickled(self.n_state, self.n_action, "./expert.pkl", device)
        self.buffer     = DataLoader(self.dataset, batch_size = opt.batchsize)
        self.test = False
        self.nbEvents = 0
        self.policy = NN(self.n_state, self.n_action, layers = opt.layers, activation=opt.activation, finalActivation = opt.finalActivation)
        self.pi_optim = torch.optim.Adam(self.policy.parameters(),lr=opt.lr)
            
    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass
    
    # Choisir une action avec Categorical
    def act(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs)
            probs = self.policy(obs)
            dist = Categorical(probs)      
            return dist.sample().item()

    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents +=1
        return self.nbEvents % self.opt.freqOptim ==0
        
    def learn(self):
        if self.test:
            pass
        else:
            print("#"*20)
            states, actions = next(iter(self.buffer))
            states = states.to(device)
            actions = actions.to(device)
            new_actions = self.policy(states)
            new_actions = torch.log(new_actions[actions.bool()])
            actor_loss = - torch.mean(new_actions)
            self.pi_optim.zero_grad()
            actor_loss.backward()
            self.pi_optim.step()



if __name__=='__main__':
    #------------------------------------------------------------------------------------------
    env, config, outdir, logger = init('./config/config_BC.yaml', "BC_Expert")
    #------------------------------------------------------------------------------------------    
    np.random.seed(config.seed)
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config.seed)
    episode_count = config["nbEpisodes"]
    #------------------------------------------------------------------------------------------
    agent = BehavioralCloning(env, config).to(device)
    #------------------------------------------------------------------------------------------
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    #------------------------------------------------------------------------------------------
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

        if verbose:
            env.render()

        new_ob = ob

        j=0 
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)

            j+=1
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