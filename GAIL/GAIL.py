
import torch
from torch import cuda, nn
import datetime
import yaml
import gym
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils import *
from Expert_Load import *
from Behavioral_Cloning import *
import numpy as np
from models import *
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class GAIL(object):
    def __init__(self, env,opt):
        super(GAIL,self).__init__()
        self.env = env
        self.opt = opt
        self.n_state = self.env.observation_space.shape[0]
        self.n_action = env.action_space.n
        self.test = False
        self.sigma_noise = opt.sigma_noise

        #Parameters
        self.gamma = opt.gamma
        self.Lambda = opt.Lambda
        self.rhu = opt.rhu
        self.numberOptimAC = opt.numberOptimAC #Nombre d'optimisation pour le PPO
        self.numberOptimDiscriminator = opt.numberOptimDiscriminator
        self.eps_clip = opt.eps_clip #Pour le PPO
        self.reg_entropy = opt.reg_entropy 

        #Memory
        self.events=Buffer() 
        self.expert_data = DatasetExpertPickled(self.n_state, self.n_action, opt.path,device=device)
        drop_last = len(self.expert_data) > opt.batchsize
        self.batchsize = len(self.expert_data) if opt.batchsize < 0 else opt.batchsize
        self.dataloader_expert = DataLoader(self.expert_data, shuffle = True, batch_size = self.batchsize, drop_last = drop_last)
        self.nbEvents=0

        #Models
        self.discriminator=Discriminator(self.n_state,self.n_action,layers=opt.discriminator_layers,activation=opt.discriminator_activation, finalActivation = opt.discriminator_factivation, sigma_noise=self.sigma_noise)
        self.ac=ActorCritic(self.n_state,self.n_action,layers=opt.ac_layers,activation=opt.ac_activation)

        #Optimizer
        self.optimizerDisc = torch.optim.AdamW(self.discriminator.parameters(),lr=opt.lr_disc, weight_decay=0.001)
        self.optimizerPolicy = torch.optim.AdamW(self.ac.actor.parameters(),lr=opt.lr_pi, weight_decay=0.001)
        self.optimizerCritic = torch.optim.AdamW(self.ac.critic.parameters(),lr=opt.lr_q, weight_decay=0.001)

        #Loss Fonction
        self.disc_loss=torch.nn.BCEWithLogitsLoss()
        self.critic_loss=torch.nn.SmoothL1Loss()
    
    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    #Renvoie l'action, la valeur, la logprob, et le rewardDisc    
    def act(self,obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs)
            value, dist = self.ac(obs)
            action = dist.sample()
        return action.item(), value, dist.log_prob(action)

    def store(self,ob, action, reward, done, value, log_prob, it): 
        if not self.test:
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            self.events.store(ob, action, reward, done, value, log_prob)
    
    # retoune vrai si c'est le moment d'entraîner l'agent.
    #a Chaque fin de trajectoire ou aprés avoir atteint maxLenght
    def timeToLearn(self, done):
        if self.test:
            return False
        self.nbEvents += 1
        return done
        #return self.nbEvents % self.opt.freqOptim == 0 and  self.nbEvents > self.opt.startEvents

    def returns_and_advantages(self, rewards, dones, values):
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            returns_to_go = 0.
            next_value = 0.
            advantages_to_go = 0.
            for step in reversed(range(len(rewards))):
                returns_to_go = rewards[step] + (self.gamma * returns_to_go * (1 - dones[step]))
                returns[step] = returns_to_go
                TD = rewards[step] + (self.gamma * next_value * (1 - dones[step])) - values[step]
                next_value = values[step]
                advantages_to_go = TD + (self.gamma * self.Lambda * advantages_to_go * (1 - dones[step]))
                advantages[step] = advantages_to_go
            advantages = (advantages - advantages.mean()) / advantages.std()
            return returns, advantages

    def sample_batch(self, batchsize):
        #####################################################################################################################
        old_states, actions, rewards, dones, old_values, old_logprobs = self.events.sample()
        old_states = torch.FloatTensor(old_states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        old_values = torch.FloatTensor(old_values)
        old_logprobs = torch.FloatTensor(old_logprobs)
        #####################################################################################################################
        batchsize = batchsize if len(old_states) > batchsize else len(old_states)
        self.dataloader_expert = DataLoader(self.expert_data, shuffle = True, batch_size = batchsize)
        #####################################################################################################################
        one_hot_actions = self.expert_data.toOneHot(actions)
        reward_discriminator = - F.logsigmoid(1 - self.discriminator(old_states, one_hot_actions)).squeeze(dim = -1)
        reward_discriminator = reward_discriminator.clip(0., self.opt.maxReward)
        returns, advantages = self.returns_and_advantages(reward_discriminator, dones, old_values)
        returns = returns
        indices_batch = torch.randperm(len(old_states))[:batchsize]
        #####################################################################################################################
        return old_states, actions, reward_discriminator, dones, old_values, old_logprobs, advantages, one_hot_actions, returns, indices_batch

    def learn(self):
        if self.test:
            pass  
        else:
            #####################################################################################################################
            expert_states, expert_actions = next(iter(self.dataloader_expert))
            old_states, actions, _, _, _, old_logprobs, advantages, one_hot_actions, returns, indices_batch = self.sample_batch(expert_states.shape[0])
            #####################################################################################################################
            for _ in range(self.numberOptimDiscriminator):
                learner = self.discriminator(old_states[indices_batch], one_hot_actions[indices_batch])
                expert = self.discriminator(expert_states, expert_actions)
                real_label = torch.ones(expert_states.shape[0], 1)
                fake_label = torch.zeros(old_states[indices_batch].shape[0], 1)
                discrim_loss = (self.disc_loss(learner, fake_label) + self.disc_loss(expert, real_label)) / 2
                self.optimizerDisc.zero_grad()
                discrim_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 50)
                self.optimizerDisc.step()
            ################################################################
            for _ in range(self.numberOptimAC):
                new_values,new_policy_dist = self.ac(old_states)    
                new_logprobs = new_policy_dist.log_prob(actions)
                new_values = new_values.squeeze(-1) 
                ##############################################################################
                ratio=torch.exp(new_logprobs-old_logprobs)
                ############################################################################## 
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                ##############################################################################
                actor_loss = -torch.min(policy_loss_1,policy_loss_2).mean()
                H =  torch.mean((new_policy_dist.probs*torch.log(new_policy_dist.probs)).mean(dim=-1))
                actor_loss = actor_loss - self.reg_entropy * H
                self.optimizerPolicy.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 10)#10
                self.optimizerPolicy.step()  
            ################################################################
            for _ in range(self.numberOptimAC):
                new_values,new_policy_dist = self.ac(old_states)
                new_values = new_values.squeeze(-1) 
                critic_loss = self.critic_loss(returns, new_values)
                self.optimizerCritic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 10)#10
                self.optimizerCritic.step()
            self.events.reset()

            
       

        
           
if __name__=='__main__':
    env, config, outdir, logger = init('./config/config_gail.yaml', "GAIL")
    ################################################################################################################
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]
    ################################################################################################################
    agent = GAIL(env,config)
    ################################################################################################################
    verbose=True
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    agent.nbEvents = 0
    ################################################################################################################
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0

        obs = env.reset()
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


        new_obs = obs

        while True:

            if verbose:
                env.render()

            obs = new_obs
            action, value, log_prob= agent.act(obs)
            new_obs, rewards, done, _ = env.step(action)
           

            j += 1


            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")


            if (not agent.test):

                agent.store(obs, action, rewards, done, value, log_prob, j)

                if agent.timeToLearn(done):
                    agent.learn()

            rsum += rewards

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0
                break

    env.close()
    





















"""if __name__ == '__main__':
    #-----------------------------------------------------------------------
    env, config, outdir, logger = init('./config/config_gail.yaml', "GAIL")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]
    #-----------------------------------------------------------------------
    agent = GAIL(env,config)
    #-----------------------------------------------------------------------
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    #-----------------------------------------------------------------------
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

        new_ob = ob

        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
        

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action,reward,done,j)
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
"""