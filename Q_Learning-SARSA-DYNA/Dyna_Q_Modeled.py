import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *
import random
import torch.utils.tensorboard

class DynaQ(object):

    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.n_plan=opt.n_plan
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.max_explo=opt.explo
        self.min_explo=0.0001
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.modelSamples=opt.nbModelSamples
        self.decay=opt.decay
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.model = {}
        self.P={}
        self.R={}
        self.nbEvents=0
        self.alpha_R=0.5

    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self,obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss



    def act(self, obs):
        #TODO remplacer par action QLearning
        if self.exploMode==0 :           #epsilon greedy
            if np.random.uniform() < self.explo:
                return self.action_space.sample()
            else:
                return np.argmax(self.values[obs])

        return self.action_space.sample()
    
    def decay_eps(self):
        self.explo*=self.decay
        self.alpha_R*=self.decay

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done
        self.model[(ob,action)]=(reward,new_ob)
       
        if (ob, action, new_ob) not in self.R.keys():
            self.R[(ob, action, new_ob)] = 0
            self.P[(ob, action, new_ob)] = 0




    def learn(self, done):
        #TODO
        
        state, action, reward, next_state,done = self.last_source,self.last_action,self.last_reward,self.last_dest,self.last_done
        ############UPDATE Q###########
        self.values[state][action]+=self.alpha*(reward + self.discount * (1-done) * np.max(self.values[next_state]) - self.values[state][action])
        ############UPDATE P & R ###########
        self.R[(state, action,next_state)] += self.alpha_R * (reward - self.R[(state, action,next_state)])
        self.P[(state, action,next_state)] += self.alpha_R * (1 - self.P[(state, action,next_state)])
        for ob,move,new_ob in self.P.keys():
            if new_ob !=next_state and ob==state and move==action:
                self.P[(ob, move,new_ob)] +=self.alpha_R * (-self.P[(ob,move,new_ob)] )


    def plan(self):
        for _ in range(self.n_plan):
            state,action=random.choice(list(self.model.keys()))
            somme_over=np.sum([self.P.get((state,action,next_state),0) * ((self.R.get((state,action,next_state),0))+self.discount*np.max(self.values[next_state])) for next_state in self.qstates.values()])
            self.values[state][action]+=self.alpha*(somme_over-self.values[state][action])

if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"DynaQ_plan_5")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    agent = DynaQ(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        ob = env.reset()

        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = False
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.nbEvents+=1
            agent.learn(done)
            agent.plan()
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                agent.decay_eps()
                break

    env.close()