import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from randomAgent import RandomAgent


class ValueIteration(RandomAgent):
    def __init__(self, env,epsilon=1e-3,gamma=0.99):
        super(ValueIteration,self).__init__(env.action_space)
        self.env = env
        self.epsilon=epsilon
        self.gamma=gamma
        self.states,self.mdp=env.getMDP()
        self.n_action=self.env.action_space.n
        self.actions=self.env.action_space
        self.n_states=len(self.states)
        self.policy = np.array([self.actions.sample() for _ in range(self.n_states)])
        self.value  = np.zeros(self.n_states)
    
    def act(self, state):
        return self.policy[state]
    

    def learn(self):
        value_new=np.zeros(self.n_states)
        while True:
            for s in self.mdp.keys():
                value_new[s]=np.max([np.sum([proba * (reward+self.gamma*self.value[next_state])  for proba,next_state,reward,_ in  self.mdp[s].get(action)]) for action in range(self.n_action)])
            if np.linalg.norm(self.value-value_new)<=self.epsilon:
                break
            self.value=copy.deepcopy(value_new)
        
        for s in self.mdp.keys():
            self.policy[s]=np.argmax([np.sum([proba * (reward+self.gamma*self.value[next_state])  for proba,next_state,reward,_ in self.mdp[s].get(action)]) for action in range(self.n_action)])


if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan1.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.reset()
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human") #visualisation sur la console
    states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    print("Nombre d'etats : ",len(states))
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}
    # Execution avec un Agent
    agent = ValueIteration(env)
    agent.learn()

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            #env.render()
            pass
        j = 0
        rsum = 0
        while True:
            action = agent.act(env.getStateFromObs(obs))
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                #env.render()
                pass
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()



    
    