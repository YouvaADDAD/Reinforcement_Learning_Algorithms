import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from randomAgent import RandomAgent


class PolicyIteration(RandomAgent):
    def __init__(self, env,epsilon=1e-3,gamma=0.99):
        super(PolicyIteration,self).__init__(env.action_space)
        self.env = env
        self.epsilon=epsilon
        self.gamma=gamma
        self.states,self.mdp=env.getMDP()
        self.n_action=self.env.action_space.n
        self.actions=self.env.action_space
        self.n_states=len(self.states)
        self.policy = np.array([self.actions.sample() for _ in range(self.n_states)])
        self.value  = np.zeros(self.n_states)

    
    def act(self, observation):
         return self.policy[observation]

    def learn(self):
        while True:
            value_new = np.zeros(self.n_states)
            while True :
                for s in self.mdp.keys():
                    action=self.policy[s]
                    value_new[s]=np.sum([proba * (reward+self.gamma*self.value[next_state])  for proba,next_state,reward,_ in  self.mdp[s].get(action)]) 
                if np.linalg.norm(self.value-value_new)<=self.epsilon:
                    break
                self.value=copy.deepcopy(value_new)
            
            policy_new=np.zeros(self.n_states,dtype=int)
            for s in self.mdp.keys():
                policy_new[s]=np.argmax([np.sum([proba_trans * (reward+self.gamma*self.value[s_prime])  for proba_trans,s_prime,reward,_ in self.mdp[s].get(a)]) for a in range(self.n_action)])

            if np.all(policy_new ==self.policy) :
                break

            self.policy=copy.deepcopy(policy_new)

                


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
    agent = PolicyIteration(env)
    agent.learn()

    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            #env.render(mode='rgb_array')
            pass
        j = 0
        rsum = 0
        while True:
            action = agent.act(env.getStateFromObs(obs))
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                #env.render(mode='rgb_array')
                pass
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()