import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

def read_ads(filename):
    file=open(filename,'r')
    lines=file.readlines()
    T=[]
    feautures=[]
    taux_clics=[]
    for line in lines: 
        t,features,taux=line.split(':')
        feature=np.array(features.split(';'),dtype=float)
        taux=np.array(taux.split(';'),dtype=float)
        T.append(t)
        feautures.append(feature)
        taux_clics.append(taux)
    return np.array(T,dtype=int),np.array(feautures),np.array(taux_clics)

#data= only taux
#Baseline 
def random_strategy(data):
    return np.cumsum(list(map(lambda x: np.random.choice(x),data)))

def staticBase_strategy(data):
    index=data.sum(0).argmax()
    return np.cumsum(list(map(lambda x: x[index],data)))

def optimale_strategy(data):
    return np.cumsum(list(map(lambda x: x.max(),data)))


class UCB(object):
    def __init__(self,taux_de_clics,confidence=1.0):
        self.c=confidence
        self.T,self.nbAnnonceur=taux_de_clics.shape
        self.data=taux_de_clics
        self.t=0
        self.quality = np.zeros((self.nbAnnonceur))
        self.counter = np.ones((self.nbAnnonceur))
        
        
    def act(self):
        self.t += 1
        action = np.argmax(self.quality + self.c * np.sqrt(2 * np.log(self.t) / (self.counter)))
        return action
    
    def learn(self):
        
        rewards=[]
        for t in range(1, self.T + 1):
            action=self.act()
            reward=self.data[t-1][action]
            self.quality[action] += (reward-self.quality[action]) / (self.counter[action])
            self.counter[action] += 1
            rewards.append(reward)
        return np.cumsum(rewards)

def LinUCB(features,data,alpha=0.1):
    T=len(data)
    nbA=len(data[0])
    dim=len(features[0])
    A=np.array([np.identity(dim) for _ in range(nbA)])
    b=np.zeros((nbA,dim,1)) 
    rewards=[]
    for t in range(T):
        x_t=features[t].reshape((dim,1))
        P=[]
        for a in range(nbA):
            rev_A=np.linalg.inv(A[a])
            theta=np.dot(rev_A,b[a])
            P.append(np.dot(theta.T,x_t)+alpha*np.sqrt((np.dot(np.dot(x_t.T,rev_A),x_t))))
        max_arms=np.argmax(P)
        reward=data[t,max_arms]
        rewards.append(reward)
        A[max_arms]+=np.dot(x_t,x_t.T)
        b[max_arms]+=reward*x_t
    return np.cumsum(rewards)