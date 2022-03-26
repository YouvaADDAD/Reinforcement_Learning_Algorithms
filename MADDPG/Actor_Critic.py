import torch.nn as nn
import torch

#########Creation des modéles afin de de pouvoir rajouter la Batch Normalization
class Actor(nn.Module):
    def __init__(self,n_state,n_action,act_limit=1,layers=[30,30],activation=nn.ReLU,finalActivation=None,dropout=0.0,use_batch_norm=False):
        super(Actor,self).__init__()
        self.n_state=n_state
        self.n_action=n_action
        self.act_limit=act_limit  #-> No need here since we have actions between -1 and 1
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
        if finalActivation:
            layer.append(finalActivation())
        self.actor=nn.Sequential(*layer)
        ##################################################

    def forward(self,obs):
        #Current observation for one agent
        #No need for act limit action between -1 and 1
        action = self.act_limit * self.actor(obs)
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
