import torch
import pickle
from torch.utils.data import Dataset

class DatasetExpertPickled(Dataset):
    def __init__(self, nbFeatures, nbActions, path, device):
        self.path = path
        self.floatTensor = torch.zeros(1, dtype=torch.float, device=device)
        self.longTensor  = torch.zeros(1, dtype=torch.long, device=device)
        self.nbFeatures  = nbFeatures
        self.nbActions   = nbActions
        self.loadExpertTransitions(path)
    
    def __getitem__(self, idx):
        return (self.expert_states[idx], self.expert_actions[idx])
            
    def __len__(self):
        return self.expert_states.shape[0]
    
    def loadExpertTransitions(self, file):
        with open(file, "rb") as handle:
            expert_data = pickle.load(handle).to(self.floatTensor)
            expert_states = expert_data[:,:self.nbFeatures]
            expert_actions = expert_data[:,self.nbFeatures:]
            self.expert_states = expert_states.contiguous()
            self.expert_actions = expert_actions.contiguous()
                        
    def toOneHot(self, actions):
        actions = actions.view(-1).to(self.longTensor)
        oneHot = torch.zeros(actions.size()[0], self.nbActions).to(self.floatTensor)
        oneHot[range(actions.size()[0]), actions] = 1
        return oneHot
    

    def toIndexAction(self, oneHot):
        ac = self.longTensor.new(range(self.nb_actions)).view(1, -1)
        ac = ac.expand(oneHot.size()[0], -1).contiguous().view(-1)
        actions = ac[oneHot.view(-1) > 0].view(-1)
        return actions