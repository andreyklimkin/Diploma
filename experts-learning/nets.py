import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


from itertools import count
from collections import namedtuple
from torch.autograd import Variable


SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])


hidden_layer1_size = 50
hidden_layer2_size = 50
hidden_layer3_size = 50

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, name):
        super(Agent, self).__init__()
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.affine1 = nn.Linear(state_dim, hidden_layer1_size)
        nn.init.xavier_normal(self.affine1.weight)
        
        self.action_linear = nn.Linear(hidden_layer2_size, hidden_layer3_size)
        nn.init.xavier_normal(self.action_linear.weight)
        self.action_head = nn.Linear(hidden_layer3_size, action_dim)
        nn.init.xavier_normal(self.action_head.weight)
        
        self.value_linear = nn.Linear(hidden_layer2_size, hidden_layer3_size)
        nn.init.xavier_normal(self.value_linear.weight)
        self.value_head = nn.Linear(hidden_layer3_size, 1)
        nn.init.xavier_normal(self.value_head.weight)

        self.saved_actions = []
        self.rewards = []

    def forward(self, s, s_star=None):
        x_onehot = torch.zeros((1, self.state_dim))
        x_onehot[0][s] = 1
        if(not (s_star is None)):
            x_onehot[0][self.state_dim // 2 + s_star] = 1
        x_onehot = Variable(x_onehot)
        x = F.relu(self.affine1(x_onehot))
        
        action_scores = self.action_head(F.relu(self.action_linear(x)))
        state_values = self.value_head(F.relu(self.value_linear(x)))
        return F.softmax(action_scores), state_values
    
    def select_action(self, state, s_star, cache_action=True):
        #state = torch.from_numpy(state).int().unsqueeze(0)
        probs, state_value = self.forward(state, s_star)
        action = probs.multinomial()
        if(cache_action == True):
            self.saved_actions.append(SavedAction(action, probs, state_value))
        return action.data
        
        
class Expert:
    def __init__(self, Alice_type):
        self.type = Alice_type
    
    def get_action_probs(self, env, s):
        if(self.type == 'left_top_corner'):
            if(s == 0):
                return np.array([0, 0, 0, 0, 1])
            if(s % int(np.sqrt(env.nS)) == 0):
                return np.array([1, 0, 0, 0, 0])
            if(s < int(np.sqrt(env.nS))):
                return np.array([0, 0, 0, 1, 0])
            return np.array([0.5, 0, 0, 0.5, 0])
        if(self.type == 'right_bottom_corner'):
            if(s == env.nS - 1):
                return np.array([0, 0, 0, 0, 1])
            if((s + 1) % int(np.sqrt(env.nS)) == 0):
                return np.array([0, 0, 1, 0, 0])
            if(s + int(np.sqrt(env.nS)) >= env.nS):
                return np.array([0, 1, 0, 0, 0])
            return np.array([0, 0.5, 0.5, 0, 0])
        if(self.type == 'global_optimal'):
            if (s // np.sqrt(env.nS) <= int(np.sqrt(env.nS)) - (s % int(np.sqrt(env.nS))) - 1):
                if(s == 0):
                    return np.array([0, 0, 0, 0, 1])
                if s % int(np.sqrt(env.nS)) == 0:
                    return np.array([1, 0, 0, 0, 0])  
                if s < int(np.sqrt(env.nS)):
                    return np.array([0, 0, 0, 1, 0]) 
                return np.array([0.5, 0, 0, 0.5, 0]) 
            else:
                if(s == env.nS - 1):
                    return np.array([0, 0, 0, 0, 1])
                if s % (int(np.sqrt(env.nS)) - 1) == 0:
                    return np.array([0, 0, 1, 0, 0])  
                if(s + int(np.sqrt(env.nS)) >= env.nS):
                    return np.array([0, 1, 0, 0, 0]) 
                return np.array([0, 0.5, 0.5, 0, 0]) 

    def select_action(self, env, s):
        action_probs = self.get_action_probs(env, s)
        action = np.random.choice(np.arange(5), p=action_probs)
        
        return action

    def precompute_v_function(self, env):
        self.v = np.zeros(env.nS)
        for s in range(len(self.v)):
            if(self.type == 'left_top_corner'):
                self.v[s] = -(s % int(np.sqrt(env.nS)) + s // int(np.sqrt(env.nS)))
            if(self.type == 'right_bottom_corner'):
                self.v[s] = (-((int(np.sqrt(env.nS)) - (s % int(np.sqrt(env.nS))) - 1) 
                        + (int(np.sqrt(env.nS)) - s // int(np.sqrt(env.nS)) - 1)))
            if(self.type == 'global_optimal'):
                left_top = s % int(np.sqrt(env.nS)) + s // int(np.sqrt(env.nS))
                right_bottom = ((int(np.sqrt(env.nS)) - (s % int(np.sqrt(env.nS))) - 1) 
                        + (int(np.sqrt(env.nS)) - s // int(np.sqrt(env.nS)) - 1))
                self.v[s] = -min(left_top, right_bottom)
    
    def get_goal(self, env, s):
        if(self.type == 'left_top_corner'):
            return 0
        if(self.type == 'right_bottom_corner'):
            return env.nS - 1 
        if(self.type == 'global_optimal'):
            if (s // np.sqrt(env.nS) <= int(np.sqrt(env.nS)) - (s % int(np.sqrt(env.nS))) - 1):
                return 0
            if (s // np.sqrt(env.nS) > int(np.sqrt(env.nS)) - (s % int(np.sqrt(env.nS))) - 1):
                return env.nS - 1

    def v_function(self, s):
        return self.v[s]