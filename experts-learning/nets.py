import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from collections import namedtuple
from itertools import count
from IPython.display import clear_output
from torch.autograd import Variable
from tqdm import tnrange


from hyper_parametrs import *


SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from collections import namedtuple
from itertools import count
from IPython.display import clear_output
from torch.autograd import Variable
from tqdm import tnrange


from hyper_parametrs import *


SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])


class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, name, metric_type="no"):
        super(Agent, self).__init__()
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.metric_type = metric_type
        
        #self.affine1 = nn.Linear(state_dim, hidden_layer1_size) #common
        #nn.init.xavier_normal(self.self.affine1.weight) #common
        
        self.common = nn.Linear(state_dim, hidden_layer1_size)
        
        self.action_head = nn.Sequential(nn.Linear(state_dim, hidden_layer1_size), nn.ReLU(),
                                         nn.Linear(hidden_layer1_size, hidden_layer2_size), nn.ReLU(),
                                         #nn.Linear(hidden_layer2_size, hidden_layer3_size), nn.ReLU(),
                                         nn.Linear(hidden_layer2_size, action_dim))
        
        self.value_head = nn.Sequential(nn.Linear(state_dim, hidden_layer1_size), nn.ReLU(),
                                         nn.Linear(hidden_layer1_size, hidden_layer2_size), nn.ReLU(),
                                         #nn.Linear(hidden_layer2_size, hidden_layer3_size), nn.ReLU(),
                                         nn.Linear(hidden_layer2_size, 1))
        
#         self.base_action = nn.Linear(state_dim, hidden_layer1_size) #separate
#         nn.init.xavier_normal(self.base_action.weight) #separate
        
#         self.base_value = nn.Linear(state_dim, hidden_layer1_size) #separate
#         nn.init.xavier_normal(self.base_value.weight) #separate
        
#         self.action_linear = nn.Linear(hidden_layer1_size, hidden_layer2_size)
#         nn.init.xavier_normal(self.action_linear.weight)
#         self.action_head = nn.Linear(hidden_layer2_size, action_dim)
#         nn.init.xavier_normal(self.action_head.weight)
        
#         self.value_linear = nn.Linear(hidden_layer1_size, hidden_layer2_size)
#         nn.init.xavier_normal(self.value_linear.weight)
#         self.value_head = nn.Linear(hidden_layer2_size, 1)
#         nn.init.xavier_normal(self.value_head.weight)

        self.saved_actions = []
        self.rewards = []

    def forward(self, s, s_star=None):
        x_onehot = torch.zeros((1, self.state_dim))
        x_onehot[0][s] = 1
        if(not (s_star is None)):
            x_onehot[0][self.state_dim // 2 + s_star] = 1
        x_onehot = Variable(x_onehot)
        
        #x = F.relu(self.affine1(x_onehot)) #common
#         x_action = F.relu(self.base_action(x_onehot))
#         x_value = F.relu(self.base_value(x_onehot))
        
#         action_scores = self.action_head(F.relu(self.action_linear(x_action)))
#         state_values = self.value_head(F.relu(self.value_linear(x_value)))
        action_scores = self.action_head(x_onehot)
        state_values = self.value_head(x_onehot)
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
                if (s + 1) % (int(np.sqrt(env.nS))) == 0:
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


class NovelExpert:
    def __init__(self, agent_type, action_probs, v_function, goal_map):
        self.type = agent_type
        self.action_probs = action_probs
        self.v_function = v_function
        self.goal_map = goal_map

    def get_action_probs(self, env, s):
        return self.action_probs[s]
    
    def select_action(self, env, s):
        action_probs = self.action_probs[s]
        action = np.random.choice(np.arange(5), p=action_probs)
        
        return action
    
    def get_goal(self, s):
        return self.goal_map[s]

    def get_v_function(self, s):
        return self.v_function[s]

    
feature_extractor_layer1_size = 100
feature_extractor_layer2_size = 100
similarity_layer1_size = 100

class SiameseNet(nn.Module):
    def __init__(self, state_dim):
        super(SiameseNet, self).__init__()
        
        self.state_dim = state_dim
        
        self.feature_extractor = nn.Sequential(nn.Linear(state_dim, feature_extractor_layer1_size), nn.ReLU(),
                                               nn.Linear(feature_extractor_layer1_size, feature_extractor_layer2_size), nn.ReLU()
                                               )
        self.similarity = nn.Sequential(nn.Linear(2 * feature_extractor_layer2_size, similarity_layer1_size), nn.ReLU(),
                                               nn.Linear(similarity_layer1_size, 1), nn.Sigmoid()
                                               )
        
    def to_one_hot(self, states):
        states_one_hot = np.zeros((len(states), self.state_dim))
        for i in range(len(states)):
            states_one_hot[i][states[i]] = 1

        return Variable(torch.FloatTensor(states_one_hot), requires_grad=False)
        
    def forward(self, X):
        s1_one_hot = self.to_one_hot(X[:, 0])
        s2_one_hot = self.to_one_hot(X[:, 1])
        s1_features = self.feature_extractor(s1_one_hot)
        s2_features = self.feature_extractor(s2_one_hot)
        return self.similarity(torch.cat([s1_features, s2_features], dim=-1))