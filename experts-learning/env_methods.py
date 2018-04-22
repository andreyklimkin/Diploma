import numpy as np

def set_random_s0(env):
    rand_s0 = np.random.choice(np.array(np.where(env.grid_map)).flatten())
    #print(rand_s0)
    env.isd = np.zeros(env.shape[0] * env.shape[1])
    env.isd[rand_s0] = 1 

def check_equivalence(s1, s2, eps=1e-9):
    return np.sum((s1 - s2) ** 2) < eps   
    
def is_terminal(env, s):
    return s == 0 or s == env.nS - 1