import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(color_codes=True)

from itertools import count
from collections import namedtuple
from collections import defaultdict
from env_methods import *
from hyper_parametrs import *
from PIL import Image


def draw_direction_probs(ax, env, s, p, scale_arrow=0.5, arrow_width=3, arrow_color='c', eps=1e-5):
    i = (s) // env.shape[1]
    j = (s) % env.shape[1]
    
    ax.arrow(j + 0.5, i + 0.5, p[1] * scale_arrow + eps, 0, linewidth=arrow_width, color=arrow_color, length_includes_head=True)
    ax.arrow(j + 0.5, i + 0.5, 0, p[2] * scale_arrow + eps, linewidth=arrow_width, color=arrow_color, length_includes_head=True)
    ax.arrow(j + 0.5, i + 0.5, -p[3] * scale_arrow + eps, 0, linewidth=arrow_width, color=arrow_color, length_includes_head=True)
    ax.arrow(j + 0.5, i + 0.5, 0, -p[0] * scale_arrow + eps, linewidth=arrow_width, color=arrow_color, length_includes_head=True)

    
def draw_value_anotate(ax, env, s, value, font_color='c'):
    i = (s) // env.shape[1]
    j = (s) % env.shape[1]
    
    ax.annotate("%.2f" % value, xy=(j-0.3, i), color='c', fontsize=11)

def set_env_s0(env, s0):
    env.isd = np.zeros(env.shape[0] * env.shape[1])
    env.isd[s0] = 1 
    

def get_episode_reward(env, model, who, s0, tmax, possbile_s_stars):
    policy_episode_reward = 0.0
    set_env_s0(env, s0)
    env.reset()
    s = env.s
    if who != "Expert":
        max_v = -np.inf
        best_s_star = None
        for s_star in possbile_s_stars:
            #print("S_star = {}".format(s_star))
            #print("Possible S_Stars = {}".format(possbile_s_stars))
            v_s_star = model(s, s_star)[1][0][0].data.numpy()[0]
            #print(v_s_star)
            if v_s_star > max_v:
                best_s_star = s_star
                max_v = v_s_star

        s_star = best_s_star
    
    gamma_factor = 1
    for i in range(tmax):
        if(is_terminal(env, s)):
            break
        if(who == "Expert"):
            a = model.select_action(env, s)
        else:
            a = model.select_action(s, s_star, cache_action=False)[0, 0]
        s, reward, _, _= env.step(a)
        policy_episode_reward += reward * gamma_factor
        gamma_factor *= gamma_rl
    
    return policy_episode_reward
        
        
def get_policy_reward_estimation(env, model, who, episodes_to_estimate, s0_list, tmax, s_star=None):
    episode_reward_estimation = []
    
    for ep in range(episodes_to_estimate):
        episode_reward_estimation.append(get_episode_reward(env, model, who, s0_list[ep], tmax, s_star))
    
    return np.array(episode_reward_estimation)    

def draw_reward_curves(writer, iteration, env, models, models_anotations, tmax, previous_rewards, possible_s_stars,     s0_list, estimation_episodes_num):
    
    colors = sns.color_palette("Set1", n_colors=len(models_anotations), desat=.75)
    
    #current_models_rewards = previous_rewards
    for i, model in enumerate(models):
        if("Expert" in models_anotations[i]):
            reward  = get_policy_reward_estimation(env, model, who="Expert", episodes_to_estimate=estimation_episodes_num, s0_list=s0_list, tmax=tmax)
        else:
            #print(model.name)
            if "goal" in models_anotations[i]:
                reward = get_policy_reward_estimation(env, model, who="Agent", episodes_to_estimate=estimation_episodes_num, 
                                                                          s0_list=s0_list, tmax=tmax, s_star=possible_s_stars)
            else:
                reward  = get_policy_reward_estimation(env, model, who="Agent", episodes_to_estimate=estimation_episodes_num, 
                                                                          s0_list=s0_list, tmax=tmax, s_star=[None])
        writer.add_scalars('learning_stats/reward_curves',
                   {
                       models_anotations[i]: np.mean(reward)
                   }, iteration)

def get_agents_reward(env, models, models_anotations, tmax, possible_s_stars,s0_list, estimation_episodes_num):
    rewards = defaultdict(int)
    
    for i, model in enumerate(models):
        if("Expert" in models_anotations[i]):
            reward  = get_policy_reward_estimation(env, model, who="Expert", episodes_to_estimate=estimation_episodes_num, s0_list=s0_list, tmax=tmax)
        else:
            #print(model.name)
            if "goal" in models_anotations[i]:
                reward = get_policy_reward_estimation(env, model, who="Agent", episodes_to_estimate=estimation_episodes_num, 
                                                                          s0_list=s0_list, tmax=tmax, s_star=possible_s_stars)
            else:
                reward  = get_policy_reward_estimation(env, model, who="Agent", episodes_to_estimate=estimation_episodes_num, 
                                                                         s0_list=s0_list, tmax=tmax, s_star=[None])
        rewards[models_anotations[i]] = reward
        
    return rewards
        

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

 
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )