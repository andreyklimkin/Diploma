import os
#MDP
gamma_rl = 1

#loss coeffs
lambda_baseline = 0.5 #best 0.25 (0.5)
scale_reward = 10
lambda_goals = 5 * 8 #best 5(1) best for 9x9 simple  5 * 20

entropy_weights = {
    "agent":0.0025, #best 0.025
}

#optimizers
lr_agent = 1e-3#best for 9x9 simple  1e-4
weight_decay = 0
decrease_lr_every = 100000

#agent net
tmax = 40 #40
train_steps = 10000#25000

goal_sampling_strategy = "random"
goal_epsilon = 0.5
goal_epsilon_decay = 0.99


#agent net
hidden_layer1_size = 100
hidden_layer2_size = 100
hidden_layer3_size = 100

#tensorboard
logs_directory = os.path.join("./", "plots/simple_5x5_siamese")
board_port=6016
board_timeout=24*60*60

eps = 1e-6
