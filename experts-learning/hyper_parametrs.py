import os
gamma_rl = 1

lambda_baseline = 0.5 #best 0.25 (0.5)
scale_reward = 10
lambda_goals = 5 #best 1(5)

entropy_weights = {
    "agent":0.025, #best 0.025
}

eps = 1e-6

lr_agent = 5e-4 #best 2e-3
tmax = 40

logs_directory = os.path.join("./", "logs")
board_port=6006
board_timeout=24*60*60