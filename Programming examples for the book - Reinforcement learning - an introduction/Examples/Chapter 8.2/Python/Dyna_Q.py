import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action
from helper_functions import take_action
from helper_functions import run_planning_step


def generate_episode_dyna_q(num_planning_steps,Q,Model, walls):

    epsilon = 0.1
    alpha = 0.1
    gamma = 0.95
    starting_row = 2
    starting_col = 0
    goal_row = 0
    goal_col = 8
    timesteps = 0

    current_row = starting_row
    current_col = starting_col
    episode_ended = False

    while not episode_ended:
        timesteps += 1
        action = select_action(current_row, current_col, Q, epsilon)
        next_row, next_col, reward, episode_ended = take_action(current_row, current_col, action, walls, goal_row, goal_col)
        Q[current_row, current_col, action] += alpha*(reward+gamma*max(Q[next_row,next_col,:])- Q[current_row, current_col, action])
        Model[current_row,current_col,action]['next_state_row'] = next_row
        Model[current_row, current_col,action]['next_state_col'] = next_col
        Model[current_row, current_col,action]['reward'] = reward
        for step in range(0, num_planning_steps):
            Q = run_planning_step(Q, Model, alpha, gamma)
        current_row = next_row
        current_col = next_col

    return timesteps, Q, Model

