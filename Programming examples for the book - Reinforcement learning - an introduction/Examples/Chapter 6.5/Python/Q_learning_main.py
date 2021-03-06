import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action
from helper_functions import take_action
from helper_functions import print_action_map

Q = np.zeros((4, 12, 4), dtype=np.double)
# Actions (moves) are as follows 1=right, 2=up, 3=left, 0=down

timesteps_vs_episodes = np.zeros(8001, dtype=np.int)

epsilon = 0.1
alpha = 0.5
gamma = 1
episode_num = 0
starting_row = 3
starting_col = 0
goal_row = 3
goal_col = 11
timesteps = 0

while episode_num < 500:
    episode_num += 1

    # Set the starting state
    current_row = starting_row
    current_col = starting_col
    episode_ended = False

    while (current_row != goal_row or current_col != goal_col) and not episode_ended:

        timesteps += 1

        action = select_action(Q, current_row, current_col, epsilon)

        # Take action
        next_row, next_col, reward, episode_ended = take_action(current_row, current_col, action, goal_row, goal_col)

        Q[current_row,current_col,action] += alpha*(reward + gamma*max(Q[next_row,next_col,:]) - Q[current_row,current_col,action])
        current_row = next_row
        current_col = next_col


print_action_map(Q)

