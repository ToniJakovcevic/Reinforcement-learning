import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action
from helper_functions import print_action_map
from helper_functions import take_action

Q = np.zeros((4, 12, 4), dtype=np.double)
# Actions (moves) are as follows 1=right, 2=up, 3=left, 0=down

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

    action = select_action(Q, current_row, current_col, epsilon)
    episode_ended = False

    while (current_row != goal_row or current_col != goal_col) and not episode_ended:

        timesteps += 1

        next_row, next_col, reward, episode_ended = take_action(current_row, current_col, action, goal_row, goal_col)

        action_next = select_action(Q, next_row, next_col, epsilon)

        Q[current_row,current_col,action] += alpha*(reward + gamma*Q[next_row,next_col,action_next] - Q[current_row,current_col,action])
        current_row = next_row
        current_col = next_col
        action = action_next


print_action_map(Q)
