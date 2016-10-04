import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action
from helper_functions import print_action_map

Q = np.zeros((7, 10, 4), dtype=np.double)
# Actions (moves) are as follows 1=right, 2=up, 3=left, 0=down

timesteps_vs_episodes = np.zeros(8001, dtype=np.int)
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] # Values of wind speed for each column

epsilon = 0.1
alpha = 0.5
gamma = 0.9
episode_num = 0
starting_row = 3
starting_col = 0
goal_row = 3
goal_col = 7
timesteps = 0

while timesteps < 8000:
    episode_num += 1

    # Set the starting state
    current_row = starting_row
    current_col = starting_col

    action = select_action(Q, current_row, current_col, epsilon)

    while (current_row != goal_row or current_col != goal_col) and timesteps < 8000:

        timesteps += 1
        timesteps_vs_episodes[timesteps] = episode_num
        # Determine the next state based on wind and action
        next_row = current_row - wind[current_col]
        next_col = current_col
        if action == 1:
            next_col += 1
        elif action == 2:
            next_row -= 1
        elif action == 3:
            next_col -= 1
        else:
            next_row += 1

        # Check that we stay within grid world
        if next_col > 9:
            next_col = 9
        if next_col < 0:
            next_col = 0
        if next_row > 6:
            next_row = 6
        if next_row < 0:
            next_row = 0

        if next_row == goal_row and next_col == goal_col:
            reward = 0
        else:
            reward = -1

        action_next = select_action(Q, next_row, next_col, epsilon)

        Q[current_row,current_col,action] += alpha*(reward + gamma*Q[next_row,next_col,action_next] - 1*Q[current_row,current_col,action])
        current_row = next_row
        current_col = next_col
        action = action_next


print_action_map(Q)
plt.plot(timesteps_vs_episodes, '-b')
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()
