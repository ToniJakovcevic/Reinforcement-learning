import matplotlib.pyplot as plt
import numpy as np
import math


def select_action(Q, current_row, current_col, epsilon):

    if np.random.uniform() < epsilon:
        # select random action
        action = np.floor(np.random.uniform(0, 4))
    else:
        # select greedy action
        action = np.argmax(Q[current_row, current_col, :])
    return action


def take_action(current_row, current_col, action, goal_row, goal_col):

    episode_ended = False
    next_row = current_row
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
    if next_col > 11:
        next_col = 11
    if next_col < 0:
        next_col = 0
    if next_row > 3:
        next_row = 3
    if next_row < 0:
        next_row = 0

    if next_row == 3 and 0 < next_col < 11:
        reward = -100
        episode_ended = True
    elif next_row == goal_row and next_col == goal_col:
        reward = 0
        episode_ended = True
        #print("Goal reached")
    else:
        reward = -1

    return next_row, next_col, reward, episode_ended


def print_action_map(Q):
    for i in range(0, 4):
        for j in range(0, 12):
            action = np.argmax(Q[i, j, :])
            if action == 1:
                character = "→"
            elif action == 2:
                character = "↑"
            elif action == 3:
                character = "←"
            else:
                character = "↓"
            print(character, end="")
        print("")
