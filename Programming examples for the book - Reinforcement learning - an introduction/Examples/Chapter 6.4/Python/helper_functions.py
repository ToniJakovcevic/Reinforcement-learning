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

def print_action_map(Q):
    for i in range(0, 7):
        for j in range(0, 10):
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
