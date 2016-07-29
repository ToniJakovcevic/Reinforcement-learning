from helper_functions import plot_value_stages
from random import randint
import numpy as np


def td0(alpha, plot=False):
    Vs = np.full(7, 0.5, dtype=np.double)
    Vs_stages = np.full((101, 7), 0.5, dtype=np.double)  # Used for storing Vs for every stage

    # Initial values for all states are 0.5 except the terminal states with value 0
    Vs[0] = 0
    Vs[6] = 0
    Vs_stages[:, 0] = 0
    Vs_stages[:, 6] = 0

    num_episode = 0

    while num_episode < 100:
        curr_state = 3
        while 0 < curr_state < 6:
            if randint(0, 1) == 1:
                next_state = curr_state + 1
            else:
                next_state = curr_state - 1
            if next_state == 6:
                reward = 1
            else:
                reward = 0
            Vs[curr_state] += alpha * (reward + Vs[next_state] - Vs[curr_state])
            curr_state = next_state

        num_episode += 1
        Vs_stages[num_episode, :] = Vs
    if plot:
        plot_value_stages(Vs_stages)
    return Vs_stages
