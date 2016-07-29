from random import randint
import numpy as np


def mc(alpha):
    Vs = np.full(7, 0.5, dtype=np.double)
    Vs_stages = np.full((101, 7), 0.5, dtype=np.double)  # Used for storing Vs for every stage
    num_visited = np.zeros(7, dtype=np.int)

    # Initial values for all states are 0.5 except the terminal states with value 0
    Vs[0] = 0
    Vs[6] = 0
    Vs_stages[:, 0] = 0
    Vs_stages[:, 6] = 0

    num_episode = 0

    while num_episode < 100:

        episode, reward = generate_episode_mc()

        for state in episode:
            num_visited[state] += 1
            Vs[state] += alpha*(reward - Vs[state])

        num_episode += 1
        Vs_stages[num_episode, :] = Vs
    return Vs_stages


def generate_episode_mc():
    curr_state = 3
    episode = []
    reward = 0
    while 0 < curr_state < 6:
        episode.append(curr_state)
        if randint(0, 1) == 1:
            next_state = curr_state + 1
        else:
            next_state = curr_state - 1
        if next_state == 6:
            reward = 1
        curr_state = next_state
    return episode, reward
