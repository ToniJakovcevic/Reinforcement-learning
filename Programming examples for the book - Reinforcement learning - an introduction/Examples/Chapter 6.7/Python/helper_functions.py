import matplotlib.pyplot as plt
import numpy as np
import math


def select_action(Q, current_state, epsilon, num_actions_from_b):

    if current_state == 2:
        if np.random.uniform() < epsilon:
            # select random action
            action = np.floor(np.random.uniform(0, 2))
        else:
            # select greedy action with randomly broken ties
            if Q[current_state, 0] == Q[current_state, 1]:
                action = np.floor(np.random.uniform(0, 2))
            else:
                action = np.argmax(Q[current_state, :])

    elif current_state == 1:
        if np.random.uniform() < epsilon:
            # select random action
            action = np.floor(np.random.uniform(0, num_actions_from_b))
        else:
            # select greedy action with randomly broken ties
            max_el_value = max(Q[current_state, :])
            indices = np.where(Q[current_state, :] == max_el_value)
            action = np.floor(np.random.uniform(0, len(indices[0])))
    return action


def take_action(current_state, action):

    episode_ended = False
    reward = 0
    next_state = 0

    if current_state == 2:
        if action == 0:
            next_state = 1
        else:
            next_state = 3
            episode_ended = True
        reward = 0
    elif current_state == 1:
        next_state = 0
        reward = np.random.normal(-0.1, 1)
        episode_ended = True

    return reward, next_state, episode_ended

def select_action_double(Q1, Q2, current_state, epsilon, num_actions_from_b):

    if current_state == 2:
        if np.random.uniform() < epsilon:
            # select random action
            action = np.floor(np.random.uniform(0, 2))
        else:
            # select greedy action with randomly broken ties
            Q_new = np.add(Q1[current_state,:],Q2[current_state,:])
            if Q_new[0] == Q_new[1]:
                action = np.floor(np.random.uniform(0, 2))
            else:
                action = np.argmax(Q_new)

    elif current_state == 1:
        if np.random.uniform() < epsilon:
            # select random action
            action = np.floor(np.random.uniform(0, num_actions_from_b))
        else:
            # select greedy action with randomly broken ties
            Q_new = np.add(Q1[current_state,:],Q2[current_state,:])
            max_el_value = max(Q_new)
            indices = np.where(Q_new == max_el_value)
            action = np.floor(np.random.uniform(0, len(indices[0])))
    return action