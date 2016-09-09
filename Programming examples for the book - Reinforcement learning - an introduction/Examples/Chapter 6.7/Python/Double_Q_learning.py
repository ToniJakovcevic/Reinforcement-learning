import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action_double
from helper_functions import take_action

def double_Q_learning(episodes_on_left, num_actions_from_b, num_episodes):

    Q1 = np.zeros((4, num_actions_from_b), dtype=np.double)
    Q2 = np.zeros((4, num_actions_from_b), dtype=np.double)
    # 4 = num states [0 - left(terminal), 1 - B, 2 - A, 3 - right(terminal)]
    # Second parameter = actions, for state A only 2 are used

    epsilon = 0.1
    alpha = 0.1
    gamma = 1
    episode_num = 0
    starting_state = 2

    while episode_num < num_episodes:
        episode_num += 1

        # Set the starting state
        current_state = starting_state
        episode_ended = False

        while not episode_ended:

            action = select_action_double(Q1, Q2, current_state, epsilon, num_actions_from_b)
            reward, next_state, episode_ended = take_action(current_state, action)
            if current_state == 2 and action == 0:
                episodes_on_left[episode_num] += 1
            if np.random.uniform() < 0.5:
                Q1[current_state,action] += alpha*(reward + gamma*Q2[next_state,max(Q1[next_state,:])] - Q1[current_state,action])
            else:
                Q2[current_state,action] += alpha*(reward + gamma*Q1[next_state,max(Q2[next_state,:])] - Q2[current_state,action])

            current_state = next_state

    return episodes_on_left
