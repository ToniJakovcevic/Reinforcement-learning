import matplotlib.pyplot as plt
import numpy as np
from helper_functions import select_action
from helper_functions import take_action

def Q_learning(episodes_on_left, num_actions_from_b, num_episodes):


    Q = np.zeros((4, num_actions_from_b), dtype=np.double)
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

            action = select_action(Q, current_state, epsilon, num_actions_from_b)
            reward, next_state, episode_ended = take_action(current_state, action)
            if current_state == 2 and action == 0:
                episodes_on_left[episode_num] += 1
            Q[current_state,action] += alpha*(reward + gamma*max(Q[next_state,:]) - Q[current_state,action])
            current_state = next_state

    return episodes_on_left
