import matplotlib.pyplot as plt
import numpy as np
import math


def select_action(current_row, current_col, Q, epsilon):

    if np.random.uniform() < epsilon:
        # select random action
        action = np.floor(np.random.uniform(0, 4))
    else:
        # select greedy action with randomly broken ties
        max_el_value = max(Q[current_row, current_col, :])
        indices = np.where(Q[current_row,current_col, :] == max_el_value)
        action = indices[0][np.floor(np.random.uniform(0, len(indices[0])))]

    return action


def take_action(current_row, current_col, action, walls, goal_row, goal_col):

    episode_ended = False
    reward = 0

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
    if next_col > 8:
        next_col = 8
    if next_col < 0:
        next_col = 0
    if next_row > 5:
        next_row = 5
    if next_row < 0:
        next_row = 0

    # Check if we have hit a wall
    if walls[next_row, next_col] == 1:
        next_row = current_row
        next_col = current_col

    if next_row == goal_row and next_col == goal_col:
        reward = 1
        episode_ended = True
    else:
        reward = 0

    return next_row, next_col, reward, episode_ended


def run_planning_step(Q, Model, alpha, gamma):
    # Get a random previously observed state
    num_observed_states, observed_states = get_observed_states(Model)
    random_state_index = np.floor(np.random.uniform(0, num_observed_states))
    current_row = observed_states[random_state_index]['row']
    current_col = observed_states[random_state_index]['col']
    # Get a random previously observed action
    action = get_observed_action(Model, current_row,current_col)

    next_row = Model[current_row,current_col,action]['next_state_row']
    next_col = Model[current_row,current_col,action]['next_state_col']
    reward = Model[current_row,current_col,action]['reward']
    Q[current_row, current_col, action] += alpha * (reward + gamma * max(Q[next_row, next_col, :]) - Q[current_row, current_col, action])
    return Q


def get_observed_states(Model):
    observed_states = np.ones(54, dtype=[('row', '>i4'), ('col', '>i4')])
    num_observed_states = 0
    for row in range(0, 6):
        for col in range (0, 9):
            for action in range(0,4):
                if Model[row,col,action]['next_state_row'] != -1:
                    observed_states[num_observed_states]['row'] = row
                    observed_states[num_observed_states]['col'] = col
                    num_observed_states += 1
                    break
    return num_observed_states, observed_states

def get_observed_action(Model, current_row, current_col):

    actions = np.zeros(4, dtype=np.int)
    num_observed_actions = 0
    for act in range(0,4):
        if Model[current_row,current_col,act]['next_state_row']!=-1:
            actions[num_observed_actions] = act
            num_observed_actions += 1
    random_action_index = np.floor(np.random.uniform(0, num_observed_actions))
    return actions[random_action_index]


def initialize_model(Model):
    # Set all the values to -1
    for x in np.nditer(Model, op_flags=['readwrite']):
        x[...]['next_state_row'] = -1 * x['next_state_row']
        x[...]['next_state_col'] = -1 * x['next_state_col']
        x[...]['reward'] = -1 * x['reward']
    return Model


