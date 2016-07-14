import matplotlib.pyplot as plt

def find_value_of_best_action(state, Vs, p_h):
    action_range = min(state, 100-state)
    maxValue = -1000
    maxAction = 0
    for action in range(1, action_range + 1): #does not include action that leads to terminal state (action_range)
        if state + action == 100:
            value = p_h + (1 - p_h) * Vs[state - action]
        else:
            value = p_h * Vs[state+action] + (1-p_h)*Vs[state-action]
        if value > maxValue:
            maxValue = value
            maxAction = action
    return maxValue, maxAction

def plot_values(Vs):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(Vs, '-b', label='Vs')
    plt.ylabel('Value estimates', fontsize=16)
    plt.xlabel('Capital', fontsize=16)
    plt.show()

def plot_policy(policy):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(policy, '-b', label='Vs')
    plt.ylabel('Final policy (stake)', fontsize=16)
    plt.xlabel('Capital', fontsize=16)
    plt.show()

