import numpy as np
from findBestAction import find_best_action

def ten_armed_bandit_testbed(algorith, epsilon):
    # 10-armed bandit model
    numSteps = 1000
    # Generate the position of the center of distribution for every arm
    arms = np.random.normal(0.0, 1.0, (10, 2000))
    numTimesUsed = np.zeros((10, 2000), dtype=np.int)
    Qt = np.full((10, 2000), bias)
    rewards = np.zeros((1000, 1))
    optimalActPercentage = np.zeros((1000, 1), dtype=np.double)
    alpha = 0.1  # This example uses constant step size parameter

    # Find  best action for each bandit
    optimalActions = np.zeros((2000, 1), dtype=np.double)
    for step in range(0, 2000):
        I = np.argmax(arms[:, step])
        optimalActions[step] = I

    for step in range(0,1000):
        sumRewards = 0
        optActCount = 0
        for bandit in range(0,2000):
            # selecting random or greedy action
            if np.random.uniform() < epsilon:
                # select random action
                action = np.floor(np.random.uniform(0, 10))
            else:
                # select greedy action
                action = find_best_action(Qt[:, bandit])

            reward = np.random.normal(arms[action, bandit], 1)
            numTimesUsed[action, bandit]=numTimesUsed[action, bandit]+1
            Qt[action, bandit] = Qt[action, bandit] + alpha*(reward-Qt[action,bandit])
            sumRewards = sumRewards + reward
            if (action == optimalActions[bandit]):
                optActCount = optActCount + 1;

        rewards[step] = sumRewards / 2000
        optimalActPercentage[step] = optActCount / 2000;
        # Display the number of current step to see the progress
        if step % 50 == 0:
            print(step)
    return (rewards, optimalActPercentage)
