import numpy as np
import math
from findBestAction import find_best_action


def ten_armed_bandit_testbed(algorithm, epsilon=0):
    # 10-armed bandit model
    numSteps = 1000
    # Generate the position of the center of distribution for every arm
    arms = np.random.normal(0.0, 1.0, (10, 2000))
    numTimesUsed = np.zeros((10, 2000), dtype=np.int)
    Qt = np.zeros((10, 2000), dtype=np.double)
    rewards = np.zeros((1000, 1), dtype=np.double)
    optimalActPercentage = np.zeros((1000, 1), dtype=np.double)
    c = 2  # UCB parameter
    a_values = np.zeros((10, 1), dtype=np.double)  # action values according to UCB

    # Find  best action for each bandit
    optimalActions = np.zeros((2000, 1), dtype=np.double)
    for step in range(0, 2000):
        I = np.argmax(arms[:, step])
        optimalActions[step] = I

    for step in range(0, 1000):
        sumRewards = 0
        optActCount = 0
        for bandit in range(0, 2000):
            # action selection based on algorithm
            action=-1
            if algorithm=="UCB":
                for a in range(0, 10):
                    if numTimesUsed[a, bandit] == 0:
                        action = a
                        break
                    a_values[a] = Qt[a, bandit] + c * math.sqrt(math.log(step) / numTimesUsed[a, bandit])
                I = np.argmax(a_values)
                if (action == -1):
                    action = I
            elif algorithm=="e-greedy":
                # selecting random or greedy action
                if np.random.uniform() < epsilon:
                    # select random action
                    action = np.floor(np.random.uniform(0, 10))
                else:
                    # select greedy action
                    action = find_best_action(Qt[:, bandit])

            reward = np.random.normal(arms[action, bandit], 1)
            numTimesUsed[action, bandit]=numTimesUsed[action, bandit]+1
            Qt[action, bandit] = Qt[action, bandit] + (1 / numTimesUsed[action, bandit]) * (reward - Qt[action, bandit])
            sumRewards = sumRewards + reward
            if (action == optimalActions[bandit]):
                optActCount = optActCount + 1

        rewards[step] = sumRewards / 2000
        optimalActPercentage[step] = optActCount / 2000
        # Display the number of current step to see the progress
        if step % 50 == 0:
            print(step)
    return (rewards, optimalActPercentage)
