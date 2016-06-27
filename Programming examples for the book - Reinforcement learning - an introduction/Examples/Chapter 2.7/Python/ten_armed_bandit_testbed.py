from GetActionFromProbabilitiies import get_action_from_probabilities
import numpy as np
import math



def ten_armed_bandit_testbed(alpha, baseline):
    # 10-armed bandit model
    numSteps = 1000
    # Generate the position of the center of distribution for every arm
    arms = np.random.normal(4.0, 1.0, (10, 2000))
    numTimesUsed = np.zeros((10, 2000), dtype=np.int)
    Qt = np.zeros((10, 2000), dtype=np.double)
    rewards = np.zeros((1000, 1), dtype=np.double)
    optimalActPercentage = np.zeros((1000, 1), dtype=np.double)
    avgReward = np.zeros((2000, 1), dtype=np.double) # used for calculation of baseline
    Pr  = np.zeros((10, 1), dtype=np.double)  # Pr(a) - Probability of taking action a at time t
    H = np.zeros((10, 2000), dtype=np.double)  # Ht(a) - action preference

    # Find  best action for each bandit
    optimalActions = np.zeros((2000, 1), dtype=np.double)
    for step in range(0, 2000):
        I = np.argmax(arms[:, step])
        optimalActions[step] = I

    for step in range(0, 1000):
        sumRewards = 0
        optActCount = 0
        for bandit in range(0, 2000):
            # first calculate the denominator in the softmax function
            denominator_sum = 0
            for it in range(0, 10):
                denominator_sum += math.exp(H[it, bandit])
            # now calculate the Pr for every action
            for act in range(0, 10):
                Pr[act] = math.exp(H[act, bandit])/denominator_sum
            # find the action with the greatest probability
            action = get_action_from_probabilities(Pr)
            reward = np.random.normal(arms[action, bandit], 1)
            if baseline:
                avgReward[bandit] += (reward - avgReward[bandit]) / (step + 1)
            # update the preferences
            for pref in range(0, 10):
                if pref == action:
                    H[pref, bandit] += alpha*(reward-avgReward[bandit])*(1-Pr[pref])
                else:
                    H[pref, bandit] -= alpha*(reward-avgReward[bandit])*(Pr[pref])

            sumRewards = sumRewards + reward
            if action == optimalActions[bandit]:
                optActCount += 1

        rewards[step] = sumRewards / 2000
        optimalActPercentage[step] = optActCount / 2000
        # Display the number of current step to see the progress
        if step % 50 == 0:
            print(step)
    return (rewards, optimalActPercentage)
