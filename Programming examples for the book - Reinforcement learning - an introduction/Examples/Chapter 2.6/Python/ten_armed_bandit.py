from ten_armed_bandit_testbed import ten_armed_bandit_testbed
import matplotlib.pyplot as plt

# UCB
rewards1,optimalActPercentage1 = ten_armed_bandit_testbed("UCB")

# epsilon greedy, epsilon=0.1;
rewards2,optimalActPercentage2 = ten_armed_bandit_testbed("e-greedy", 0.1)

# plot rewards
plt.figure(1)
plt.plot(rewards1, '-b', label='UCB')
plt.plot(rewards2, '-r', label='epsilon greedy, epsilon=0.1')
plt.legend(loc='lower right')
fig = plt.figure(1)
#fig.suptitle('Rewards', fontsize=16)
plt.ylabel('Rewards', fontsize=16)
plt.xlabel('Steps', fontsize=16)
plt.show()

# plot optimal action percentage
plt.figure(2)
plt.plot(optimalActPercentage1, '-b', label='UCB')
plt.plot(optimalActPercentage2, '-r', label='epsilon greedy, epsilon=0.1')
plt.legend(loc='lower right')
fig = plt.figure(2)
#fig.suptitle('Rewards', fontsize=20)
plt.ylabel('% optimal action', fontsize=16)
plt.xlabel('Steps', fontsize=16)
plt.show()


