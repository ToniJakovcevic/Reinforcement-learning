from ten_armed_bandit_testbed import ten_armed_bandit_testbed
import matplotlib.pyplot as plt

# greedy
rewards = ten_armed_bandit_testbed(0)
plt.plot(rewards, '-b', label='greedy')
# epsilon=0.01;
rewards = ten_armed_bandit_testbed(0.01)
plt.plot(rewards, '-r', label='epsilon=0.01')
# epsilon=0.1;
rewards = ten_armed_bandit_testbed(0.1)
plt.plot(rewards, '-g', label='epsilon=0.1')
plt.legend(loc='lower right')
plt.show()


