from ten_armed_bandit_testbed import ten_armed_bandit_testbed
import matplotlib.pyplot as plt


rewards1,optimalActPercentage1 = ten_armed_bandit_testbed(0.1, True)
rewards2,optimalActPercentage2 = ten_armed_bandit_testbed(0.1, False)
rewards3,optimalActPercentage3 = ten_armed_bandit_testbed(0.4, True)
rewards4,optimalActPercentage4 = ten_armed_bandit_testbed(0.4, False)

# plot rewards
plt.figure(1)
plt.plot(rewards1, '-b', label='alpha=0.1 with baseline')
plt.plot(rewards2, '-r', label='alpha=0.1 without baseline')
plt.plot(rewards3, '-g', label='alpha=0.4 with baseline')
plt.plot(rewards4, '-k', label='alpha=0.4 without baseline')
plt.legend(loc='lower right')
fig = plt.figure(1)
#fig.suptitle('Rewards', fontsize=16)
plt.ylabel('Rewards', fontsize=16)
plt.xlabel('Steps', fontsize=16)
plt.show()

# plot optimal action percentage
plt.figure(2)
plt.plot(optimalActPercentage1, '-b', label='alpha=0.1 with baseline')
plt.plot(optimalActPercentage2, '-r', label='alpha=0.1 without baseline')
plt.plot(optimalActPercentage3, '-g', label='alpha=0.4 with baseline')
plt.plot(optimalActPercentage4, '-k', label='alpha=0.4 without baseline')
plt.legend(loc='lower right')
fig = plt.figure(2)
#fig.suptitle('Rewards', fontsize=20)
plt.ylabel('% optimal action', fontsize=16)
plt.xlabel('Steps', fontsize=16)
plt.show()


