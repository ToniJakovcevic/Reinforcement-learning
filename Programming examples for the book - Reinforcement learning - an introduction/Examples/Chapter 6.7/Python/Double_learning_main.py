import matplotlib.pyplot as plt
import numpy as np
from Q_learning import Q_learning
from Double_Q_learning import double_Q_learning

num_episodes = 1000
num_actions_from_b = 6

episodes_on_left_q = np.zeros(num_episodes +1, dtype=np.double)
episodes_on_left_double_q = np.zeros(num_episodes +1, dtype=np.double)

num_runs = 1000
for i in range(0,num_runs):
    episodes_on_left_q = Q_learning(episodes_on_left_q, num_actions_from_b, num_episodes)
    episodes_on_left_double_q = double_Q_learning(episodes_on_left_double_q, num_actions_from_b, num_episodes)

episodes_on_left_q = episodes_on_left_q / num_runs * 100
episodes_on_left_double_q = episodes_on_left_double_q / num_runs * 100

plt.plot(episodes_on_left_q, '-b', label='Q learning')
plt.plot(episodes_on_left_double_q, '-r', label='Double Q learning')
plt.ylabel('% left actions from A')
plt.xlabel('Episodes')
plt.legend(loc='upper right')
plt.title('num_episodes=' + str(num_episodes) + '; num_actions_from_b=' + str(num_actions_from_b))
plt.show()