from helper_functions import plot_values
from helper_functions import plot_policy
from helper_functions import find_value_of_best_action
from pandas import *

Vs = np.zeros((100), dtype=np.double)
policy = np.zeros((100), dtype=np.int)
gamma = 0.9
theta = 1e-7
p_h = 0.4

while True:
    delta = 0
    for state in range(1,100):
        v = Vs[state]
        Vs[state], action = find_value_of_best_action(state, Vs, p_h)
        delta = max(delta, abs(v-Vs[state]))
    #plot_values(Vs)
    if delta < theta:
        break

plot_values(Vs)

#Calculate the final policy
for state in range(1, 100):
    v, policy[state] = find_value_of_best_action(state, Vs, p_h)

plot_policy(policy)
