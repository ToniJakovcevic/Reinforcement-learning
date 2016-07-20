from helper_functions import plot_values
from helper_functions import generate_player_turn
from helper_functions import generate_dealer_turn
import numpy as np

Vs_usable_ace = np.zeros((11,22), dtype=np.double)
Vs_no_usable_ace = np.zeros((11,22), dtype=np.double)
num_visited_usable_ace = np.zeros((11,22), dtype=np.int)
num_visited_no_usable_ace = np.zeros((11,22), dtype=np.int)

num_episodes = 0

while num_episodes<500001:
    episode, episode_usable = generate_player_turn()
    dealer_first_card, dealer_sum = generate_dealer_turn()
    if episode[-1] > 21:
        reward = -1
    elif dealer_sum > 21:
        reward = 1
    else:
        if episode[-1] > dealer_sum:
            reward = 1
        elif episode[-1] == dealer_sum:
            reward = 0
        else:
            reward = -1
    if episode[-1] > 21:
        episode.pop()
    for i in range(0, len(episode)):
        if episode_usable[i] == 0:
            num_visited_no_usable_ace[dealer_first_card][episode[i]] += 1
            Vs_no_usable_ace[dealer_first_card][episode[i]] += (reward-Vs_no_usable_ace[dealer_first_card][episode[i]])/num_visited_no_usable_ace[dealer_first_card][episode[i]]
        else:
            num_visited_usable_ace[dealer_first_card][episode[i]] += 1
            Vs_usable_ace[dealer_first_card][episode[i]] += (reward - Vs_usable_ace[dealer_first_card][episode[i]]) / num_visited_usable_ace[dealer_first_card][episode[i]]
    num_episodes += 1
    if num_episodes == 10000:
        plot_values(Vs_no_usable_ace)
        plot_values(Vs_usable_ace)
    if num_episodes % 100000 == 0:
        print(num_episodes)

plot_values(Vs_no_usable_ace)
plot_values(Vs_usable_ace)
