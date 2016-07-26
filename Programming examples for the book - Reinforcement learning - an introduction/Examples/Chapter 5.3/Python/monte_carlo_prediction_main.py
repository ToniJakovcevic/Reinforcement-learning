from helper_functions import plot_values
from helper_functions import generate_player_turn
from helper_functions import generate_dealer_turn
from helper_functions import plot_heatmap
from helper_functions import plot_values_for_action
from random import randint
import numpy as np

Q_usable_ace = np.zeros((11, 22, 2), dtype=np.double)
Q_no_usable_ace = np.zeros((11, 22, 2), dtype=np.double)
policy_usable_ace = np.ones((11, 22), dtype=np.bool)
policy_no_usable_ace = np.ones((11, 22), dtype=np.bool)  # 0 == stick, 1 == hit

#  The initial policy is to stick only on 20 or 21
policy_usable_ace[:, 20:22] = 0
policy_no_usable_ace[:, 20:22] = 0

num_visited_usable_ace = np.zeros((11, 22, 2), dtype=np.int)
num_visited_no_usable_ace = np.zeros((11, 22, 2), dtype=np.int)

num_episodes = 0

while num_episodes<5000001:
    # Generate a random starting state
    player_sum = randint(12, 21)
    if randint(1, 13) == 1:
        has_usable_ace = True
    else:
        has_usable_ace = False
    if randint(1, 2) == 1:
        starting_action = 0
    else:
        starting_action = 1
    #  We can actually generate dealers episode first, because its policy doesn't depend on players cards
    dealer_first_card, dealer_sum, dealer_episode = generate_dealer_turn()
    #  Generate players episode
    episode, episode_usable, episode_action = generate_player_turn(player_sum, has_usable_ace, starting_action, policy_usable_ace, policy_no_usable_ace, dealer_first_card)
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
            num_visited_no_usable_ace[dealer_first_card][episode[i]][episode_action[i]] += 1
            Q_no_usable_ace[dealer_first_card][episode[i]][episode_action[i]] += (reward-Q_no_usable_ace[dealer_first_card][episode[i]][episode_action[i]])/num_visited_no_usable_ace[dealer_first_card][episode[i]][episode_action[i]]
            #  Update policy for this state
            if Q_no_usable_ace[dealer_first_card][episode[i]][0] > Q_no_usable_ace[dealer_first_card][episode[i]][1]:
                policy_no_usable_ace[dealer_first_card][episode[i]] = 0
            else:
                policy_no_usable_ace[dealer_first_card][episode[i]] = 1
        else:
            num_visited_usable_ace[dealer_first_card][episode[i]][episode_action[i]] += 1
            Q_usable_ace[dealer_first_card][episode[i]][episode_action[i]] += (reward - Q_usable_ace[dealer_first_card][episode[i]][episode_action[i]]) / num_visited_usable_ace[dealer_first_card][episode[i]][episode_action[i]]
            # Update policy for this state
            if Q_usable_ace[dealer_first_card][episode[i]][0] > Q_usable_ace[dealer_first_card][episode[i]][1]:
                policy_usable_ace[dealer_first_card][episode[i]] = 0
            else:
                policy_usable_ace[dealer_first_card][episode[i]] = 1

    num_episodes += 1
    if num_episodes == 10000000:
        plot_values(Q_no_usable_ace)
        plot_values(Q_usable_ace)
    if num_episodes % 100000 == 0:
        print(num_episodes)

plot_heatmap(policy_no_usable_ace)
plot_heatmap(policy_usable_ace)

plot_values_for_action(Q_no_usable_ace,0)
plot_values_for_action(Q_no_usable_ace,1)


plot_values(Q_no_usable_ace)
plot_values(Q_usable_ace)
