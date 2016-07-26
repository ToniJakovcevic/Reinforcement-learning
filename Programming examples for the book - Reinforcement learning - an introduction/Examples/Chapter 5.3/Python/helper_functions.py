import matplotlib.pyplot as plt
import numpy as np
from random import randint
from mpl_toolkits.mplot3d import Axes3D


def generate_player_turn(player_sum, has_usable_ace, starting_action, policy_usable_ace, policy_no_usable_ace, dealer_first_card):

    if has_usable_ace:
        usable_aces = 1
    else:
        usable_aces = 0
    episode = []
    episode_usable = []
    episode_action = []
    episode.append(player_sum)
    episode_usable.append(usable_aces)
    episode_action.append(starting_action)

    #  execute starting action
    if starting_action == 0:
        return episode, episode_usable, episode_action
    else:
        episode, episode_usable, usable_aces = flip_card(usable_aces, episode, episode_usable)

    #  follow policy afterwards
    while episode[-1] < 22:
        if usable_aces > 0:
            action = policy_usable_ace[dealer_first_card][episode[-1]]
        else:
            action = policy_no_usable_ace[dealer_first_card][episode[-1]]
        episode_action.append(action)
        if action == 0:
            return episode, episode_usable, episode_action
        else:
            episode, episode_usable, usable_aces = flip_card(usable_aces, episode, episode_usable)

    return episode, episode_usable, episode_action


def flip_card(usable_aces, episode, episode_usable):
    card = randint(1, 13)
    player_sum = episode[-1]
    if card == 1:
        usable_aces += 1
        player_sum += 11
    elif card > 9:
        player_sum += 10
    else:
        player_sum += card
    if player_sum > 21 and usable_aces > 0:
        player_sum -= 10
        usable_aces -= 1
    episode.append(player_sum)
    episode_usable.append(usable_aces)
    return episode, episode_usable, usable_aces


def generate_dealer_turn():
    dealer_first_card = 0
    dealer_sum = dealer_first_card
    usable_aces = 0
    num_cards = 0
    dealer_episode = []
    while dealer_sum < 17:
        card = randint(1, 13)
        if card == 1:
            usable_aces += 1
            dealer_sum += 11
        elif card > 9:
            dealer_sum += 10
        else:
            dealer_sum += card
        if num_cards == 0:
            dealer_first_card = dealer_sum
            if dealer_first_card == 11:
                dealer_first_card = 1
        num_cards += 1
        if dealer_sum > 21 and usable_aces > 0:
            dealer_sum -= 10
            usable_aces -= 1
        dealer_episode.append(dealer_sum)
    return dealer_first_card, dealer_sum, dealer_episode


def plot_values(Q):
    Vs = compute_state_value(Q)
    Vs = Vs[1:11, 12:22]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(12, 22, 1).tolist()
    Y = np.arange(1, 11, 1).tolist()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Vs, rstride=1, cstride=1, alpha=0.3)
    ax.set_zlim(-1, 1)
    plt.draw()
    plt.show()

def plot_values_for_action(Q,action):
    Vs = Q[:,:,action]#compute_state_value(Q)
    Vs = Vs[1:11, 12:22]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(12, 22, 1).tolist()
    Y = np.arange(1, 11, 1).tolist()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Vs, rstride=1, cstride=1, alpha=0.3)
    ax.set_zlim(-1, 1)
    plt.draw()
    plt.show()

def plot_heatmap(policy):
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(policy, cmap=plt.cm.Reds)
    # legend
    cbar = plt.colorbar(heatmap)
    plt.ylim([1, 11])
    plt.xlim([1, 21])
    fig.suptitle('Policy', fontsize=20)

    plt.show()
    #plt.close()

def compute_state_value(Q):
    Vs = np.zeros((11, 22), dtype=np.double)
    for i in range (0, 11):
        for j in range (0, 22):
            Vs[i][j] = max(Q[i,j,:])
    return Vs