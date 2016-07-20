import matplotlib.pyplot as plt
import numpy as np
from random import randint
from mpl_toolkits.mplot3d import Axes3D

def generate_player_turn():
    player_sum = 0
    usable_aces = 0
    episode = []
    episode_usable = []
    while player_sum < 20:
        card = randint(1, 13)
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
    return episode, episode_usable

def generate_dealer_turn():
    dealer_first_card = 0
    dealer_sum = dealer_first_card
    usable_aces = 0
    num_cards = 0
    episode_usable = []
    while dealer_sum < 16:
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
    return dealer_first_card, dealer_sum

def plot_values(Vs):
    Vs=Vs[1:11, 12:22]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(12, 22, 1).tolist()
    Y = np.arange(1, 11, 1).tolist()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Vs, rstride=1, cstride=1, alpha=0.3)
    #fig.suptitle('State values - iteration ' + str(policyIteration), fontsize=20)
    plt.show()

def plot_policy(policy):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(policy, '-b', label='Vs')
    plt.ylabel('Final policy (stake)', fontsize=16)
    plt.xlabel('Capital', fontsize=16)
    plt.show()

