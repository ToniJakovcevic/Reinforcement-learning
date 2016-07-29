import matplotlib.pyplot as plt
import numpy as np
import math


def plot_value_stages(Vs_stages):

    true_values = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
    x_axis_ticks = [1, 2, 3, 4, 5]

    plt.plot(true_values, '-ob', label='True values')
    plt.plot(Vs_stages[0, :], '-oy', label='After 0 episodes')
    plt.plot(Vs_stages[1, :], '-og', label='After 1 episode')
    plt.plot(Vs_stages[10, :], '-or', label='After 10 episodes')
    plt.plot(Vs_stages[100, :], '-oc', label='After 100 episodes')
    plt.legend(loc='best')
    plt.xlim([1, 5])
    labels = ['A', 'B', 'C', 'D', 'E']
    plt.xticks(x_axis_ticks, labels)
    plt.xlabel('States')
    plt.ylabel('State value estimates')
    plt.show()


def calculate_rms(func, args):
    sequences = []
    rms = np.zeros(101, dtype=np.double)
    true_values = [1/6, 2/6, 3/6, 4/6, 5/6]

    #  Generate 100 sequences of 100 episodes
    for sequence in range(0, 100):
        sequences.append(func(args))

    for episode in range(0, 101):
        total_rms = 0
        for sequence in range(0, 100):
            total_rms_residuals = 0
            for state in range(1, 6):
                total_rms_residuals += (true_values[state-1]-sequences[sequence][episode][state])**2
            total_rms += math.sqrt(total_rms_residuals/5)
        rms[episode] = total_rms / 100
    return rms
