from helper_functions import calculate_rms
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from td0 import td0
from mc import mc

#  Plot the estimated values after 100 episodes
td0(0.01, True)

#  Calculate and plot rms errors
rms_td0_a_0_1 = calculate_rms(td0, (0.1))
rms_td0_a_0_05 = calculate_rms(td0, (0.05))
rms_td0_a_0_15 = calculate_rms(td0, (0.15))
rms_mc_a_0_01 = calculate_rms(mc, (0.01))
rms_mc_a_0_02 = calculate_rms(mc, (0.02))
rms_mc_a_0_03 = calculate_rms(mc, (0.03))
rms_mc_a_0_04 = calculate_rms(mc, (0.04))

plt.plot(rms_td0_a_0_1, '-r', label='TD alpha=0.1')
plt.plot(rms_td0_a_0_05, '--r', label='TD alpha=0.05')
plt.plot(rms_td0_a_0_15, '-.r', label='TD alpha=0.15')
plt.plot(rms_mc_a_0_01, '-b', label='MC alpha=0.01')
plt.plot(rms_mc_a_0_02, '--b', label='MC alpha=0.02')
plt.plot(rms_mc_a_0_03, '-.b', label='MC alpha=0.03')
plt.plot(rms_mc_a_0_04, ':b', label='MC alpha=0.04')
plt.legend(loc='best')
plt.xlabel('Episodes')
plt.ylabel('RMS error')
plt.show()
