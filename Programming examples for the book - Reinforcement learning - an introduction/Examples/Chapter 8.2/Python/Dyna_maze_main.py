import matplotlib.pyplot as plt
import numpy as np
from helper_functions import initialize_model
from Dyna_Q import generate_episode_dyna_q


#  Define the location of the walls
walls = np.zeros((6, 9), dtype=np.int)
walls[1:4, 2] = 1
walls[4, 5] = 1
walls[0:3, 7] = 1

num_steps_per_episode = np.zeros((51, 3), dtype=np.double)
#  Second dimension is the method (0 = 0 planning steps, 1 = 5 planning steps, 2 = 50 planning steps)

for run in range(0,30):

    # Actions (moves) are as follows 1=right, 2=up, 3=left, 0=down
    Q0 = np.zeros((6, 9, 4), dtype=np.double)
    Q5 = np.zeros((6, 9, 4), dtype=np.double)
    Q50 = np.zeros((6, 9, 4), dtype=np.double)
    Model0 = np.ones((6, 9, 4), dtype=[('next_state_row', '>i4'), ('next_state_col', '>i4'), ('reward', '>i4')])
    Model0 = initialize_model(Model0)
    Model5 = np.ones((6, 9, 4), dtype=[('next_state_row', '>i4'), ('next_state_col', '>i4'), ('reward', '>i4')])
    Model5 = initialize_model(Model5)
    Model50 = np.ones((6, 9, 4), dtype=[('next_state_row', '>i4'), ('next_state_col', '>i4'), ('reward', '>i4')])
    Model50 = initialize_model(Model50)

    for episode in range(1,51):
        seed=round(np.random.uniform(0, 100000))
        np.random.seed(seed)
        timesteps0, Q0, Model0 = generate_episode_dyna_q(0, Q0, Model0, walls)
        np.random.seed(seed)
        timesteps5, Q5, Model5 = generate_episode_dyna_q(5, Q5, Model5, walls)
        np.random.seed(seed)
        timesteps50, Q50, Model50 = generate_episode_dyna_q(50, Q50, Model50, walls)
        num_steps_per_episode[episode, 0] += timesteps0
        num_steps_per_episode[episode, 1] += timesteps5
        num_steps_per_episode[episode, 2] += timesteps50

    print(run)

num_steps_per_episode = num_steps_per_episode / 30
plt.plot(num_steps_per_episode[:,0], '-b', label='0 planning steps')
plt.plot(num_steps_per_episode[:,1], '-g', label='5 planning steps')
plt.plot(num_steps_per_episode[:,2], '-r', label='50 planning steps')
plt.ylabel('Steps per episode')
plt.xlabel('Episodes')
plt.legend(loc='upper right')
plt.xlim([1,50])
plt.show()
