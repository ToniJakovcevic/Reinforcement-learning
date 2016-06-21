import numpy as np


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.

def find_best_action(Qt):
    """Returns the index of the best action,
    if there are several actions with the same Qt it breaks the ties randomly"""

    max = Qt[0]
    indices = [0]

    for i in range (2,10):
        if Qt[i] > max:
            max = Qt[i]
            indices = [i]
        elif Qt[i] == max:
            indices.append(i)
    if len(indices) == 1:
        action = indices[0]
    else:
        action = indices[int(np.floor(np.random.uniform() * len(indices)))]
    return action
