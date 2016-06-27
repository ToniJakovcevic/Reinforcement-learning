import numpy as np


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.

def get_action_from_probabilities(Pr):
    """Returns the the action based on action probabilities"""
    value = np.random.uniform()
    sum=0
    for i in range(0,10):
        sum=sum+Pr[i]
        if value<sum:
            return i
    return i



