import numpy as np
from pandas import *

gridworld = np.zeros((4, 4), dtype=np.double)
gamma = 1  # was not given

def main():
    global gridworld, gamma
    num_iterations = 0
    currVs = 0
    minDelta = 1.0e-10  # using very small value to prevent stopping before good convergence
    delta = 5  # temp value before assignment
    while delta > minDelta:
        delta = 0
        for i in range(0, 4):
            for j in range(0, 4):
                if ((i == 0) and (j == 0)) or ((i == 3) and (j == 3)):
                    continue
                currVs = gridworld[i][j]
                calculateVs(i, j)
                delta = max(delta, abs(currVs-gridworld[i][j]))
        num_iterations += 1
        print(num_iterations)
        print(delta)
    print(num_iterations)
    print(DataFrame(gridworld))


def calculateVs(i, j):
    #  Calculates the new value for state s, the v(s)
    sum=0
    currentVs=gridworld[i][j]
    sum+= -1 + gamma*(getValueOfNetxtStep(i-1,j,currentVs))
    sum+= -1 + gamma*(getValueOfNetxtStep(i+1,j,currentVs))
    sum+= -1 + gamma*(getValueOfNetxtStep(i,j-1,currentVs))
    sum+= -1 + gamma*(getValueOfNetxtStep(i,j+1,currentVs))
    sum/=4
    gridworld[i][j]=sum
    return


def getValueOfNetxtStep(nextI, nextJ, currentVs):
    if (nextI<0) or (nextI>3) or (nextJ<0) or (nextJ>3):
        return currentVs
    return gridworld[nextI,nextJ]


main()