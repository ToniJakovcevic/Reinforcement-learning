import numpy as np
from pandas import *
import matplotlib.pyplot as plt
import math

gamma = 0.9

def evaulatePolicy(policy, Vs, policyIteration, pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc):
    global gamma
    num_iterations = 0
    currVs = 0
    minDelta = 1e-7
    delta = 1000  # temp value before assignment
    tempVs = np.zeros((21, 21), dtype=np.double)

    while delta > minDelta:
        delta = 0
        for i in range(0, 21):
            for j in range(0, 21):
                currVs = Vs[i][j]
                # Get the reward and new state by following the current policy
                action=policy[i,j]
                Vs[i][j]=0
                Vs[i][j]+=CalculateNewVs(i,j,action,Vs,gamma,pFirstLoc,pSecondLoc,RFirstLoc,RSecondLoc)
                delta = max(delta, abs(currVs-Vs[i][j]))
        num_iterations += 1
        print("Num iterations=",num_iterations)
        print("Delta=", delta)
    print("Num iterations=", num_iterations)
    return Vs

def CalculateNewVs(currI,currJ,action,Vs,gamma,pFirstLoc,pSecondLoc,RFirstLoc,RSecondLoc):

    #apply the action
    locI = currI - action
    locJ = currJ + action
    negReward = -abs(action)*2
    newVs=negReward
    for i in range(0, 21):
        for j in range(0, 21):
            newVs +=  pFirstLoc[locI][i]*pSecondLoc[locJ][j]*(RFirstLoc[locI]+RSecondLoc[locJ] + gamma*Vs[i][j])
    return newVs

def CalculateProbabilitiesAndRewards(lambdaRequests, lambdaReturns):

    #two dimensional probabilities 1) current location/state 2) next state
    P=np.zeros((21,21), dtype=np.double)
    R=np.zeros((21,1), dtype=np.double)
    max_avialable=0
    for currLoc in range(0, 21):
        #Calculate rewards (don't depend on returns)
        for requests in range(0, 21):
            max_avialable=min(currLoc,requests)
            R[currLoc]+=probabilityPoisson(requests,lambdaRequests)*max_avialable*10
    new_state=0
     #Calculate probabilities
    for requests in range(0,21):
        for returns in range (0,21):
            for state in range(0,21):
                max_avialable = min(requests, state)
                new_state=state+returns-max_avialable
                if new_state>20:
                    new_state=20
                P[state][new_state]+=probabilityPoisson(requests,lambdaRequests)*probabilityPoisson(returns,lambdaReturns)

    return P, R


def plotPolicy(policy, policyIteration):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = Y= np.arange(0, 21, 1).tolist()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, policy, rstride=1, cstride=1, alpha=0.3)
    fig.suptitle('Policy - iteration ' + str(policyIteration), fontsize=20)
    plt.show()

def plotHeatmap(policy,policyIteration):
    fig, ax = plt.subplots()
    heatmap = plt.pcolor(policy, cmap=plt.cm.Reds)
    # legend
    cbar = plt.colorbar(heatmap)
    plt.ylim([0, 20])
    plt.xlim([0, 20])
    fig.suptitle('Policy - iteration ' + str(policyIteration), fontsize=20)

    plt.show()
    #plt.close()

def plotValues(Vs, policyIteration):
    print(DataFrame(Vs))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = Y= np.arange(0, 21, 1).tolist()
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Vs, rstride=1, cstride=1, alpha=0.3)
    fig.suptitle('State values - iteration ' + str(policyIteration), fontsize=20)
    plt.show()

def probabilityPoisson(n,lambda_p):
    return ((lambda_p**n) / (math.factorial(n)))*math.exp(-lambda_p)