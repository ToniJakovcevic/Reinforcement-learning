from policyEvaluation import evaulatePolicy
from pandas import *
from policyEvaluation import plotValues, plotPolicy,plotHeatmap, evaulatePolicy,CalculateNewVs


def policyImprovement(policy,Vs, pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc):

    policy_stable=False
    policyIteration=1
    while(not policy_stable):
        policy_stable = True
        for i in range(0, 21):
            for j in range(0, 21):
                action=policy[i][j]
                policy[i][j]=findMaxActionForState(i,j,Vs,policy,pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc)
                if action != policy[i][j]:
                    policy_stable=False
        plotHeatmap(policy,policyIteration)
        if policy_stable:
            return
        else:
            Vs=evaulatePolicy(policy,Vs,policyIteration, pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc)
            policyIteration+=1
    return Vs,policy



def findMaxActionForState(i,j,Vs,policy,pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc):
    '''Iterate over possible actions from this state,
    and find the one with the greatest reward'''

    maxAction=0 #initializing temp values
    maxReward=-1000
    currAction=policy[i][j]

    if j>5:
        rangeJ=5
    else:
        rangeJ=j
    if i > 5:
        rangeI = 5
    else:
        rangeI = i

    for action in range(-rangeJ,rangeI+1):
        if j+action>20 or i-action>20:
            continue
        reward=CalculateNewVs(i,j,action,Vs,0.9,pFirstLoc,pSecondLoc,RFirstLoc,RSecondLoc)
        if reward>maxReward:
            maxReward=reward
            maxAction=action
    return maxAction