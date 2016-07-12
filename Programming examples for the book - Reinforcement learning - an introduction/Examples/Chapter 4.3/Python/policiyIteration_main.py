import numpy as np
import matplotlib.pyplot as plt
from policyImprovement import policyImprovement
from policyEvaluation import plotValues, plotPolicy,plotHeatmap, evaulatePolicy, CalculateProbabilitiesAndRewards
from mpl_toolkits.mplot3d import Axes3D
from pandas import *

Vs = np.zeros((21, 21), dtype=np.double)
policy = np.zeros((21, 21), dtype=np.int)

pFirstLoc, RFirstLoc = CalculateProbabilitiesAndRewards(3, 3)
pSecondLoc, RSecondLoc = CalculateProbabilitiesAndRewards(4, 2)

Vs=evaulatePolicy(policy, Vs, 0, pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc)
Vs,policy=policyImprovement(policy,Vs, pFirstLoc, RFirstLoc, pSecondLoc, RSecondLoc)


