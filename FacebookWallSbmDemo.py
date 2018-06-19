# -*- coding: utf-8 -*-
"""
Script to fit stochastic block models to Facebook wall posts data.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sbm import *

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-wall-filtered-adj.txt')
plt.ion()
plt.figure()
plt.spy(adj)
plt.show()

#%% Estimate cluster memberships using spectral clustering
clusterId = spectralCluster(adj,directed=True)
nClusters = np.max(clusterId)+1
clusterSizes = np.histogram(clusterId, bins=nClusters)[0]
print(clusterSizes)
print(clusterId)

# Re-order nodes by class memberships and re-examine adjacency matrix
sortId = np.argsort(clusterId)
print(clusterId[sortId])
plt.figure()
plt.spy(adj[sortId[:,np.newaxis],sortId])
plt.show()

#%% Estimate edge probabilities at the block level
blockProb,logLik = estimateBlockProb(adj,clusterId,directed=True)
print(blockProb)
print(logLik)

# View estimated edge probabilities as a heat map
plt.figure()
plt.imshow(blockProb)
plt.colorbar()
plt.show()

#%% Compute reciprocity and transitivity of actual network using NetworkX
net = nx.DiGraph(adj)
recip = nx.overall_reciprocity(net)
print(recip)
trans = nx.transitivity(net)
print(trans)

#%% Simulate new networks from SBM fit to check model goodness of fit
nRuns = 100
blockProbSim = np.zeros((nClusters,nClusters,nRuns))
recipSim = np.zeros(nRuns)
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = generateSbm(clusterId,blockProb,directed=True)
    netSim = nx.DiGraph(adjSim)
    blockProbSim[:,:,run] = estimateBlockProb(adjSim,clusterId,
                                              directed=True)[0]
    recipSim[run] = nx.overall_reciprocity(netSim)
    transSim[run] = nx.transitivity(netSim)
meanBlockProbSim = np.mean(blockProbSim,axis=2)
stdBlockProbSim = np.std(blockProbSim,axis=2)
print(blockProb)
print(meanBlockProbSim)
print(meanBlockProbSim-2*stdBlockProbSim)
print(meanBlockProbSim+2*stdBlockProbSim)
meanRecipSim = np.mean(recipSim)
stdRecipSim = np.std(recipSim)
print(recip)
print(meanRecipSim)
print(meanRecipSim-2*stdRecipSim)
print(meanRecipSim+2*stdRecipSim)
meanTransSim = np.mean(transSim)
stdTransSim = np.std(transSim)
print(trans)
print(meanTransSim)
print(meanTransSim-2*stdTransSim)
print(meanTransSim+2*stdTransSim)
