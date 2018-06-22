# -*- coding: utf-8 -*-
"""
Script to fit stochastic block models to Facebook wall posts data. This is a
demo presented during the ICWSM 2018 tutorial on generative models for social
media data and is intended to be run 1 line at a time in IPython.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sbm

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-wall-filtered-adj.txt')
plt.ioff()
plt.figure()
plt.spy(adj)
plt.show()

#%% Estimate cluster memberships using spectral clustering
"""
Due to a bug in Python 3.6, the interactive figure window displaying the
eigenvalues may be frozen. As a workaround, set the matplotlib backend to 
inline, e.g. by running
    %matplotlib inline
in the IPython console so that the figure shows in the console itself.
Reference: https://github.com/matplotlib/matplotlib/issues/9206/
"""
clusterId = sbm.spectralCluster(adj,directed=True)
nClusters = np.max(clusterId)+1
clusterSizes = np.histogram(clusterId, bins=nClusters)[0]
print(clusterSizes)
print(clusterId)

# Re-order nodes by class memberships and re-examine adjacency matrix
sortId = np.argsort(clusterId)
print(clusterId[sortId])
plt.ioff()
plt.figure()
plt.spy(adj[sortId[:,np.newaxis],sortId])
plt.show()

#%% Estimate edge probabilities at the block level
blockProb,logLik = sbm.estimateBlockProb(adj,clusterId,directed=True)
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
nRuns = 50
blockProbSim = np.zeros((nClusters,nClusters,nRuns))
recipSim = np.zeros(nRuns)
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = sbm.generateAdj(clusterId,blockProb,directed=True)
    netSim = nx.DiGraph(adjSim)
    blockProbSim[:,:,run] = sbm.estimateBlockProb(adjSim,clusterId,
                                                  directed=True)[0]
    recipSim[run] = nx.overall_reciprocity(netSim)
    transSim[run] = nx.transitivity(netSim)
meanBlockProbSim = np.mean(blockProbSim,axis=2)
stdBlockProbSim = np.std(blockProbSim,axis=2)
print('Actual block densities:')
print(blockProb)
print('Mean simulated block densities:')
print(meanBlockProbSim)
print('95% confidence interval lower bound:')
print(meanBlockProbSim-2*stdBlockProbSim)
print('95% confidence interval upper bound:')
print(meanBlockProbSim+2*stdBlockProbSim)
plt.figure()
plt.hist(recipSim)
plt.title('Actual reciprocity: %f' % recip)
plt.show()
plt.figure()
plt.hist(transSim)
plt.title('Actual transitivity: %f' % trans)
plt.show()
