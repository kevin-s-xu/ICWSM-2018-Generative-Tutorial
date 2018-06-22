# -*- coding: utf-8 -*-
"""
Script to fit stochastic block models and latent space models to Facebook
friendship data. This is a demo presented during the ICWSM 2018 tutorial on
generative models for social media data and is intended to be run 1 line at
a time in IPython.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sbm
import latentSpaceModel as lsm

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-links-filtered-adj.txt')
net = nx.Graph(adj)
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
clusterId = sbm.spectralCluster(adj,directed=False)
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
blockProb,logLik = sbm.estimateBlockProb(adj,clusterId,directed=False)
print(blockProb)
print(logLik)

# View estimated edge probabilities as a heat map
plt.figure()
plt.imshow(blockProb)
plt.colorbar()
plt.show()

#%% Compute transitivity of actual network using NetworkX
trans = nx.transitivity(net)
print(trans)

#%% Simulate new networks from SBM fit to check model goodness of fit
nRuns = 50
blockProbSim = np.zeros((nClusters,nClusters,nRuns))
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = sbm.generateAdj(clusterId,blockProb,directed=False)
    netSim = nx.Graph(adjSim)
    blockProbSim[:,:,run] = sbm.estimateBlockProb(adjSim,clusterId,
                                                  directed=False)[0]
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
plt.hist(transSim)
plt.title('Actual transitivity: %f' % trans)
plt.show()

#%% Fit latent space model using 2 latent dimensions
posEst,biasEst,logLik,optRes = lsm.estimateParams(adj,dim=2)
print(biasEst)
print(logLik)

plt.figure()
plt.axis('off')
nx.draw_networkx(net,pos=dict(enumerate(posEst)),node_size=200,
                 width=0.5,alpha=0.5)
plt.show()
plt.figure()
plt.axis('off')
nx.draw_networkx(net,pos=nx.kamada_kawai_layout(net),node_size=200,
                 width=0.5,alpha=0.5)
plt.show()

#%% Compute density of actual network using NetworkX
density = nx.density(net)
print(density)

#%% Simulate new networks from LSM fit to check model goodness of fit
nRuns = 50
densitySim = np.zeros(nRuns)
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = lsm.generateAdj(posEst,biasEst)
    netSim = nx.Graph(adjSim)
    densitySim[run] = nx.density(netSim)
    transSim[run] = nx.transitivity(netSim)
plt.figure()
plt.hist(densitySim)
plt.title('Actual density: %f' % density)
plt.show()
plt.figure()
plt.hist(transSim)
plt.title('Actual transitivity: %f' % trans)
plt.show()

