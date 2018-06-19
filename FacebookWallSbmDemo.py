# -*- coding: utf-8 -*-
"""
Script to fit stochastic block models to Facebook wall posts data.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from fitSbm import *

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-wall-filtered-adj.txt')
plt.figure()
plt.spy(adj)
plt.show()

#%% Estimate cluster memberships using spectral clustering
clusterId = spectralCluster(adj,directed=True)
clusterSizes = np.histogram(clusterId, bins=np.max(clusterId)+1)[0]
print(clusterSizes)
print(clusterId)

#%% Estimate edge probabilities at the block level
blockProb,logLik = estimateBlockProb(adj,clusterId,directed=True)
print(blockProb)
print(logLik)

plt.figure()
plt.imshow(blockProb)
plt.colorbar()
plt.show()
