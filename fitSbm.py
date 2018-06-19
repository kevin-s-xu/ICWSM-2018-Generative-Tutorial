# -*- coding: utf-8 -*-
"""
Functions to fit stochastic block model to network.

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def spectralCluster(adj,nClusters=0,directed=False):
    # Compute full SVD of adjacency matrix
    u, s, v = np.linalg.svd(adj)
    v = v.T
    
    if nClusters == 0:
        # User didn't pre-select the number of clusters so plot the top 20
        # singular values and let the user enter the number of clusters
        plt.ion()
        plt.plot(s[:20],'x')
        plt.show()
        nClusters = int(input('Enter the number of clusters: '))
    
    s = np.diag(s)
    sqrtS = np.sqrt(s)
    z = u.dot(sqrtS)
    if directed == True:
        z = np.c_[z,v.dot(sqrtS)]
    
    km = KMeans(n_clusters=nClusters)
    clusterId = km.fit_predict(z)
    
    return clusterId

def estimateBlockProb(adj,clusterId,directed=False):
    nClusters = np.max(clusterId)+1
    clusterSizes = np.histogram(clusterId,
                                bins=np.max(clusterId)+1)[0].astype(float)
    
    # Number of formed edges in each block
    nEdgesBlock = np.zeros((nClusters,nClusters))
    for c1 in range(nClusters):
        inC1 = np.where(clusterId == c1)[0]
        for c2 in range(nClusters):
            inC2 = np.where(clusterId == c2)[0]
            nEdgesBlock[c1,c2] = np.sum(adj[inC1[:,np.newaxis],inC2])
    
    # Number of node pairs (possible edges) in each block
    nPairsBlock = clusterSizes[:,np.newaxis].dot(clusterSizes[np.newaxis,:])
    # Reduce the number of possible pairs for diagonal block pairs since no
    # self-edges are permitted
    nPairsBlock[np.eye(nClusters,dtype=bool)] -= clusterSizes
    # For undirected graphs, halve the number of edges and pairs along
    # diagonal block pairs
    if directed == False:
        nEdgesBlock[np.eye(nClusters,dtype=bool)] /= 2
        nPairsBlock[np.eye(nClusters,dtype=bool)] /= 2
    
    # Edge probabilities at the block level
    blockProb = nEdgesBlock/nPairsBlock
    
    # Compute log-likelihood: compute only over lower diagonal blocks for
    # undirected graphs to avoid duplicating block pairs
    blockMask = np.full((nClusters,nClusters),True)
    if directed == False:
        blockMask = np.tril(blockMask)
    logLik = np.sum(nEdgesBlock*np.log(blockProb) + (nPairsBlock-nEdgesBlock)
                    * np.log(1-blockProb))
    
    return (blockProb,logLik)