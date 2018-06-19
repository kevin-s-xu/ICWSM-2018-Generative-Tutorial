# -*- coding: utf-8 -*-
"""
Functions for working with stochastic block models.

@author: Kevin S. Xu
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import bernoulli as bern
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def generateSbm(clusterId,blockProb,directed=False):
    nNodes = np.shape(clusterId)[0]
    nClusters = np.shape(blockProb)[0]
    
    adj = np.zeros((nNodes,nNodes))
    for c1 in range(nClusters):
        inC1 = np.where(clusterId == c1)[0]
        nNodesC1 = np.size(inC1)
        # Form edges between nodes in same cluster (if more than 1 node)
        if nNodesC1 > 1:
            # Generate Bernoulli samples for lower triangular portion
            adjBlock = np.zeros((nNodesC1,nNodesC1))
            indicesL = np.tril_indices(nNodesC1,-1)
            nIndices = np.shape(indicesL)[1]
            adjBlock[indicesL] = bern.rvs(blockProb[c1,c1],size=nIndices)
            indicesU = np.triu_indices(nNodesC1,1)
            if directed == True:
                # Generate Bernoulli samples for upper triangular portion
                adjBlock[indicesU] = bern.rvs(blockProb[c1,c1],size=nIndices)
            else:
                # Copy lower triangle to upper triangle
                adjBlock[indicesU] = adjBlock.T[indicesU]
            # Copy block to adjacency matrix
            adj[inC1[:,np.newaxis],inC1] = adjBlock
            
        # For edges between nodes in different clusters. Loop start index
        # depends on whether graph is directed or undirected.
        if directed == True:
            startIdx = 0
        else:
            startIdx = c1+1
        for c2 in range(startIdx,nClusters):
            # Diagonal block pairs were already considered, so ignore them
            if c2 == c1:
                continue
            
            inC2 = np.where(clusterId == c2)[0]
            nNodesC2 = np.size(inC2)
            if nNodesC2 == 0:
                continue
            adj[inC1[:,np.newaxis],inC2] = bern.rvs(blockProb[c1,c2],
                                                    size=(nNodesC1,nNodesC2))
            if directed == False:
                # Copy block over to lower diagonal portion of adjacency
                # matrix
                adj[inC2[:,np.newaxis],inC1] = adj[inC1[:,np.newaxis],inC2].T
    
    return adj
    
def spectralCluster(adj,nClusters=0,directed=False):
    # Compute largest 20 singular values and vectors of adjacency matrix
    u,s,v = svds(adj,k=20)
    v = v.T
    
    # Sort in decreasing order of magnitude
    sortId = np.argsort(-s)
    s = s[sortId]
    u = u[:,sortId]
    v = v[:,sortId]
    
    if nClusters == 0:
        # User didn't pre-select the number of clusters so plot the top 20
        # singular values and let the user enter the number of clusters
        plt.ion()
        plt.plot(s[:20],'x')
        plt.show()
        nClusters = int(input('Enter the number of clusters: '))
    
    # Truncate matrices of singular vectors to include only nClusters leading
    # singular vectors
    s = s[:nClusters]
    u = u[:,:nClusters]
    v = v[:,:nClusters]
    
    s = np.diag(s)
    sqrtS = np.sqrt(s)
    z = u.dot(sqrtS)
    if directed == True:
        z = np.c_[z,v.dot(sqrtS)]
    
    km = KMeans(n_clusters=nClusters,n_init=10)
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
