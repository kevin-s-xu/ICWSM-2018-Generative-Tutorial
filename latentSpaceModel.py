# -*- coding: utf-8 -*-
"""
Functions for working with latent space models.

@author: Kevin S. Xu
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.optimize import minimize
import networkx as nx
from sklearn.manifold import MDS

def generateAdj(pos,bias=0):
    logistic = lambda x: 1 / (1 + np.exp(-x))
    
    affinity = bias - pdist(pos)
    dyadProb = logistic(affinity)
    
    adj = np.random.rand(np.size(dyadProb))
    maskEdges = adj < dyadProb
    adj[maskEdges] = 1
    adj[np.logical_not(maskEdges)] = 0
    
    adj = squareform(adj)
    return adj

def logLikelihood(adj,pos,bias):
    # Convert adjacency matrix to vectorized form of lower triangle to
    # remove redundant entries
    if adj.ndim > 1:
        adj = squareform(adj)
    
    affinity = bias - pdist(pos)
    logLik = np.sum( affinity*adj - np.log(1 + np.exp(affinity)) )
    
    return logLik

def negLogLikFun(x,adj,nNodes,dim):
    pos = np.reshape(x[:-1],(nNodes,dim))
    bias = x[-1]
    return -logLikelihood(adj,pos,bias)

def estimateParams(adj,dim=2,initBias=0):
    nNodes = np.shape(adj)[0]
    
    # Find lengths of shortest paths between all pairs of nodes. Initialize
    # all shortest path distances to -1 to easily find disconnected node
    # pairs, which need to be handled separately.
    print('Initializing latent node positions by multidimensional scaling')
    initDist = -np.ones((nNodes,nNodes))
    net = nx.Graph(adj)
    iLengths = nx.all_pairs_shortest_path_length(net)
    for source,lengths in iLengths:
        for target in lengths.keys():
            initDist[source,target] = lengths[target]
    # Set node pairs with no paths to 2x diameter (required if network is
    # disconnected)
    initDist[initDist==-1] = 2*np.max(initDist)
    
    # Choose initial position by multidimensional scaling on the matrix of
    # all shortest paths
    netMDS = MDS(n_components=dim,dissimilarity='precomputed')
    initPos = netMDS.fit_transform(initDist)
    
    # Iterative maximize likelihood using quasi-Newton approach
    print('Iteratively maximizing likelihood using quasi-Newton approach')
    adj = squareform(adj)
    xInit = np.r_[np.ravel(initPos),initBias]
    optRes = minimize(negLogLikFun,xInit,method='BFGS',args=(adj,nNodes,dim))
    pos = np.reshape(optRes.x[:-1],(nNodes,dim))
    bias = optRes.x[-1]
    logLik = -optRes.fun
    
    return pos,bias,logLik,optRes