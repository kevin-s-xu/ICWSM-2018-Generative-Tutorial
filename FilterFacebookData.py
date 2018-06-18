# -*- coding: utf-8 -*-
"""
Script to import adjacency matrices from Facebook data and filter out
low-degree nodes to leave a small network with only a few hundred nodes.

This script should be applied to Facebook friendship and wall post data
collected by Viswanath et al. (2009) to prepare it for the ICWSM 2018
tutorial demos.

Data source: http://socialnetworks.mpi-sws.org/data-wosn2009.html

@author: Kevin S. Xu
"""

frCutoff = 300
wallCutoff = 75

import networkx as nx
import numpy as np

#%% Import friendship network from edge list
# Create graph from edge list in text file. The edge list is in format
#   profileUser friend timestamp
# so we load the timestamp as an edge attribute (of type string, because some
# are unknown, labeled as '\N').
print('Loading friendship edge list')
fbFr = nx.read_edgelist('facebook-links.txt',data=(('Timestamp',str),))
print('%i nodes' % fbFr.number_of_nodes())
print('Removing all nodes with degree less than %i' % frCutoff)
frDeg = dict(fbFr.degree)
for nodeId,deg in frDeg.items():
    if deg < frCutoff:
        fbFr.remove_node(nodeId)
print('%i nodes' % fbFr.number_of_nodes())
print('Saving filtered friendship edge list')
#nx.write_edgelist(fbFr,'facebook-links-filtered.txt',data=False)
fbFrAdj = nx.to_numpy_matrix(fbFr)
np.savetxt('facebook-links-filtered-adj.txt',fbFrAdj,fmt='%i')

#%% Import wall post network from edge list
# Create digraph from edge list in text file. The edge list is in format
#   recipient poster timestamp
# while NetworkX expects
#   fromNode toNode
# so we load the timestamp as an edge attribute (of type int) and then
# reverse the digraph so that the edges are in the proper direction.
print('Loading wall edge list')
fbWall = nx.read_edgelist('facebook-wall.txt',create_using=nx.DiGraph(),
                          data=(('Timestamp',int),)).reverse()
print('Removing all self-edges')
selfEdges = nx.selfloop_edges(fbWall)
fbWall.remove_edges_from(selfEdges)
print('%i nodes' % fbWall.number_of_nodes())
print('Removing all nodes with both in- and out-degree less than %i'
      % wallCutoff)
wallInDeg = dict(fbWall.in_degree)
wallOutDeg = dict(fbWall.out_degree)
for nodeId,inDeg in wallInDeg.items():
    outDeg = wallOutDeg[nodeId]
    if (inDeg < wallCutoff) and (outDeg < wallCutoff):
        fbWall.remove_node(nodeId)
print('%i nodes' % fbWall.number_of_nodes())
print('Removing all nodes that now have both in- and out-degree 0')
wallNodes = list(fbWall.nodes)
for nodeId in wallNodes:
    if (fbWall.in_degree[nodeId] == 0) and (fbWall.out_degree[nodeId] == 0):
        fbWall.remove_node(nodeId)
print('%i nodes' % fbWall.number_of_nodes())
print('Saving filtered wall post edge list')
#nx.write_edgelist(fbWall,'facebook-wall-filtered.txt',data=False)
fbWallAdj = nx.to_numpy_matrix(fbWall)
np.savetxt('facebook-wall-filtered-adj.txt',fbWallAdj,fmt='%i')

