# Generative Models for Social Media Analytics: Networks, Text, and Time
ICWSM 2018 Tutorial

Monday, June 25: 1:30pm - 5:30pm

## Abstract

Traditional social network models aim to understand social phenomena by studying graphs which represent the connections between individuals. In the age of social media, in which many of our social interactions are recorded digitally, social network data have become much richer and more complex. It has become increasingly clear that our models need to go beyond a single network, to include aspects such as textual and temporal information, and to handle data with multiple relations. Generative probabilistic models are well suited for such analyses of social media data, as they provide a natural framework for reasoning collectively over multiple data modalities.

This tutorial presents recent advances in generative models for social media analytics, focusing on models that encode social phenomena with latent (i.e. hidden) attributes, which are subsequently recovered from data. The tutorial begins with a review of generative models for social networks, including latent space models, block models, and modern variants of these such as mixed membership models. The second part of the tutorial showcases richer models for social media data that include text and dynamics, alongside illustrative case studies. The tutorial aims to serve a multidisciplinary audience, including scholars from both the social and computational sciences.

## Organizers

Kevin S. Xu is an assistant professor in the EECS Department at the University of Toledo. His main research interests are in machine learning and statistical signal processing with applications to network science and human dynamics. He received his PhD in 2012 from the University of Michigan.

James R. Foulds (Jimmy) is an assistant professor in the Department of Information Systems at the University of Maryland, Baltimore County. His research interests are in machine learning, focusing on probabilistic latent variable models and the inference algorithms to learn them from social networks and text data. 

## Tutorial Outline

The tutorial will consist of 3 parts:

1:30pm-2:15pm: Mathematical representations and generative models for social networks

- Introduction to generative approach
- Connections to sociological principles

15 min break

2:30pm-3:30pm: Fitting generative social network models to data

- Application scenarios with demos
- Model selection and evaluation

30 min coffee break

4:00pm-5:30pm: Rich generative models for social media data

- Network models augmented with text and dynamics
- Case studies on social media data

## Demos

In the second part of the tutorial, we will be providing demos on fitting generative models to some Facebook data collected by Viswanath et al. The entire data set contains over 60,000 nodes and is available at http://socialnetworks.mpi-sws.org/data-wosn2009.html. We will be making use of subsets of the data, so downloading the entire data set is not necessary.

### Requirements

These demos are written in Python and will make use of many scientific computing packages in Python, including NumPy, SciPy, and NetworkX. We recommend installing the [Anaconda distribution](https://www.anaconda.com/download/), which already includes all necessary packages. We will be running demos on the Python 3.6 version of Anaconda in IPython using the Spyder IDE (both included with Anaconda).

### Demo 1: Facebook wall post network

- 101 nodes, directed: edge from node `i` to `j` denotes that `i` posted on `j`'s wall.
- Contains all nodes with at least in- or out-degree of 75, i.e. node `i` is included if `i` has posts to their wall from at least 75 other nodes or if node `i` made posts to the walls of at least 75 other people.
- Data file (adjacency matrix in text format): `facebook-wall-filtered-adj.txt`
- Demo Python script: `FacebookWallSbmDemo.py`

### Demo 2: Facebook friendship network

- 106 nodes, undirected: edge between nodes `i` and `j` denotes a friendship between them.
- Contains all nodes with at least 300 friends.
- Data file (adjacency matrix in text format): `facebook-links-filtered-adj.txt`
- Demo Python script: `FacebookFriendsSbmLsmDemo.py`

