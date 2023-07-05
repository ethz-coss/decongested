<img src="images/logo.png" align="left" width="45%"/>
<img src="images/coci_logo.png" align="center" width="45%"/>

# DeCongested: learning to share for self-organized logistics

## Overview
DeCongested is a repository built and maintained by the Computational Social Science group at ETHZ, supported by the ’Co-Evolving City Life - CoCi’ project through the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 833168). 

The main research question guiding this research is:
> How can self-organization and resilience principles be used in a way that makes cities more sustainable?

This particular project focuses on transport and logistics networks and studies the ability of learning agents to leverage self-organization and continuously learn to share networked resources. This investigation involves frontier research at the intersection of game theory, machine learning, traffic science, and complex systems.

### Repository Contents
- dqn_agent contains the functions to run a Deep Q-Learning model
- dqn_grid runs a DQN agent in the specified environment
- dqn_grid_results is a helper jupyter notebook designed to plot and analyze the results from runs from dqn_grid
- environment contains the environment classes that implement the decentralized and asynchronous multi-player atomic routing simulation 


## Introduction
Multi agent systems require coordination to achieve system optimal performance. Except for few simple cases, this coordination can not be achieved without explicit collaboration, reliably. 

Cities create the playground within which a rich multi agent system can develop. Within this multi agent system we can identify on the basis of simplified perspectives that there are many simultaneous, concurrent 'games' that agents engage in. By games, we intend the game theoretic understanding of a game: a setting defined by a set of players, a set of actions, and a means of assigning value/utility/payoff/cost to the possible outcomes as constrained by a the chosen actions of all players. Games are simplified settings within which we may begin to quantitatively study the complex playgrounds that cities create.

In this project we focus on a the particular games that can be found within transportation and logistics, that can be found whenever there are shared resources that many seek to use simultaneously, but that all wish they could be the sole users of. We are thinking particularly of route choice in cities, pickup and delivery in cities, and packet routing on the internet. Over the years, each of these domains has been 'gameified', and described through the lens of congestion games.

Congestion games are in their simplest form a game where $N$ players share a set of $K$ resources, where the utility of using resource $k$ given that $m$ players pick it is a monotonically decreasing function $u_k(m)$ in $m$. In other words, when many pick the same resource they 'congest' and reduce the experienced utility for each other. We can readily imagine such situations in traffic: when many drivers pick the same road they create traffic jams which slow everyone down. 

What is even more interesting in Cities, and why such research has relevance beyond the theoretical results it can provide, is that traffic jams on single roads can have consequences that extend beyond those roads, even beyond traffic: e.g. drivers change their paths to avoid that road, drivers take to walking or public transport, pollution levels increase which may affect the value of real estate. Cities are open systems, which means that the games played locally by a subset of individuals may have consequences for the players of the many concurrent games that occur in the neighbourhood of the game.

In this paper we take MDP congestion games as our ground model, create a simulation framework to test reinforcement learning algorithms, and explore the online collaborative learning potential of reinforcement learning agents that learn to be more circular over time by sharing their knowledge and experience of the networked repeated congestion games that they play.

**Research Questions**
- How can self-organization and resilience principles be used in a way that makes cities more sustainable?
- How can local interaction rules enable effective Online Collaborative MARL?
- How can circularity be defined in the context of logistics and transportation?

## Preliminary Results

<img src="images/congestion_in_non_stationary_logistics_networks.png" align="center" width="100%"/>

## Background

### Circular Economy
The EU defines the Circular Economy as:

> `a model of production and consumption, which involves sharing, leasing, reusing, repairing, refurbishing and recycling existing materials and products as long as possible. In this way, the life cycle of products is extended'.

In this definition are featured the producers and consumers in an economy, but we miss a crucial component which connects the two: the supply chain. Supply chains are to the economy what the electrical grid is to the internet, what the arteries and veins are to animal bodies.

The circular economy defined for the producers and consumers has the clear target of the products. Producers should find ways to reuse and recycle materials in their pipelines. Consumers should find ways to similarly, reuse and share products they have purchased and seek to repair and refurbish their used goods. Again this is a very product centered, materials focused perspective which is necessary for this relationship. But if we think of supply chains, the infrastructure of the economy, which is not concerned with the products themselves, but with the distribution and connections that enable these products to be exchanged; what is then circularity for a supply chain?

In this research we will focus on information as a key component of supply chain circularity in its effects on the effective use of shared resources. The information available to the participants in a supply chain is often partial and incomplete which can lead to an ineffective use of share resources like road networks, shipping containers, and warehouses. Moreover, the participants in supply chains can be both independent actors in competition and coordinated players in of the same company. Therefore, a variety of tools from game theory can be useful to analyze the systems. In this research we will focus on the case of shared resource games, also known as congestion games, which have been often used to model congestion in traffic on roads and the internet. We will consider cases where the shared resources are subject to exogenous shocks and assess the ability of the supply network to adapt and recover from the changes to the underlying network. In particular, we will assess the differences in dynamics that are due to a degree of centralization: a fully decentralized network is one where all actors are independent and make their decisions by considering information available to them locally, while a fully centralized network is guided by a unique decision makers which has access to all the global information to coordinate the individual actors.

### Routing, route choice, congestion games

**Congestion Games** complexity of pure NE \cite{fabrikant2004complexity}. For symmetric games, pure NE can be computed in polynomial time with the potential function. For general games complexity is PLS-complete. Atomic Congestion Games introduced in \cite{tekin2012atomic}.

**MDP Congestion Games tolling**, changing the reward function, to acoid convestion \cite{li2019tolling}. Considered continuous MDPCG. Also \cite{calderone2017markov} do similar work, with continuous MDPCG. Extended in \cite{li2022congestion} to robot path planning to avoid congestion in warehouses.

**Decentralized Training, Online Learning** \cite{gaborattention} preprint proposing a method for decentralized training. 

**Logistics, Pick-up and Delivery Problems**
A review classification and survey of Static PDP \cite{berbeglia2007static}. Large Neighbourhood Search Heuristics first introduced in \cite{shaw1998using} and very successful for solving routing problems. PDP-TW, for pickups with time windows \cite{ropke2006adaptive}. PDPTW-SL for pickup with time windows and scheduled lines of public transport \cite{ghilas2016adaptive}. Centralized Approaches. Hardly Adaptive. 

**Demand Adaptive Systems** Desiging the master schedule \cite{crainic2012designing}.

### Reinforcement learning

**COMA** achieves difference style rewards \cite{tumer2007distributed}, for counterfactuals, such that agents can compare the what if scenarios of their alternative actions. However, this relies on a centralized critic trained during training. Also from Foerster \cite{foerster2018counterfactual}.

**Learning to Communicate, RIAL** \cite{foerster2016learning} Few agents. Create Deep Learning protocol for differentiable message passing between agents, and solve some simple multi player games: switch riddle (3 and 4 agents), and MNIST games (2 agents). Highly Cited Research.

**MACKRL** uses a common knowledge framework to train a multi agent system of reinforcement learners \cite{schroeder2019multi}. Simply assumes a common knowledge function, accessible in both training and testing to the independent RL agents, in a Dec-POMDP.

**Collective Intelligence for Deep Learning**
\cite{ha2022collective} survey state of the art ways in which multi agent systems are developed with deep learning, for large systems with MANY agents. Many agent reinforcement learning. They break it down into Image Processing, Deep Reinforcement Learning, Multi Agent Learning, and Meta Learning. Cool self-organization on GraphNNs in \cite{grattarola2021learning}. 

