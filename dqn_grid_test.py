import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pickle

from dqn_agent import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from environment import roadGrid
    from networkx import grid_graph
    import networkx as nx
    import numpy as np

    # size = 4
    # G = grid_graph(dim=(size, size))
    # for e in G.edges():
    #     G.edges[e]["cost"] = lambda x: 1 + x / 100

    # G = nx.DiGraph()
    # G.add_nodes_from([(0, 0), (0, 1), (1, 0), (1, 1)])
    # G.add_edges_from([
    #     ((0, 0), (0, 1)),
    #     # ((0, 1), (0, 0)),
    #     ((0, 0), (1, 0)),
    #     # ((1, 0), (0, 0)),
    #     ((1, 0), (1, 1)),
    #     # ((1, 1), (1, 0)),
    #     ((0, 1), (1, 1)),
    #     # ((1, 1), (0, 1)),
    #     # ((0, 1), (1, 0)),
    #     # ((1, 0), (0, 1)),
    # ])
    # for e in G.edges():
    #     G.edges[e]["cost"] = lambda x: 0.2
    #
    # G.adj[(0, 0)][(0, 1)]["cost"] = lambda x: 0.5
    # G.adj[(0, 1)][(1, 1)]["cost"] = lambda x: 0.5

    size = 4
    G = nx.DiGraph()
    G.add_nodes_from([(0, 0), (1, 1), (2, 2), (3, 3)])
    G.add_edges_from([
        ((0, 0), (1, 1), {"cost": lambda x: x/100}),
        ((0, 0), (2, 2), {"cost": lambda x: 1}),
        ((1, 1), (2, 2), {"cost": lambda x: 0}),
        ((1, 1), (3, 3), {"cost": lambda x: 1}),
        ((2, 2), (3, 3), {"cost": lambda x: x/100}),
    ])

    # positions = {node: node for node in G.nodes()}
    # fig, ax = plt.subplots(figsize=(8, 6))
    # nx.draw(G, pos=positions, ax=ax)
    # plt.show()

    LOAD_SAVED_DRIVERS = True

    n_agents = 100
    n_actions = 4
    n_states = 4
    n_iter = 1000
    episode_timeout = 16

    BATCH_SIZE = 4
    GAMMA = 0.9
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = n_iter * 2  # larger is slower
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer
    NAME = f"test_dqn_grid_N{n_agents}_I{n_iter}_S{len(G.nodes())}_B{BATCH_SIZE}_EXP{EPS_START - EPS_END}_G{GAMMA}_LR{LR}"

    n_observations = int(size * 2) + 2 * n_actions + 1  # state space

    if LOAD_SAVED_DRIVERS:
        drivers_file = "drivers_dqn_grid_egreedy_iot_N100_I2000_S16_B4_EXP0.85_G0.9_LR0.01"
        with open(drivers_file, "rb") as file:
            drivers = pickle.load(file)

    else:
        drivers = {}
        for n in range(n_agents):
            drivers[n] = model(
                n_observations, n_actions, device, LR, TAU, GAMMA, BATCH_SIZE=BATCH_SIZE, max_memory=1000
            )

    data = {}

    env = roadGrid(graph=G, n_agents=n_agents, n_actions=n_actions, size=size, timeout=episode_timeout)

    state, info, base_state, agents_at_base_state = env.reset()

    done = False
    while not done:

        action_list = [
            driver.select_action(
                state=torch.tensor(state[n], dtype=torch.float32, device=device),
                EPS_END=0,
                EPS_START=0,
                EPS_DECAY=1).unsqueeze(0)
            for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]
        A = torch.cat(action_list)
        actions = A.cpu().numpy()

        state, base_state, agents_at_base_state, transitions, done = env.step(actions, drivers)

    print("success", env.agents_at_final_state.mean())

    data[0] = {
        "trajectory": env.trajectory
    }

    with open(f"trajectory_{NAME}", "wb") as file:
        pickle.dump(data, file)

    # with open(f"drivers_{NAME}", "wb") as file:
    #     pickle.dump(drivers, file)

    # plt.figure(0, figsize=(10, 6))
    # plt.plot([data[t]["success"].mean() for t in data.keys()])
    # plt.savefig(f"dqn_grid_success_{NAME}.pdf")
    #
    # plt.show()
