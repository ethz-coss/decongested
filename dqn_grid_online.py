import copy
import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pickle

from dqn_agent import model, ReplayMemory

from environment import roadGridOnline
from networkx import grid_graph
import networkx as nx
import numpy as np
from pathlib import Path
import copy
import plotting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(n_iter, next_destination_method="simple", exploration_method="random", agents_see_iot_nodes=True,
         save_path="experiments"):
    SIZE = 4
    G = grid_graph(dim=(SIZE, SIZE))
    for e in G.edges():
        G.edges[e]["cost"] = lambda x: 1 + x / 100
    ENVIRONMENT = f"symmetric_grid_S{len(G.nodes())}"
    Path(f"{save_path}/{ENVIRONMENT}").mkdir(parents=True, exist_ok=True)

    N_AGENTS = 100
    N_ACTIONS = 4
    N_STATES = 4
    N_OBSERVATIONS = int(SIZE * 2) * 2 + 2 * N_ACTIONS + 1  # state space
    EPISODE_TIMEOUT = 16
    NEXT_DESTINATION_METHOD = next_destination_method

    AGENTS_SEE_IOT_NODES = agents_see_iot_nodes

    N_ITER = n_iter
    BATCH_SIZE = 64
    GAMMA = 0.9
    EPS_START = 0.5
    EPS_END = 0.05
    EPS_DECAY = N_ITER / 1000  # larger is slower
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer
    EXPLORATION_METHOD = exploration_method
    AGENTS = f"dqn_{N_OBSERVATIONS}_exploration_{EXPLORATION_METHOD}_iot_{AGENTS_SEE_IOT_NODES}"
    TRAINING_SETTINGS = f"N{N_AGENTS}_dex-{NEXT_DESTINATION_METHOD}_I{N_ITER}_B{BATCH_SIZE}_EXP{EPS_START - EPS_END}_G{GAMMA}_LR{LR}"
    PATH = f"{save_path}/{ENVIRONMENT}/{AGENTS}_{TRAINING_SETTINGS}"
    Path(PATH).mkdir(parents=True, exist_ok=True)

    LOAD_PRETRAINED_MODEL = True
    PRETRAINED_MODEL = f"pretrained_models/drivers_single_driver_random_N1_I200000_S16_B64_EXP0.5700000000000001_G0.9_LR0.01"
    if LOAD_PRETRAINED_MODEL:
        with open(PRETRAINED_MODEL, "rb") as file:
            drivers = pickle.load(file)
        for driver in drivers.values():
            driver.steps_done = 0
            driver.memory = ReplayMemory(10000)
        driver = drivers[0]
        for n in range(N_AGENTS):
            drivers[n] = copy.deepcopy(driver)
    else:
        drivers = {}
        for n in range(N_AGENTS):
            drivers[n] = model(
                N_OBSERVATIONS, N_ACTIONS, DEVICE, LR, TAU, GAMMA, BATCH_SIZE=BATCH_SIZE, max_memory=10000
            )

    data = {}

    env = roadGridOnline(
        graph=G,
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        size=SIZE,
        next_destination_method=NEXT_DESTINATION_METHOD,
        agents_see_iot_nodes=AGENTS_SEE_IOT_NODES
    )

    state, info, base_state, agents_at_base_state = env.reset()
    for t in range(N_ITER):
        if EXPLORATION_METHOD == "neighbours":
            neighbour_beliefs = {
                tuple(state[i]): [
                    driver.judge_state(
                        state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE))
                    for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]
                for i in range(N_AGENTS) if agents_at_base_state[i]
            }

            for state_tuple, beliefs in neighbour_beliefs.items():
                beliefs = torch.stack(beliefs)
                beliefs = beliefs.cpu().numpy()
                beliefs = np.mean(beliefs, axis=0)
                neighbour_beliefs[state_tuple] = beliefs
        else:
            neighbour_beliefs = None

        action_list = [
            driver.select_action(
                state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE),
                EPS_END=EPS_END,
                EPS_START=EPS_START,
                EPS_DECAY=EPS_DECAY,
                method=EXPLORATION_METHOD,
                neighbour_beliefs=neighbour_beliefs).unsqueeze(0)
            for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]
        A = torch.cat(action_list)
        actions = A.cpu().numpy()

        state, base_state, agents_at_base_state, transitions, done = env.step(actions, drivers)

        for n, transition in transitions:
            drivers[n].memory.push(
                transition["state"],
                transition["action"],
                transition["next_state"],
                transition["reward"])
            drivers[n].optimize_model()

        if t % 100 == 0:
            print("step: ", t, "welfare: ", env.average_trip_time, "success rate:", env.reached_destinations.mean(),
                  "exploration rate:", drivers[0].eps_threshold)

        # SAVE PROGRESS DATA[agents]
        data[t] = {
            # "T": env.T,
            # "S": env.S,
            "average_trip_time": env.average_trip_time,
            "transitions": transitions,
        }

    # plotting.generate_plots(env.trips, N_AGENTS, PATH)

    with open(f"{PATH}/data", "wb") as file:
        pickle.dump(data, file)

    for driver in drivers.values():
        driver.memory = ReplayMemory(10000)  # clear buffer for storage
    with open(f"{PATH}/drivers", "wb") as file:
        pickle.dump(drivers, file)

    with open(f"{PATH}/trips", "wb") as file:
        pickle.dump(dict(env.trips), file)  # calling `dict' to offset defaultdict lambda for pickling

    with open(f"{PATH}/trajectory", "wb") as file:
        pickle.dump(env.trajectory, file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int)
    parser.add_argument('next_destination_method', type=str)
    parser.add_argument('exploration_method', type=str)
    parser.add_argument('iot_nodes', type=bool)
    parser.add_argument('save_path', type=bool)
    args = parser.parse_args()

    N_ITER = args.n_iter
    NEXT_DESTINATION_METHOD = args.next_destination_method
    EXPLORATION_METHOD = args.exploration_method
    AGENTS_SEE_IOT_NODES = args.iot_nodes

    main(
        n_iter=N_ITER,
        next_destination_method=NEXT_DESTINATION_METHOD,
        exploration_method=EXPLORATION_METHOD,
        agents_see_iot_nodes=AGENTS_SEE_IOT_NODES,
        save_path=args.save_path
    )
