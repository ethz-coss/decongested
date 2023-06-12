import torch
import pickle

from dqn_agent import model, ReplayMemory

from environment import roadGridOnline

import numpy as np
from pathlib import Path
import copy
import plotting
from grids import generator_functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(n_iter, next_destination_method="simple", exploration_method="random", agents_see_iot_nodes=True,
         save_path="experiments", grid_name="uniform"):
    SIZE = 4

    G = generator_functions.generate_4x4_grids(costs=grid_name)

    print([node for node in G.nodes()])

    ENVIRONMENT = f"4x4_grid_{grid_name}"
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
    EPS_DECAY = N_ITER / 10000  # larger is slower
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer
    EXPLORATION_METHOD = exploration_method
    AGENTS = f"dqn_{N_OBSERVATIONS}_exploration_{EXPLORATION_METHOD}_iot_{AGENTS_SEE_IOT_NODES}"
    TRAINING_SETTINGS = f"N{N_AGENTS}_dex-{NEXT_DESTINATION_METHOD}_I{N_ITER}_B{BATCH_SIZE}_EXP{EPS_START - EPS_END}_G{GAMMA}_LR{LR}"
    PATH = f"{save_path}/{ENVIRONMENT}/{AGENTS}_{TRAINING_SETTINGS}"
    Path(PATH).mkdir(parents=True, exist_ok=True)

    print(PATH)

    # initialized centralized agent
    with open(f"{PATH}/data", "rb") as file:
        data = pickle.load(file)
    agent = model(N_OBSERVATIONS, N_ACTIONS, DEVICE, LR=1e-4, TAU=0.005, gamma=0.9, BATCH_SIZE=1024, max_memory=10000000,
                  hidden1=4096, hidden2=2048)

    # push data to agent memory buffer
    for d in data.values():
        transitions = d["transitions"]
        for pair in transitions:
            n, t = pair
            agent.memory.push(
                t["state"],
                t["action"],
                t["next_state"],
                t["reward"])

    # train centralized agent
    training_iterations = 10000
    for step in range(training_iterations):
        agent.optimize_model()

    del data  # remove data from memory

    agent.memory = ReplayMemory(10000)  # clear buffer for storage
    with open(f"{PATH}/agent", "wb") as file:
        pickle.dump(agent, file)

    print(PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int)
    parser.add_argument('next_destination_method', type=str)
    parser.add_argument('exploration_method', type=str)
    parser.add_argument('--iot_nodes', action="store_true", default=False)
    parser.add_argument('save_path', type=str)
    parser.add_argument('grid_name', type=str)
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
        save_path=args.save_path,
        grid_name=args.grid_name
    )
