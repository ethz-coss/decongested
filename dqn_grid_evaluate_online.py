import torch
import pickle

from dqn_agent import model, ReplayMemory

from environment import roadGridOnline

import numpy as np
from pathlib import Path
import copy
import plotting
from grids import generator_functions

from dqn_grid_online import compose_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(n_iter, next_destination_method="simple", exploration_method="random", agents_see_iot_nodes=True,
         save_path="experiments", grid_name="uniform", train=False, centralized_ratio=0):
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

    PRETRAINED_MODEL = f"{PATH}/drivers"
    with open(PRETRAINED_MODEL, "rb") as file:
        drivers = pickle.load(file)
    for driver in drivers.values():
        driver.steps_done = 0
        driver.memory = ReplayMemory(10000)

    if centralized_ratio > 0:
        with open(f"{PATH}/agent", "wb") as file:
            agent = pickle.load(file)
    centralized_mask = np.random.binomial(n=1, p=centralized_ratio, size=N_AGENTS).astype(bool)

    PATH = f"{PATH}/evaluations"
    Path(PATH).mkdir(parents=True, exist_ok=True)

    data = {}

    env = roadGridOnline(
        graph=G,
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        size=SIZE,
        next_destination_method=NEXT_DESTINATION_METHOD,
        agents_see_iot_nodes=AGENTS_SEE_IOT_NODES
    )

    EVALUATE_ITER = 20000
    state, info, base_state, agents_at_base_state = env.reset()
    for t in range(EVALUATE_ITER):

        action_list = [
            driver.select_action(
                state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE),
                EPS_END=0,
                EPS_START=0,
                EPS_DECAY=1,
                method=EXPLORATION_METHOD,
                neighbour_beliefs=None).unsqueeze(0)
            for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]

        agents_that_receive_centralized_recommendation = np.argwhere(agents_at_base_state * centralized_mask)

        if centralized_ratio > 0:
            for n in agents_that_receive_centralized_recommendation:
                action_list[n] = agent.select_action(
                    state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE),
                    EPS_END=0,
                    EPS_START=0,
                    EPS_DECAY=1,
                    method="random",
                    neighbour_beliefs=None).unsqueeze(0)

        A = torch.cat(action_list)
        actions = A.cpu().numpy()

        state, base_state, agents_at_base_state, transitions, done = env.step(actions, drivers)

        if train:
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

    plotting.generate_plots(env.trips, N_AGENTS, PATH)

    with open(f"{PATH}/data_evaluate_ratio({centralized_ratio})", "wb") as file:
        pickle.dump(data, file)

    # for driver in drivers.values():
    #     driver.memory = ReplayMemory(10000)  # clear buffer for storage
    # with open(f"{PATH}/drivers", "wb") as file:
    #     pickle.dump(drivers, file)

    with open(f"{PATH}/trips_evaluate_ratio({centralized_ratio})", "wb") as file:
        pickle.dump(dict(env.trips), file)  # calling `dict' to offset defaultdict lambda for pickling

    with open(f"{PATH}/trajectory_evaluate_ratio({centralized_ratio})", "wb") as file:
        pickle.dump(env.trajectory, file)

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
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('centralized_ratio', type=float)
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
        grid_name=args.grid_name,
        train=args.train,
        centralized_ratio=args.centralized_ratio
    )
