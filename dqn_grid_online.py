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


def compose_path(save_path, grid_name, n_observations, exploration_method, agents_see_iot_nodes,
                 n_agents, next_destination_method, n_iter, batch_size, eps_start, eps_end, gamma, lr):
    ENVIRONMENT = f"4x4_grid_{grid_name}"
    AGENTS = f"dqn_{n_observations}_exploration_{exploration_method}_iot_{agents_see_iot_nodes}"
    TRAINING_SETTINGS = f"N{n_agents}_dex-{next_destination_method}_I{n_iter}_B{batch_size}_EXP{eps_start - eps_end}_G{gamma}_LR{lr}"
    PATH = f"{save_path}/{ENVIRONMENT}/{AGENTS}_{TRAINING_SETTINGS}"
    return PATH


def train_decentralized_agents(n_iter, drivers, env, exploration_method, path, eps_start=0.9, eps_end=0.05):
    eps_decay = n_iter/10000
    n_agents = len(drivers)

    data = {}

    state, info, base_state, agents_at_base_state = env.reset()
    for t in range(n_iter):
        if exploration_method == "neighbours":
            neighbour_beliefs = {
                tuple(state[i]): [
                    driver.judge_state(
                        state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE))
                    for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]
                for i in range(n_agents) if agents_at_base_state[i]
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
                EPS_END=eps_end,
                EPS_START=eps_start,
                EPS_DECAY=eps_decay,
                method=exploration_method,
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
            "average_trip_time": env.average_trip_time,
            "transitions": transitions,
        }

    plotting.generate_plots(env.trips, N_AGENTS, PATH, internal_save_path="test_plots")

    with open(f"{path}/data", "wb") as tmp_file:
        pickle.dump(data, tmp_file)

    for driver in drivers.values():
        driver.memory = ReplayMemory(10000)  # clear buffer for storage
    with open(f"{path}/drivers", "wb") as tmp_file:
        pickle.dump(drivers, tmp_file)

    with open(f"{path}/trips", "wb") as tmp_file:
        pickle.dump(dict(env.trips), tmp_file)  # calling `dict' to offset defaultdict lambda for pickling

    with open(f"{path}/trajectory", "wb") as tmp_file:
        pickle.dump(env.trajectory, tmp_file)

    print(PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int)
    parser.add_argument('next_destination_method', type=str)
    parser.add_argument('exploration_method', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('grid_name', type=str)
    parser.add_argument('--iot_nodes', action="store_true", default=False)
    parser.add_argument('--train', action="store_true", default=False)
    args = parser.parse_args()

    SIZE = 4
    N_AGENTS = 100
    N_ACTIONS = 4
    N_STATES = 4
    N_OBSERVATIONS = int(SIZE * 2) * 2 + 2 * N_ACTIONS + 1  # state space
    EPISODE_TIMEOUT = 16
    BATCH_SIZE = 64
    GAMMA = 0.9
    EPS_START = 0.5
    EPS_END = 0.05
    EPS_DECAY = args.n_iter / 10000  # larger is slower
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer
    PATH = compose_path(
        save_path=args.save_path,
        grid_name=args.grid_name,
        n_observations=25,
        exploration_method=args.exploration_method,
        agents_see_iot_nodes=args.iot_nodes,
        n_agents=100,
        next_destination_method=args.next_destination_method,
        n_iter=args.n_iter,
        batch_size=64,
        eps_start=0.5,
        eps_end=0.05,
        gamma=0.9,
        lr=1e-2
    )
    Path(PATH).mkdir(parents=True, exist_ok=True)
    print(PATH)

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

    G = generator_functions.generate_4x4_grids(costs=args.grid_name)
    env = roadGridOnline(
        graph=G,
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        size=SIZE,
        next_destination_method=args.next_destination_method,
        agents_see_iot_nodes=args.iot_nodes
    )

    train_decentralized_agents(
        n_iter=args.n_iter,
        drivers=drivers,
        env=env,
        exploration_method=args.exploration_method,
        path=PATH,
    )
