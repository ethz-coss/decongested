import os.path

import torch
import pickle

from dqn_agent import model, ReplayMemory

from environment import roadGridOnline

import numpy as np
from pathlib import Path
import copy
import plotting
from grids import generator_functions
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compose_path(save_path, grid_name, n_observations, exploration_method, agents_see_iot_nodes,
                 n_agents, next_destination_method, n_episodes, n_iter, batch_size, eps_start, eps_end, gamma, lr,
                 averaging_method):
    ENVIRONMENT = f"4x4_grid_{grid_name}"
    AGENTS = f"dqn_{n_observations}_exploration_{exploration_method}_iot_{agents_see_iot_nodes}"
    # TRAINING_SETTINGS = f"N{n_agents}_dex-{next_destination_method}_I{n_iter}_B{batch_size}_EXP{eps_start - eps_end}_G{gamma}_LR{lr}"
    TRAINING_SETTINGS = f"N{n_agents}_dex-{next_destination_method}_E{n_episodes}_I{n_iter}_FL{averaging_method}"
    PATH = f"{save_path}/{ENVIRONMENT}/{AGENTS}_{TRAINING_SETTINGS}"
    return PATH


def get_unique_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename

    filename, ext = os.path.splitext(base_filename)
    index = 1
    while True:
        new_filename = f"{filename}_{index}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        index += 1


def save_pickle_with_unique_filename(data, filename):
    unique_filename = get_unique_filename(filename)
    with open(unique_filename, 'wb') as file:
        pickle.dump(data, file)


def average_weights(w):  # expects list of state_dict objects from models
    """
    Returns the average of the weights.
    taken from https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/update.py#L54
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def train_decentralized_agents(n_episodes, n_iter, drivers, env, exploration_method, path, eps_start=0.9, eps_end=0.05,
                               averaging_method="None"):
    eps_decay = n_episodes/4
    n_agents = len(drivers)
    episode_counter = 0

    data = {}

    state, info, base_state, agents_at_base_state = env.reset()
    for episode in range(n_episodes):

        data[episode] = {}  # initialize episode dict
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)

        for t in range(n_iter):
            neighbour_beliefs = {
                tuple(state[n]):
                    driver.judge_state(
                        state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE))
                    for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]
            }

            # for state_tuple, beliefs in neighbour_beliefs.items():
            #     beliefs = torch.stack(beliefs)
            #     beliefs = beliefs.cpu().numpy()

            action_list = [
                driver.select_action(
                    state=torch.tensor(state[n], dtype=torch.float32, device=DEVICE),
                    EPS_END=eps_threshold,
                    EPS_START=eps_threshold,
                    EPS_DECAY=1,
                    method=exploration_method,
                    neighbour_beliefs=None).unsqueeze(0)
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

            if averaging_method is not None:
                if averaging_method == "agents_at_base_node":
                    w = [driver.policy_net.state_dict() for n, driver in enumerate(drivers.values()) if agents_at_base_state[n]]
                    if len(w) > 0:
                        w_avg = average_weights(w)
                        for n, driver in enumerate(drivers.values()):
                            if agents_at_base_state[n]:
                                driver.policy_net.load_state_dict(w_avg)
                        # check if averaging worked
                        # base_state_indices = np.argwhere(agents_at_base_state)
                        # print(drivers[int(base_state_indices[0])].policy_net.state_dict().values(), drivers[int(base_state_indices[-1])].policy_net.state_dict().values())
                elif averaging_method == "agents_not_at_base_node":
                    w = [driver.policy_net.state_dict() for n, driver in enumerate(drivers.values()) if not
                         agents_at_base_state[n]]
                    if len(w) > 0:
                        w_avg = average_weights(w)
                        for n, driver in enumerate(drivers.values()):
                            if not agents_at_base_state[n]:
                                driver.policy_net.load_state_dict(w_avg)
                elif averaging_method == "all_agents":
                    w = [driver.policy_net.state_dict() for n, driver in enumerate(drivers.values())]
                    if len(w) > 0:
                        w_avg = average_weights(w)
                        for n, driver in enumerate(drivers.values()):
                            driver.policy_net.load_state_dict(w_avg)
                elif averaging_method == "None":
                    pass
                else:
                    raise f"averaging method '{averaging_method}' not found"

            episode_method = "fixed_step"
            if episode_method == "fixed_step":
                if done:
                    print("episode: ", episode_counter, "welfare: ", np.mean(env.T, axis=0), "success rate:",
                          env.reached_destinations.mean(),
                          "exploration rate:", drivers[0].eps_threshold)
                    current_state, info, base_state, agents_at_base_state = env.reset()
                    episode_counter += 1
                    break
            else:
                if t % 100 == 0:
                    print("step: ", t, "welfare: ", env.average_trip_time, "success rate:", env.reached_destinations.mean(),
                          "exploration rate:", drivers[0].eps_threshold)

            # SAVE PROGRESS DATA[agents]
            data[episode][t] = {
                "average_trip_time": env.average_trip_time,
                "transitions": transitions,
                "q-values": neighbour_beliefs
            }

    # plotting.generate_plots(env.trips, N_AGENTS, PATH, internal_save_path=path)

    save_pickle_with_unique_filename(data, f"{path}/data")
    for driver in drivers.values():
        driver.memory = ReplayMemory(0)
    save_pickle_with_unique_filename(drivers, f"{path}/drivers")
    save_pickle_with_unique_filename(dict(env.trips), f"{path}/trips")
    save_pickle_with_unique_filename(env.trajectory, f"{path}/trajectory")

    print(PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('n_episodes', type=int)
    parser.add_argument('n_iter', type=int)
    parser.add_argument('next_destination_method', type=str)
    parser.add_argument('exploration_method', type=str)
    parser.add_argument('averaging_method', type=str)
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
        n_episodes=args.n_episodes,
        n_iter=args.n_iter,
        batch_size=64,
        eps_start=0.5,
        eps_end=0.05,
        gamma=0.9,
        lr=1e-2,
        averaging_method=args.averaging_method
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

    G = generator_functions.generate_4x4_grids(costs=args.grid_name, seed=0)
    env = roadGridOnline(
        graph=G,
        n_agents=N_AGENTS,
        n_actions=N_ACTIONS,
        size=SIZE,
        next_destination_method=args.next_destination_method,
        agents_see_iot_nodes=args.iot_nodes
    )

    train_decentralized_agents(
        n_episodes=args.n_episodes,
        n_iter=args.n_iter,
        drivers=drivers,
        env=env,
        exploration_method=args.exploration_method,
        path=PATH,
        averaging_method=args.averaging_method
    )
