import matplotlib.pyplot as plt
import numpy as np


def generate_plots(trips, n_agents, PATH):
    travel_times = {}
    travel_steps = {}
    max_step = 0
    for agent, trip in trips.items():
        trip_time = np.array([metrics[1] for metrics in trip]).flatten()  # extract the travel time
        trip_step = np.array([metrics[0] for metrics in trip]).flatten().astype(int)  # extract the steps
        trip_freeflow_length = np.abs(
            np.diff(np.array([metrics[2] for metrics in trip[1:]]), axis=0).sum(axis=1).sum(axis=1))
        trip_freeflow_length = np.concatenate([np.array([6]), trip_freeflow_length])
        normalization = np.where(trip_freeflow_length == 0, 2, trip_freeflow_length)
        max_step = max(max_step, trip_step[-1])
        travel_times[agent] = np.diff(trip_time) / normalization
        travel_steps[agent] = trip_step[1:]

    x = np.arange(0, max_step)
    Y = np.zeros((n_agents, int(max_step)))
    for agent, times in travel_times.items():
        y = np.interp(x, travel_steps[agent], times)
        Y[agent, :] = y
        plt.plot(x, y)

    plt.ylabel("trip length")
    plt.xlabel("step")
    plt.savefig(f"{PATH}/trip_lengths_timeseries.png")
    plt.close()

    plt.plot(Y.mean(axis=0))
    plt.ylabel("average trip time/distance")
    plt.xlabel("step")
    plt.savefig(f"{PATH}/system_performance_timeseries.png")
    plt.close()

    plt.hist([len(steps) for steps in travel_times.values()])
    plt.ylabel("frequency")
    plt.xlabel("completed trips")
    plt.savefig(f"{PATH}/completed_trips_histogram.png")
    plt.close()

    plt.hist([steps[-1] for steps in travel_steps.values()])
    plt.ylabel("frequency")
    plt.xlabel("total steps")
    plt.savefig(f"{PATH}/trip_length_histogram.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', type=str)
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

    ENVIRONMENT = f"4x4_grid_{args.grid_name}"

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
    EPS_DECAY = N_ITER / 10000  # larger is slower
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer
    AGENTS = f"dqn_{N_OBSERVATIONS}_exploration_{EXPLORATION_METHOD}_iot_{AGENTS_SEE_IOT_NODES}"
    TRAINING_SETTINGS = f"N{N_AGENTS}_dex-{NEXT_DESTINATION_METHOD}_I{N_ITER}_B{BATCH_SIZE}_EXP{EPS_START - EPS_END}_G{GAMMA}_LR{LR}"
    PATH = f"{args.save_path}/{ENVIRONMENT}/{AGENTS}_{TRAINING_SETTINGS}"

    if os.path.exists(f"{PATH}/trips"):
        with open(f"{PATH}/trips", "rb") as file:
            trips = pickle.load(file)
    else:
        print(f"path error: {PATH}/trips not found")

    generate_plots(
        trips=trips,
        n_agents=N_AGENTS,
        PATH=PATH
    )
