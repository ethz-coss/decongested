import matplotlib.pyplot as plt
import matplotlib.animation
import pickle
import numpy as np
import networkx as nx
import seaborn as sns
from IPython.display import HTML


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

    plt.plot(Y.mean(axis=0))
    plt.ylabel("average trip time/distance")
    plt.xlabel("step")
    plt.savefig(f"{PATH}/system_performance_timeseries.png")

    plt.hist([len(steps) for steps in travel_times.values()])
    plt.ylabel("frequency")
    plt.xlabel("completed trips")
    plt.savefig(f"{PATH}/completed_trips_histogram.png")

    plt.hist([steps[-1] for steps in travel_steps.values()])
    plt.ylabel("frequency")
    plt.xlabel("total steps")
    plt.savefig(f"{PATH}/trip_length_histogram.png")
