import os
import pickle

from dqn_grid_online import compose_path
from plotting import extract_normalized_trip_lengths_per_agent

SAVE_PATH = "/cluster/scratch/ccarissimo/decongested"
N_ITER = 400000

for GRID in ["uniform", "random", "braess"]:
    for NEXT_DESTINATION_METHOD in ["simple", "one-way", "random", "work-commute"]:
        if GRID == "braess" and NEXT_DESTINATION_METHOD != "one-way":
            continue
        for EXPLORATION_METHOD in ["random", "neighbours"]:
            for IOT_NODES in [True, False]:
                path = compose_path(
                    save_path=SAVE_PATH,
                    grid_name=GRID,
                    n_observations=25,
                    exploration_method=EXPLORATION_METHOD,
                    agents_see_iot_nodes=IOT_NODES,
                    n_agents=100,
                    next_destination_method=NEXT_DESTINATION_METHOD,
                    n_iter=N_ITER,
                    batch_size=64,
                    eps_start=0.5,
                    eps_end=0.05,
                    gamma=0.9,
                    lr=1e-2
                )

                with open(f"{path}/trips", "rb") as file:
                    trips = pickle.load(file)

                Y = extract_normalized_trip_lengths_per_agent(trips, n_agents=100)
