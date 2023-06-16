import numpy as np
from scipy.stats import entropy


def extract_transitions_from_data(data):
    transitions = [data[t]["transitions"] for t in data.keys()]
    return transitions


def extract_moving_average_from_data(data):
    moving_average = [data[t]["average_trip_time"] for t in data.keys()]
    return moving_average


def count_state_action_visits(transitions, size, n_actions, n_agents):
    """
    :param transitions: list of lists, of tuples (n "agent_index", Transition "named tuple")
    :param size: int length of square grid
    :param n_actions: int number of actions
    :param n_agents: int number of agents
    :return: dictionary with agent index keys and values as arrays of state action visit counts
    """

    state_action_visits = {
        n: np.zeros((size, size, n_actions)) for n in range(n_agents)
    }
    for steps in transitions:
        for tup in steps:
            n, transition = tup

            state = transition["state"]
            state = state.squeeze(0)
            state_x, state_y = state[0:4], state[4:8]
            amax_x, amax_y = state_x.argmax(0), state_y.argmax(0)
            x, y = int(amax_x), int(amax_y)

            action = transition["action"]
            action = action.squeeze(0)
            amax_action = action.argmax(0)
            a = int(amax_action)

            state_action_visits[n][x, y, a] += 1
    return state_action_visits


def calculate_driver_entropy(state_action_visits):
    """

    :param state_action_visits: dictionary with agent index keys and values as arrays of state action visit counts
    :return: dictionary with agent index keys and values as entropy of state action visit arrays
    """
    H = {}

    for agent, visits in state_action_visits.items():
        total_actions = visits.sum()
        flat_visits = (visits/total_actions).flatten()
        h = entropy(flat_visits)
        H[agent] = h
    return H


def extract_normalized_trip_lengths_per_agent(trips, n_agents):
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

    return Y, travel_times, travel_steps


if __name__ == "__main__":
    from dqn_grid_online import compose_path
    import argparse
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int)
    parser.add_argument('next_destination_method', type=str)
    parser.add_argument('exploration_method', type=str)
    parser.add_argument('save_path', type=str)  # main directory
    parser.add_argument('grid_name', type=str)  # subdirectory
    parser.add_argument('centralized_ratio', type=float)
    parser.add_argument('internal_save_path', type=str)  # where to save the processed data
    parser.add_argument('--iot_nodes', action="store_true", default=False)
    parser.add_argument('--with_agent_ids', action="store_true", default=False)
    parser.add_argument('--non_stationary', action="store_true", default=False)
    args = parser.parse_args()

    path = compose_path(
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

    # online trips
    with open(f"{path}/trips", "rb") as file:
        trips = pickle.load(file)

    per_agent_interpolated_trip_lengths, _, _ = extract_normalized_trip_lengths_per_agent(trips, n_agents=100)
    system_interpolated_trip_lengths = per_agent_interpolated_trip_lengths.mean(0)
    average_trip_length_during_training = system_interpolated_trip_lengths.mean()
    average_trip_length_end_of_training = system_interpolated_trip_lengths[-1000:-1].mean()
    variance_trip_length_end_of_training = system_interpolated_trip_lengths[-1000:-1].var()

    del trips

    # online data
    with open(f"{path}/data", "rb") as file:
        data = pickle.load(file)

    transitions = extract_transitions_from_data(data)
    state_action_visits = count_state_action_visits(transitions, size=4, n_actions=4, n_agents=100)
    empirical_entropy = calculate_driver_entropy(state_action_visits)
    average_empirical_entropy_training = np.mean(np.array([driver_entropy for driver_entropy in empirical_entropy.values()]))

    moving_average = extract_moving_average_from_data(data)
    moving_average_all_training = np.array(moving_average).mean()
    moving_average_final_training = np.array(moving_average[-1000:-1]).mean()

    del data

    evaluations_path = f"{path}/evaluations"
    # evaluate trips
    with open(f"{evaluations_path}/trips", "rb") as file:
        trips = pickle.load(file)

    per_agent_interpolated_trip_lengths, _, _ = extract_normalized_trip_lengths_per_agent(trips, n_agents=100)
    system_interpolated_trip_lengths = per_agent_interpolated_trip_lengths.mean(0)
    average_trip_length_during_testing = system_interpolated_trip_lengths.mean()
    variance_trip_length_during_testing = system_interpolated_trip_lengths.var()

    del trips

    # evaluate data
    with open(f"{evaluations_path}/data_evaluate_ratio_{args.centralized_ratio}", "rb") as file:
        data = pickle.load(file)

    transitions = extract_transitions_from_data(data)
    state_action_visits = count_state_action_visits(transitions, size=4, n_actions=4, n_agents=100)
    empirical_entropy = calculate_driver_entropy(state_action_visits)
    average_empirical_entropy_testing = np.mean(
        np.array([driver_entropy for driver_entropy in empirical_entropy.values()]))

    moving_average = extract_moving_average_from_data(data)
    moving_average_all_testing = np.array(moving_average).mean()

    del data

    row = {
        "grid": args.grid_name,
        "dex": args.next_destination_method,
        "exploration": args.exploration_method,
        "epsilon": 0,
        "iot-nodes": args.iot_nodes,
        "ratio": args.centralized_ratio,
        "online_all": average_trip_length_during_training,
        "online_end": average_trip_length_end_of_training,
        "online_var": variance_trip_length_end_of_training,
        "evaluate": average_trip_length_during_testing,
        "evaluate_var": variance_trip_length_during_testing,
        "ma_all_training": moving_average_all_training,
        "ma_final_training": moving_average_final_training,
        "ma_all_testing": moving_average_all_testing,
        "entropy_training": average_empirical_entropy_training,
        "entropy_testing": average_empirical_entropy_testing,
    }

    Path(args.internal_save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{args.internal_save_path}/row_{args.grid_name}_{args.next_destination_method}_{args.exploration_method}"
              f"_{args.iot_nodes}_{args.centralized_ratio}", "wb") as file:
        pickle.dump(row, file)
