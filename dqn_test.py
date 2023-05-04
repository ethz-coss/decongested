import torch

import tqdm
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt

from agent_functions import bellman_update_q_table, e_greedy_select_action
import numpy as np

from run_functions import *
from run_functions import *
import copy
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #         print("forward", x.shape)
        #         x = x.T
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


steps_done = 0


def select_action(state, policy_net):  # epsilon greedy action selection, perfect
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)  # epsilon is decayed manually
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(-1)[1].view(1, 1)  # changed max(1) to max(0) fixing bug
    else:
        return torch.tensor([[np.random.randint(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        # print("here")
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    next_state_batch = [s for s in batch.next_state if s is not None]
    if len(next_state_batch) > 0:
        non_final_next_states = torch.stack(next_state_batch)
    else:
        non_final_next_states = None

    #     print("batch.next_state", batch.next_state)
    #     print("nfns,", non_final_next_states)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # print(state_batch, action_batch)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(0))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if len(next_state_batch) > 0:
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad(set_to_none=True)  # set to none option has slightly different behaviour than setting to 0
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == "__main__":
    from environment import roadNetwork

    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4])
    G.add_edges_from([
        (0, 1, {"cost": lambda x: x}),
        (0, 2, {"cost": lambda x: 100}),
        (1, 2, {"cost": lambda x: 0}),
        (1, 3, {"cost": lambda x: 100}),
        (2, 3, {"cost": lambda x: x}),
        (3, 1, {"cost": lambda x: 100}),
        (3, 2, {"cost": lambda x: x}),
        # (2, 4, {"cost": lambda x: 50 + x}),
        # (4, 3, {"cost": lambda x: x})
    ])

    positions = {0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 1), 4: (1.5, 0)}

    edge_labels = {
            (0, 1): "0",
            (0, 2): "1",
            (1, 2): "0",
            (1, 3): "1",
            (2, 3): "0",
            (2, 4): "1",
            (4, 3): "0"
        }

    # fig, ax = plt.subplots(figsize=(8, 6))
    # nx.draw_networkx_edges(G, pos=positions, ax=ax)
    # nx.draw_networkx_nodes(G, pos=positions, ax=ax)
    # nx.draw_networkx_edge_labels(G, pos=positions, ax=ax,
    #                              edge_labels=edge_labels, font_color="red")
    # nx.draw_networkx_labels(G, pos=positions, ax=ax, font_color="white")
    # plt.show()

    n_agents = 100
    n_actions = 2
    n_states = 5
    n_iter = 1000

    BATCH_SIZE = 16
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = n_agents * n_iter/3
    TAU = 0.05  # TAU is the update rate of the target network
    LR = 1e-2  # LR is the learning rate of the AdamW optimizer

    state, info = torch.Tensor([0, 0, 0, 0, 1]), {}  # env.reset()
    n_observations = 5  # state space

    drivers = {}
    for n in range(n_agents):
        drivers[n] = {}
        drivers[n]["policy_net"] = DQN(n_observations, n_actions).to(device)
        drivers[n]["target_net"] = DQN(n_observations, n_actions).to(device)
        drivers[n]["target_net"].load_state_dict(drivers[n]["policy_net"].state_dict())  # not sure what this is for
        drivers[n]["optimizer"] = optim.AdamW(drivers[n]["policy_net"].parameters(), lr=LR, amsgrad=True)
        drivers[n]["memory"] = ReplayMemory(100)

    print(
        [driver["policy_net"](torch.tensor(np.array([0, 0, 0, 0, 1]), dtype=torch.float32, device=device).unsqueeze(0)) for
         driver in drivers.values()])

    data = {}

    env = roadNetwork(graph=G, n_agents=n_agents)

    for t in range(n_iter):
        state, info, base_state, agents_at_base_state = env.reset()

        done = False
        while not done:

            A = torch.cat([
                select_action(state=torch.tensor(state[n], dtype=torch.float32, device=device),
                              policy_net=attr["policy_net"]).unsqueeze(0) for n, attr in drivers.items() if
                agents_at_base_state[n]])
            actions = A.cpu().numpy()

            state, base_state, agents_at_base_state, transitions, done = env.step(actions)

            for n, transition in transitions:
                s, a, s_, r = transition
                drivers[n]["memory"].push(s, a, s_, r)
                # print(drivers[n]["memory"].memory)

        # Training Loop once all agents have Terminated
        for driver in drivers.values():
            # Perform one step of the optimization (on the policy network)
            # print("optimize")
            optimize_model(
                driver["memory"],
                driver["policy_net"],
                driver["target_net"],
                driver["optimizer"]
            )
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = driver["target_net"].state_dict()
            policy_net_state_dict = driver["policy_net"].state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[
                    key] * (1 - TAU)
            driver["target_net"].load_state_dict(target_net_state_dict)

        if t % 100 == 0:
            print("step: ", t, "welfare: ", np.mean(env.T), "exploration rate:", EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY))

        ## SAVE PROGRESS DATA[agents]
        data[t] = {
            "T": env.T,
            "S": env.S,
            "A": env.actions_taken,  # convert it to numpy from a tensor only at the end of the simulation
            "trajectory": env.trajectory
        }

    beliefs = [
        driver["policy_net"](torch.tensor(np.array([0, 0, 0, 0, 1]), dtype=torch.float32, device=device).unsqueeze(0)) for
        driver in drivers.values()]
    print(beliefs)

    with open("dqn_test_data", "wb") as file:
        pickle.dump(data, file)

    with open("dqn_test_drivers", "wb") as file:
        pickle.dump(drivers, file)

    plt.figure(0, figsize=(10, 6))
    plt.plot([data[i]["T"].mean() for i in data.keys()])
    plt.savefig("dqn_bp_average_travel_time.pdf")

    plt.figure(1, figsize=(10, 8))
    plt.plot([data[i]["A"][0] for i in data.keys()], label=["0->1", "0->2"])
    plt.plot([data[i]["A"][1] for i in data.keys()], label=["1->2", "1->3"])
    # plt.plot([data[i]["A"][2][0] for i in data.keys()], label="2->3")
    plt.title("actions from states")
    plt.legend()
    plt.savefig("dqn_bp_path_choices.pdf")

    fig, ax = plt.subplots(figsize=(8, 6), **{"num": 2})
    nx.draw_networkx_edges(G, pos=positions, ax=ax)
    nx.draw_networkx_nodes(G, pos=positions, ax=ax)
    nx.draw_networkx_edge_labels(G, pos=positions, ax=ax,
                                 edge_labels=edge_labels, font_color="red")
    nx.draw_networkx_labels(G, pos=positions, ax=ax, font_color="white")

    plt.show()

# while np.where(S == final_states, False, True).sum() > 0:
#     remaining_agents = np.where(S != final_states, True, False)
#
#     next_agents = np.where(T == T[remaining_agents].min(), True, False) * remaining_agents
#
#     uni = np.unique(S[next_agents])
#     done = np.zeros(n_agents)
#
#     for s in uni:
#
#         edges = [neighbour[1] for neighbour in G.edges(s)]
#
#         agents_at_s = np.where(S == s, True, False) * next_agents
#
#         A = torch.cat([
#             select_action(state=torch.tensor(self.state_binary[S[n]], dtype=torch.float32, device=device),
#                           policy_net=attr["policy_net"]).unsqueeze(0) for n, attr in drivers.items() if
#             agents_at_s[n]])
#
#         A = torch.clip(A, 0, len(edges) - 1)
#         A = torch.flatten(A)
#         # print(A, type(A), s)
#         counts = torch.bincount(A, minlength=n_actions).cpu().numpy()
#         actions_taken[s] += counts
#
#         rewards = [G.adj[s][neighbour[1]]["cost"](counts[i]) for i, neighbour in enumerate(G.edges(s))]
#
#         R = np.array([rewards[a] for a in A])
#
#         S[agents_at_s] = np.array([edges[a] for a in A]).astype(int)
#
#         observations = np.array([edges[a] for a in A]).astype(int)
#
#         # reward = -np.mean(R)
#
#         for i, n in enumerate(np.argwhere(agents_at_s == True).flatten()):
#             # print("in loop")
#             driver = drivers[n]
#             observation = state_binary[observations[i]]
#             action = A[i]
#             reward = -R[i]
#             terminated = True if (observation == state_binary[final_states[n]]).all() else False
#             truncated = False
#
#             reward = torch.tensor([reward], device=device)
#             done[n] = terminated or truncated
#
#             if terminated:
#                 next_state = None
#             else:
#                 next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # .unsqueeze(0)
#
#             # Store the transition in memory
#             state = torch.tensor(state_binary[S[n]], dtype=torch.float32, device=device).unsqueeze(0)
#             action = action.unsqueeze(0)  # torch.tensor(A, dtype=torch.int64, device=device).unsqueeze(0)
#
#             driver["memory"].push(state, action, next_state, reward)
#
#         T[agents_at_s] += R
#
#         trajectory.append(tuple([s, counts, rewards, edges]))
