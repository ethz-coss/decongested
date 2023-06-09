import numpy as np
import torch
from collections import defaultdict, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class roadGridOnline:
    def __init__(self, graph, n_agents, n_actions, size, next_destination_method, agents_see_iot_nodes):
        self.agents_see_iot_nodes = agents_see_iot_nodes
        self.next_destination_method = next_destination_method
        self.work_commute_destinations = {}
        if self.next_destination_method == "work-commute":
            # generate work-commute OD pairs
            for n in range(n_agents):
                origin = np.random.randint(0, size, size=2)
                destination = np.random.randint(0, size, size=2)
                while (origin == destination).all():
                    destination = np.random.randint(0, size, size=2)
                self.work_commute_destinations[n] = [origin, destination]

        self.destinations_counts = None
        self.stored_transitions = None
        self.current_state = None
        self.states_counts = None
        self.reached_destinations = None
        self.last_agent_transits = None
        self.agents_in_transit = None
        self.step_counter = None
        self.done = None
        self.trajectory = None
        self.actions_taken = None
        self.destinations = None
        self.S = None
        self.T = None
        self.base_state = None
        self.agents_at_base_state = None
        self.G = graph
        self.n_agents = n_agents
        self.n_actions = n_actions  # max node degree
        self.n_states = len(self.G.nodes)  # to be defined
        self.size = size
        self.one_hot_enc = {(l, r): np.concatenate(
            [np.array([0 if i != l else 1 for i in range(self.size)]),
             np.array([0 if i != r else 1 for i in range(self.size)])]) for
            (l, r) in self.G.nodes()}
        self.training_visitation_counts = defaultdict(self._return_0)
        self.node_last_utilization = {node: np.ones(self.n_actions) / self.n_actions for node in self.G.nodes}
        self.trips = defaultdict(lambda: [tuple([0, 0, np.array([0, 0])])])
        self.average_trip_time = self.size * 2

    def reset(self):
        self.step_counter = np.zeros(self.n_agents)
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros((self.n_agents, 2)).astype(int)  # two dim, for two dim states

        self.destinations = np.ones((self.n_agents, 2)) * np.array([self.size - 1, self.size - 1])
        if self.next_destination_method == "work-commute":
            for agent, od_pair in self.work_commute_destinations.items():
                next_destination = od_pair.pop()
                self.destinations[agent] = next_destination
                od_pair.insert(0, next_destination)

        self.actions_taken = np.zeros((self.size, self.size, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        self.stored_transitions = {}
        remaining_agents = np.ones(self.n_agents, dtype=bool)
        first_agents = remaining_agents
        uni = np.unique(self.S[first_agents], axis=0)

        self.base_state = tuple(uni[np.random.randint(len(uni))])
        self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True, False) * first_agents
        self.agents_in_transit = defaultdict(self._return_0)
        self.last_agent_transits = defaultdict(self._return_none)
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0,
                                         zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))

        self.current_state = [self.partial_observation(agent) for agent in range(self.n_agents)]

        return self.current_state, info, self.base_state, self.agents_at_base_state

    def partial_observation(self, agent):
        tuple_state = tuple(self.S[agent])
        node = self.one_hot_enc[tuple_state]
        pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
        edges = self.G.edges(tuple_state)
        destination = self.one_hot_enc[tuple(self.destinations[agent])]

        utilization = np.zeros(self.n_actions)
        if self.agents_see_iot_nodes:
            utilization = self.node_last_utilization[tuple_state]

        if len(edges) == 4:
            agents_on_outgoing_edges = np.array(
                [self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])
        else:
            edges = list(edges) + [self.base_state for i in range(4 - len(edges))]
            agents_on_outgoing_edges = np.array(
                [self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])

        return np.concatenate(
            [node, destination, pct_occupied, agents_on_outgoing_edges, utilization])

    def _return_0(self):
        return 0

    def _return_none(self):
        return None

    def step(self, actions, drivers):
        # pre-process actions
        actions = actions.flatten()
        counts = np.bincount(actions, minlength=self.n_actions)
        self.actions_taken[self.base_state[0], self.base_state[1]] += counts

        ema_coefficient = 0.9
        pct_counts = counts / counts.sum()
        self.node_last_utilization[tuple(self.base_state)] = ema_coefficient * self.node_last_utilization[
            tuple(self.base_state)] + (1 - ema_coefficient) * pct_counts

        # calculate rewards
        neighbour_nodes = [edge[1] for edge in self.G.edges(self.base_state)]
        reward_per_action = [self.G.adj[self.base_state][edge[1]]["cost"](
            counts[i] + self.agents_in_transit[edge]) for i, edge in enumerate(self.G.edges(self.base_state))]
        if len(neighbour_nodes) < self.n_actions:
            reward_per_action = np.concatenate(
                [reward_per_action, 2 * np.ones((self.n_actions - len(neighbour_nodes)))])
            for i in range(self.n_actions - len(neighbour_nodes)):
                neighbour_nodes.append(self.base_state)
        rewards = np.array([reward_per_action[a] for a in actions])

        # update current states and agents in transit
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        destinations, destinations_counts = np.unique(self.destinations, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0,
                                         zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))
        drivers_destination_counts = defaultdict(self._return_0,
                                                 zip([tuple(s) for s in destinations], list(destinations_counts)))
        edges_actions = {(self.base_state, neighbour): counts[a] if a < len(neighbour_nodes) else None for a, neighbour
                         in enumerate(neighbour_nodes)}
        self.trajectory.append(tuple(
            [self.base_state, dict(self.states_counts), dict(edges_actions), rewards, dict(drivers_destination_counts)]
        ))
        self.agents_in_transit.update(edges_actions)
        next_nodes = [neighbour_nodes[a] for a in actions]
        self.S[self.agents_at_base_state] = np.array(next_nodes).astype(int)
        self.T[self.agents_at_base_state] += rewards
        for a, node in enumerate(neighbour_nodes):
            self.training_visitation_counts[node] += counts[a]

        # prepare (s, a, s_, r) tuples for independent memory buffers
        for i, n in enumerate(np.argwhere(self.agents_at_base_state == True).flatten()):
            self.last_agent_transits[n] = tuple([self.base_state, tuple(next_nodes[i])])
            state = torch.tensor(self.current_state[int(n)], dtype=torch.float32, device=device).unsqueeze(0)
            action = actions[i]
            action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)
            # observation = self.partial_observation(int(n))
            terminated = True if (next_nodes[i] == self.destinations[n]).all() else False
            state_ = None  # if terminated else torch.tensor(observation, dtype=torch.float32, device=device)
            reward = 0 if terminated else -rewards[i]
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            truncated = False
            self.done[n] = terminated or truncated
            self.stored_transitions[n] = {
                "state": state,
                "action": action,
                "next_state": state_,
                "reward": reward
            }
            self.step_counter[n] += 1

        # determine terminated agents, the fastest agents, and base state
        self.reached_destinations = (self.S == self.destinations).all(axis=1)
        done = False
        fastest_travel_time = self.T.min() * 1.05  # 5% threshold to evaluate the next agents that travel
        fastest_agents = self.T <= fastest_travel_time
        uni = np.unique(self.S[fastest_agents], axis=0)
        self.base_state = tuple(uni[np.random.randint(len(uni))])
        self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True,
                                             False) * fastest_agents

        # update agents in transit, and complete transitions of agents in transit
        base_state_indices = np.argwhere(self.agents_at_base_state == True)
        transitions = []
        complete_state = self.partial_observation(int(base_state_indices[0]))

        for n in base_state_indices:
            edge = self.last_agent_transits[int(n)]
            self.agents_in_transit[edge] -= 1
            state_ = torch.tensor(complete_state, dtype=torch.float32, device=device)
            self.stored_transitions[int(n)]["next_state"] = state_
            transitions.append((int(n), self.stored_transitions[int(n)]))
            self.current_state[int(n)] = complete_state

        finished_indices = np.argwhere((self.reached_destinations * fastest_agents) == True)
        for n in finished_indices:
            trip_length = int(self.step_counter[n])
            trip_time = float(self.T[n])
            self.average_trip_time = ema_coefficient * self.average_trip_time + \
                                     (1 - ema_coefficient) * (trip_time - self.trips[int(n)][-1][1])
            destination = self.destinations[n]
            # self.step_counter[n] = 0
            # self.T[n] = 0
            self.trips[int(n)].append(tuple([trip_length, trip_time, destination]))
            self.assign_next_destination(n, method=self.next_destination_method)

        return self.current_state, self.base_state, self.agents_at_base_state, transitions, done

    def assign_next_destination(self, agent, method="random"):
        if method == "commute":
            # self.final_states[agent] =
            pass

        elif method == "simple":
            if (self.destinations[agent] == np.array([0, 0])).all():
                self.destinations[agent] = np.array([self.size - 1, self.size - 1])
            else:
                self.destinations[agent] = np.array([0, 0])

        elif method == "random":
            self.destinations[agent] = np.random.randint(0, self.size, size=2)

        elif method == "work-commute":
            next_destination = self.work_commute_destinations[int(agent)].pop()
            self.work_commute_destinations[int(agent)].insert(0, next_destination)
            self.destinations[int(agent)] = next_destination
        else:
            raise "method not found"

        return None


class roadGrid:
    def __init__(self, graph, n_agents, n_actions, size, timeout):
        self.stored_transitions = None
        self.current_state = None
        self.states_counts = None
        self.agents_at_final_state = None
        self.last_agent_transits = None
        self.agents_in_transit = None
        self.step_counter = None
        self.done = None
        self.trajectory = None
        self.actions_taken = None
        self.final_states = None
        self.S = None
        self.T = None
        self.base_state = None
        self.agents_at_base_state = None
        self.timeout = timeout
        self.G = graph
        self.n_agents = n_agents
        self.n_actions = n_actions  # max node degree
        self.n_states = len(self.G.nodes)  # to be defined
        self.size = size
        self.one_hot_enc = {(l, r): np.concatenate(
            [np.array([0 if i != l else 1 for i in range(self.size)]),
             np.array([0 if i != r else 1 for i in range(self.size)])]) for
            (l, r) in self.G.nodes()}
        self.training_visitation_counts = defaultdict(self._return_0)
        # self.one_hot_enc = {
        #     (0, 0): np.array([0, 0, 0, 1]),
        #     (0, 1): np.array([0, 0, 1, 0]),
        #     (1, 0): np.array([0, 1, 0, 0]),
        #     (1, 1): np.array([1, 0, 0, 0]),
        # }
        self.node_last_utilization = {node: np.ones(self.n_actions) / self.n_actions for node in self.G.nodes}

    def reset(self):
        self.step_counter = np.zeros(self.n_agents)
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros((self.n_agents, 2)).astype(int)  # two dim, for two dim states
        self.final_states = np.ones((self.n_agents, 2)) * np.array(
            [self.size - 1, self.size - 1])  # np.random.randint(0, 7, size=(self.n_agents, 2))
        self.actions_taken = np.zeros((self.size, self.size, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        self.stored_transitions = {}
        remaining_agents = np.ones(self.n_agents, dtype=bool)
        first_agents = remaining_agents
        uni = np.unique(self.S[first_agents], axis=0)

        self.base_state = tuple(uni[np.random.randint(len(uni))])
        self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True, False) * first_agents
        self.agents_in_transit = defaultdict(self._return_0)
        self.last_agent_transits = defaultdict(self._return_none)
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0,
                                         zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))

        self.current_state = [self.partial_observation(agent) for agent in range(self.n_agents)]

        return self.current_state, info, self.base_state, self.agents_at_base_state

    # def counterfactual_observation(self, agent):
    #     tuple_state = tuple(self.S[agent])
    #     node = self.one_hot_enc[tuple_state]
    #     pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
    #
    #     uniform_belief_counterfactual = np.ones(self.n_actions) / self.n_actions
    #
    #     return np.concatenate([node, pct_occupied, uniform_belief_counterfactual])
    # def partial_observation(self, agent):
    #     tuple_state = tuple(self.S[agent])
    #     node = self.one_hot_enc[tuple_state]
    #     pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
    #     edges = self.G.edges(tuple_state)
    #
    #     if len(edges) == 4:
    #         agents_on_outgoing_edges = np.array([self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])
    #     else:
    #         edges = list(edges) + [self.base_state for i in range(4 - len(edges))]
    #         agents_on_outgoing_edges = np.array([self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])
    #
    #     return np.concatenate([node, pct_occupied, agents_on_outgoing_edges])
    # def partial_observation(self, agent):
    #     tuple_state = tuple(self.S[agent])
    #     node = self.one_hot_enc[tuple_state]
    #     pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
    #
    #     return np.concatenate([node, pct_occupied, self.node_last_utilization[tuple_state]])
    # def partial_observation(self, agent):
    #     tuple_state = tuple(self.S[agent])
    #     node = self.one_hot_enc[tuple_state]
    #     pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
    #     return np.concatenate([node, pct_occupied])

    def partial_observation(self, agent):
        tuple_state = tuple(self.S[agent])
        node = self.one_hot_enc[tuple_state]
        pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
        edges = self.G.edges(tuple_state)
        destination = self.one_hot_enc[tuple(self.final_states[agent])]

        if len(edges) == 4:
            agents_on_outgoing_edges = np.array(
                [self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])
        else:
            edges = list(edges) + [self.base_state for i in range(4 - len(edges))]
            agents_on_outgoing_edges = np.array(
                [self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])

        return np.concatenate(
            [node, destination, pct_occupied, agents_on_outgoing_edges, self.node_last_utilization[tuple_state]])

    def _return_0(self):
        return 0

    def _return_none(self):
        return None

    def step(self, actions, drivers):
        # pre-process actions
        actions = actions.flatten()
        counts = np.bincount(actions, minlength=self.n_actions)
        self.actions_taken[self.base_state[0], self.base_state[1]] += counts

        ema_coefficient = 0.9
        pct_counts = counts / counts.sum()
        self.node_last_utilization[tuple(self.base_state)] = ema_coefficient * self.node_last_utilization[
            tuple(self.base_state)] + (1 - ema_coefficient) * pct_counts

        # calculate rewards
        neighbour_nodes = [edge[1] for edge in self.G.edges(self.base_state)]
        reward_per_action = [self.G.adj[self.base_state][edge[1]]["cost"](
            counts[i] + self.agents_in_transit[edge]) for i, edge in enumerate(self.G.edges(self.base_state))]
        if len(neighbour_nodes) < self.n_actions:
            reward_per_action = np.concatenate(
                [reward_per_action, 2 * np.ones((self.n_actions - len(neighbour_nodes)))])
            for i in range(self.n_actions - len(neighbour_nodes)):
                neighbour_nodes.append(self.base_state)
        rewards = np.array([reward_per_action[a] for a in actions])

        # update current states and agents in transit
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0,
                                         zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))
        edges_actions = {(self.base_state, neighbour): counts[a] if a < len(neighbour_nodes) else None for a, neighbour
                         in enumerate(neighbour_nodes)}
        self.trajectory.append(tuple([self.base_state, dict(self.states_counts), dict(edges_actions), rewards]))
        self.agents_in_transit.update(edges_actions)
        destinations = [neighbour_nodes[a] for a in actions]
        self.S[self.agents_at_base_state] = np.array(destinations).astype(int)
        self.T[self.agents_at_base_state] += rewards
        for a, node in enumerate(neighbour_nodes):
            self.training_visitation_counts[node] += counts[a]

        # prepare (s, a, s_, r) tuples for independent memory buffers
        for i, n in enumerate(np.argwhere(self.agents_at_base_state == True).flatten()):
            self.last_agent_transits[n] = tuple([self.base_state, tuple(destinations[i])])
            state = torch.tensor(self.current_state[int(n)], dtype=torch.float32, device=device).unsqueeze(0)
            action = actions[i]
            action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)
            # observation = self.partial_observation(int(n))
            terminated = True if (destinations[i] == self.final_states[n]).all() else False
            state_ = None  # if terminated else torch.tensor(observation, dtype=torch.float32, device=device)
            reward = -rewards[i]
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            truncated = False
            self.done[n] = terminated or truncated
            self.stored_transitions[n] = {
                "state": state,
                "action": action,
                "next_state": state_,
                "reward": reward
            }
            self.step_counter[n] += 1

        # determine terminated agents, the fastest agents, and base state
        self.agents_at_final_state = (self.S == self.final_states).all(axis=1)
        if self.agents_at_final_state.sum() < self.n_agents:
            if (self.step_counter >= self.timeout).any():  # step counter is an array
                done = True
                self.base_state = None
                self.agents_at_base_state = None
                transitions = []
            else:
                done = False
                non_terminated_agents = np.logical_not(self.agents_at_final_state)
                fastest_travel_time = self.T[non_terminated_agents].min()
                fastest_agents = (self.T == fastest_travel_time)
                fastest_non_terminated_agents = fastest_agents * non_terminated_agents
                uni = np.unique(self.S[fastest_non_terminated_agents], axis=0)

                self.base_state = tuple(uni[np.random.randint(len(uni))])
                self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True,
                                                     False) * fastest_non_terminated_agents

                # update agents in transit, and complete transitions of agents in transit
                base_state_indices = np.argwhere(self.agents_at_base_state == True)
                transitions = []

                complete_state = self.partial_observation(int(base_state_indices[0]))

                for n in base_state_indices:
                    edge = self.last_agent_transits[int(n)]
                    self.agents_in_transit[edge] -= 1
                    state_ = torch.tensor(complete_state, dtype=torch.float32, device=device)
                    self.stored_transitions[int(n)]["next_state"] = state_
                    transitions.append((int(n), self.stored_transitions[int(n)]))
                    self.current_state[int(n)] = complete_state
                    # self.step_counter[int(n)] += 1

                finished_indices = np.argwhere((self.agents_at_final_state * fastest_agents) == True)
                for n in finished_indices:
                    # print(n)
                    edge = self.last_agent_transits[int(n)]
                    self.agents_in_transit[edge] -= 1
                    transitions.append((int(n), self.stored_transitions[int(n)]))
                    self.current_state[int(n)] = None
                    # self.step_counter[int(n)] += 1
        else:
            done = True
            self.base_state = None
            self.agents_at_base_state = None
            transitions = []

        # self.step_counter += 1
        # self.current_state = next_state

        return self.current_state, self.base_state, self.agents_at_base_state, transitions, done


class roadNetwork:
    def __init__(self, graph, n_agents):
        self.done = None
        self.trajectory = None
        self.actions_taken = None
        self.final_states = None
        self.S = None
        self.T = None
        self.base_state = None
        self.agents_at_base_state = None
        self.G = graph
        self.n_agents = n_agents
        self.n_actions = 2  # max node degree
        self.n_states = len(graph.nodes)  # to be defined

        self.one_hot_enc = {
            0: np.array([0, 0, 0, 0, 1]),
            1: np.array([0, 0, 0, 1, 0]),
            2: np.array([0, 0, 1, 0, 0]),
            3: np.array([0, 1, 0, 0, 0]),
            4: np.array([1, 0, 0, 0, 0])
        }
        self.one_hot_enc = {(l, r): np.concatenate(
            [np.array([0 if i != l else 1 for i in range(8)]), np.array([0 if i != r else 1 for i in range(8)])]) for
            (l, r) in graph.nodes()}

    def reset(self):
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros(self.n_agents).astype(int)
        self.final_states = np.array([(i % 2) + 2 for i in range(self.n_agents)])
        self.actions_taken = np.zeros((self.n_states, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        state = [self.one_hot_enc[self.S[n]] for n in range(self.n_agents)]

        remaining_agents = np.where(self.S != self.final_states, True, False)
        first_agents = np.where(self.T == self.T[remaining_agents].min(), True, False) * remaining_agents
        uni = np.unique(self.S[first_agents])
        self.base_state = np.random.choice(uni)
        self.agents_at_base_state = np.where(self.S == self.base_state, True, False) * first_agents

        # print(self.base_state, self.agents_at_base_state)

        return state, info, self.base_state, self.agents_at_base_state

    def step(self, actions):
        edges = [neighbour[1] for neighbour in G.edges(self.base_state)]

        actions = np.clip(actions, 0, len(edges) - 1)
        actions = actions.flatten()
        # print(actions)
        counts = np.bincount(actions, minlength=n_actions)
        self.actions_taken[self.base_state] += counts

        reward_per_action = [G.adj[self.base_state][neighbour[1]]["cost"](counts[i]) for i, neighbour in
                             enumerate(G.edges(self.base_state))]

        rewards = np.array([reward_per_action[a] for a in actions])

        self.S[self.agents_at_base_state] = np.array([edges[a] for a in actions]).astype(int)

        observations = np.array([edges[a] for a in actions]).astype(int)

        # reward = -np.mean(rewards)

        transitions = []

        for i, n in enumerate(np.argwhere(self.agents_at_base_state == True).flatten()):
            # print("in loop")
            driver = drivers[n]
            observation = self.one_hot_enc[observations[i]]
            action = actions[i]
            terminated = True if (observation == self.one_hot_enc[self.final_states[n]]).all() else False
            reward = 100 if terminated else -rewards[i]
            truncated = False

            reward = torch.tensor([reward], device=device)
            self.done[n] = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # .unsqueeze(0)

            # Store the transition in memory
            state = torch.tensor(self.one_hot_enc[self.S[n]], dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)

            transitions.append((n, (state, action, next_state, reward)))

        self.T[self.agents_at_base_state] += rewards

        self.trajectory.append(tuple([self.base_state, counts, rewards, edges]))

        next_state = [self.one_hot_enc[self.S[n]] for n in range(self.n_agents)]

        if np.where(self.S == self.final_states, False, True).sum() > 0:
            done = False
            non_terminated_agents = np.where(self.S != self.final_states, True, False)
            first_agents = np.where(self.T == self.T[non_terminated_agents].min(), True,
                                    False) * non_terminated_agents
            uni = np.unique(self.S[first_agents])
            self.base_state = np.random.choice(uni)
            # print(uni, self.base_state, self.final_states[non_terminated_agents])
            self.agents_at_base_state = np.where(self.S == self.base_state, True, False) * first_agents
            # print(self.agents_at_base_state)
            # print(self.S)
        else:
            done = True
            self.base_state = None
            self.agents_at_base_state = None

        return next_state, self.base_state, self.agents_at_base_state, transitions, done


if __name__ == "__main__":
    import networkx as nx
