import numpy as np
import torch
from collections import defaultdict, namedtuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
            [np.array([0 if i != l else 1 for i in range(self.size)]), np.array([0 if i != r else 1 for i in range(self.size)])]) for
            (l, r) in self.G.nodes()}
        self.training_visitation_counts = defaultdict(self._return_0)
        # self.one_hot_enc = {
        #     (0, 0): np.array([0, 0, 0, 1]),
        #     (0, 1): np.array([0, 0, 1, 0]),
        #     (1, 0): np.array([0, 1, 0, 0]),
        #     (1, 1): np.array([1, 0, 0, 0]),
        # }

    def reset(self):
        self.step_counter = np.zeros(self.n_agents)
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros((self.n_agents, 2)).astype(int)  # two dim, for two dim states
        self.final_states = np.ones((self.n_agents, 2)) * np.array([self.size-1, self.size-1])  # np.random.randint(0, 7, size=(self.n_agents, 2))
        self.actions_taken = np.zeros((self.size, self.size, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        self.stored_transitions = {}
        # state = [self.one_hot_enc[tuple(self.S[n])] for n in range(self.n_agents)]
        remaining_agents = np.ones(self.n_agents, dtype=bool)
        first_agents = remaining_agents
        uni = np.unique(self.S[first_agents], axis=0)

        self.base_state = tuple(uni[np.random.randint(len(uni))])
        self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True, False) * first_agents
        self.agents_in_transit = defaultdict(self._return_0)
        self.last_agent_transits = defaultdict(self._return_none)
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0, zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))

        self.current_state = [self.partial_observation(agent) for agent in range(self.n_agents)]

        return self.current_state, info, self.base_state, self.agents_at_base_state

    def counterfactual_observation(self, agent):
        tuple_state = tuple(self.S[agent])
        node = self.one_hot_enc[tuple_state]
        pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])

        uniform_belief_counterfactual = np.ones(self.n_actions) / self.n_actions

        return np.concatenate([node, pct_occupied, uniform_belief_counterfactual])

    def partial_observation(self, agent):
        tuple_state = tuple(self.S[agent])
        node = self.one_hot_enc[tuple_state]
        pct_occupied = np.array([self.states_counts[tuple_state] / self.n_agents])
        edges = self.G.edges(tuple_state)

        if len(edges) == 4:
            agents_on_outgoing_edges = np.array([self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])
        else:
            edges = list(edges) + [self.base_state for i in range(4 - len(edges))]
            agents_on_outgoing_edges = np.array([self.agents_in_transit[edge[0], edge[1]] / self.n_agents for edge in edges])

        return np.concatenate([node, pct_occupied, agents_on_outgoing_edges])

    def _return_0(self):
        return 0

    def _return_none(self):
        return None

    def step(self, actions, drivers):
        # pre-process actions
        actions = actions.flatten()
        counts = np.bincount(actions, minlength=self.n_actions)
        self.actions_taken[self.base_state[0], self.base_state[1]] += counts

        # calculate rewards
        neighbour_nodes = [edge[1] for edge in self.G.edges(self.base_state)]
        reward_per_action = [self.G.adj[self.base_state][edge[1]]["cost"](
            counts[i]+self.agents_in_transit[edge]) for i, edge in enumerate(self.G.edges(self.base_state))]
        if len(neighbour_nodes) < self.n_actions:
            reward_per_action = np.concatenate([reward_per_action, 2*np.ones((self.n_actions-len(neighbour_nodes)))])
            for i in range(self.n_actions - len(neighbour_nodes)):
                neighbour_nodes.append(self.base_state)
        rewards = np.array([reward_per_action[a] for a in actions])

        # update current states and agents in transit
        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        self.states_counts = defaultdict(self._return_0, zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))
        edges_actions = {(self.base_state, neighbour): counts[a] if a < len(neighbour_nodes) else None for a, neighbour
                         in enumerate(neighbour_nodes)}
        self.trajectory.append(tuple([self.base_state, dict(self.states_counts), dict(edges_actions), rewards]))
        self.agents_in_transit.update(edges_actions)
        destinations = [neighbour_nodes[a] for a in actions]
        self.S[self.agents_at_base_state] = np.array(destinations).astype(int)
        self.T[self.agents_at_base_state] += rewards
        for a, node in enumerate(neighbour_nodes):
            self.training_visitation_counts[node] += counts[a]

        # calculate destination counterfactual beliefs
        # destination_states = np.unique(destinations)
        # destination_observations = np.concatenate([])
        #
        # counterfactual_states = [
        #     np.concatenate([self.current_state[agent][0:-4], np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])]) for agent in drivers.keys()
        # ]
        # counterfactual_beliefs = [
        #     driver.judge_state(
        #         state=torch.tensor(counterfactual_states[agent], dtype=torch.float32, device=device)
        #     ) for agent, driver in drivers.items()
        # ]
        # counterfactual_beliefs = torch.cat(counterfactual_beliefs)
        # counterfactual_beliefs = counterfactual_beliefs.cpu().numpy()
        # base_state_counterfactual = np.mean(counterfactual_beliefs[agents_at_base_state])
        # complete_states = [
        #     np.concatenate([self.current_state[agent][0:-4], base_state_counterfactual])
        #     for n, agent in drivers.items() if agents_at_base_state[n]
        # ]

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
            self.current_state[n] = self.counterfactual_observation(n)

        # next_state = [self.counterfactual_observation(agent) for agent in range(self.n_agents)]  # state vector for all agents

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

                # counterfactual_states = [
                #     np.concatenate([self.current_state[agent][0:-4], np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])])
                #     for agent in base_state_indices
                # ]

                counterfactual_beliefs = [
                    drivers[int(agent)].judge_state(
                        state=torch.tensor(self.current_state[int(agent)], dtype=torch.float32, device=device)
                    ) for agent in base_state_indices
                ]

                counterfactual_beliefs = torch.stack(counterfactual_beliefs)
                counterfactual_beliefs = counterfactual_beliefs.cpu().numpy()
                base_state_counterfactual = np.mean(counterfactual_beliefs, axis=0)
                complete_state = np.concatenate([self.current_state[int(base_state_indices[0])][0:-4], base_state_counterfactual])
                #
                # [
                #     np.concatenate([self.current_state[agent][0:-4], base_state_counterfactual])
                #     for n, agent in drivers.items() if agents_at_base_state[n]
                # ]

                for n in base_state_indices:
                    edge = self.last_agent_transits[int(n)]
                    self.agents_in_transit[edge] -= 1
                    state_ = torch.tensor(complete_state, dtype=torch.float32, device=device)
                    self.stored_transitions[int(n)]["next_state"] = state_
                    transitions.append((int(n), self.stored_transitions[int(n)]))
                    self.current_state[int(n)] = complete_state
                    self.step_counter[int(n)] += 1

                finished_indices = np.argwhere((self.agents_at_final_state * fastest_agents) == True)
                for n in finished_indices:
                    # print(n)
                    edge = self.last_agent_transits[int(n)]
                    self.agents_in_transit[edge] -= 1
                    transitions.append((int(n), self.stored_transitions[int(n)]))
                    self.current_state[int(n)] = None
                    self.step_counter[int(n)] += 1
        else:
            done = True
            self.base_state = None
            self.agents_at_base_state = None
            transitions = []

        self.step_counter += 1
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
