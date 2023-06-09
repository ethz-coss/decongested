import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math
import numpy as np


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # if self.method == "random":
        return random.sample(self.memory, batch_size)
        # elif self.method == "sequential":
        #     return [self.memory.pop() for ]

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden1=32, hidden2=16):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #         print("forward", x.shape)
        #         x = x.T
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class model:
    def __init__(self, n_observations, n_actions, device, LR, TAU, gamma, BATCH_SIZE=128, max_memory=1000, hidden1=32, hidden2=16):
        self.policy_net = DQN(n_observations, n_actions, hidden1=hidden1, hidden2=hidden2).to(device)
        self.target_net = DQN(n_observations, n_actions, hidden1=hidden1, hidden2=hidden2).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(max_memory)
        self.steps_done = 0
        self.device = device
        self.LR = LR
        self.gamma = gamma
        self.TAU = TAU
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.batch_size = BATCH_SIZE
        self.eps_threshold = 1
        self.loss = 0

    def judge_state(self, state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # y = self.policy_net(state)
            # x = y.max(0)[1]
            # w = y.max(-1)[1]
            # z = x.view(1, 1)
            return self.policy_net(state)

    def select_action(self, state, EPS_END=0.01, EPS_START=0.9,
                      EPS_DECAY=1000, method="random", neighbour_beliefs=None):  # epsilon greedy action selection, perfect
        sample = random.random()
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)  # epsilon is decayed manually
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # y = self.policy_net(state)
                # x = y.max(0)[1]
                # w = y.max(-1)[1]
                # z = x.view(1, 1)
                return self.policy_net(state).max(-1)[1].view(1, 1)  # changed max(1) to max(0) fixing bug
        else:
            if method == "random":
                return torch.tensor([[np.random.randint(self.n_actions)]], device=self.device, dtype=torch.long)
            elif method == "neighbours":
                return torch.tensor([[np.argmax(neighbour_beliefs)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            # print("here")
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if len(next_state_batch) > 0:
            with torch.no_grad():
                x = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = x.max(1)[0]
                y = x.max(1)[1]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))  # .unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad(
            set_to_none=True)  # set to none option has slightly different behaviour than setting to 0
        self.loss.backward()

        x = [p for p in self.policy_net.parameters()]

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[
                key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)
