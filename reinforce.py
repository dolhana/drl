import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym


def train(env: gym.Env, policy_network, n_episodes: int, gamma: float, alpha: float =1e-3):
    policy = Policy(policy_network)
    optim = torch.optim.Adam(policy_network.parameters(), lr=alpha)

    scores = []
    for _i_episode in range(n_episodes):
        episode = run_episode(env, policy)
        rewards, observations, actions = zip(*episode)
        scores.append(np.sum(rewards[1:]))

        gammas = (gamma ** i for i in range(len(rewards) - 1))
        discounted_rewards = [g * r for g, r in zip(gammas, rewards[1:])]
        g = np.sum(discounted_rewards)

        actions = np.vstack(actions[:-1])

        observations = np.array(observations)
        log_probs = policy.log_prob(observations[:-1], actions)
        loss = - (log_probs * g).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()

        if _i_episode % 100 == 0:
            print(f'episode: {_i_episode}\taverage of last 100: {np.mean(scores[-100:])}')
        if np.mean(scores[-100:]) > 195.:
            break

    return policy, scores

def run_episode(env, policy):
    """Runs one episode and returns the trajectory.
    Returns:
      [(reward, observation, action)]
    """
    observation = env.reset()
    reward = None
    done = False

    episode = []
    while not done:
        action = policy.action(observation).detach().cpu().numpy()
        episode.append((reward, observation, action))
        observation, reward, done, _ = env.step(action)
    episode.append((reward, observation, None))

    return episode


class Policy():
    def __init__(self, policy_network):
        self.policy_network = policy_network

    def action(self, state):
        return self.pd(state).sample()

    def log_prob(self, state, action):
        action = torch.as_tensor(action).squeeze(-1)
        return self.pd(state).log_prob(action)

    def pd(self, state):
        state = torch.as_tensor(state, dtype=torch.float)
        logits = self.policy_network(state)
        probs = F.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs)


class PolicyNetwork(nn.Module):
    def __init__(self, n_state_dims, n_action_dims, hidden_units=[16]):
        super(PolicyNetwork, self).__init__()
        self.n_state_dims = n_state_dims
        self.n_action_dims = n_action_dims

        hidden_layers = nn.ModuleList()
        input_units = self.n_state_dims
        for units in hidden_units:
            hidden_layers.append(nn.Linear(input_units, units))
            input_units = units
        self.hidden_layers = hidden_layers
        self.output_layer = nn.Linear(input_units, self.n_action_dims)

    def forward(self, state):
        x = state
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
