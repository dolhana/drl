import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym

import util


def train(env: gym.Env, policy_network, n_episodes: int, gamma: float, alpha: float =1e-3):
    policy = util.Policy(policy_network)
    optim = torch.optim.Adam(policy_network.parameters(), lr=alpha)

    scores = []
    for _i_episode in range(n_episodes):
        episode = util.run_episode(env, policy)
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
            print(f'episode: {_i_episode}\tlast 100 average: {np.mean(scores[-100:])}')
        if np.mean(scores[-100:]) > 195.:
            break

    return policy, scores
