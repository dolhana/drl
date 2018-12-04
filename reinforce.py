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
        rewards, observations, actions = [np.vstack(x) for x in zip(*episode)]

        scores.append(np.sum(rewards))
        last_100_score_average = np.mean(scores[-100:])

        if _i_episode % 100 == 0:
            print(f'episode: {_i_episode}\tlast 100 score average: {last_100_score_average}')

        if last_100_score_average > 196.:
            print(f'SOLVED at episode {_i_episode}: last 100 score average: {last_100_score_average} > 195.')
            break

        gammas = gamma ** np.arange(len(rewards) - 1)[:, np.newaxis]
        discounted_rewards = gammas * rewards[1:]
        g = np.sum(discounted_rewards)

        actions = np.vstack(actions[:-1])

        observations = np.array(observations)
        log_probs = policy.log_prob(observations[:-1], actions)
        loss = - (log_probs * g).sum()
        optim.zero_grad()
        loss.backward()
        optim.step()

    return policy, scores
