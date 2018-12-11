import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import tqdm

import util

def train(run_one_episode: util.RunOneEpisodeFunc, policy_network: nn.Module, n_episodes=1, clip_epsilon=0.2, gamma=1., alpha=1e-3, weight_decay=1e-2, entropy_beta=1e-2):
    policy = util.Policy(policy_network)
    optim = torch.optim.Adam(policy_network.parameters(), lr=alpha, weight_decay=weight_decay)

    scores = []

    with tqdm.trange(n_episodes, ncols=100) as pbar:
        pbar_update_interval = max(2, n_episodes // 50)
        for _i_episode in pbar:
            episode = run_one_episode(policy)
            rewards, observations, actions = [np.vstack(x) for x in zip(*episode)]

            scores.append(np.sum(rewards))
            last_100_score_average = np.mean(scores[-100:])

            # if last_pbar_update is None or (_i_episode - last_pbar_update) >= pbar_update_interval:
            #     last_pbar_update = _i_episode
            pbar.set_postfix({'avg score': np.mean(scores[-pbar_update_interval:])}, refresh=False)

            if _i_episode % 100 == 0:
                print(f'episode: {_i_episode}\tlast 100 score average: {last_100_score_average}')

            if last_100_score_average > 196.:
                print(f'SOLVED at episode {_i_episode}: last 100 score average: {last_100_score_average} > 195.')
                break

            gammas = gamma ** np.arange(len(episode) - 1)[:, np.newaxis]

            # G for each time-step sums only the future rewards for credit assignment
            gs = (gammas * rewards[1:])[::-1].cumsum(axis=0)[::-1]
            gs = torch.as_tensor(gs.copy(), dtype=torch.float)

            # reward normalization
            gs_mean = gs.mean()
            gs_std = gs.std() + 1e-10
            gs_normalized = (gs - gs_mean) / gs_std

            actions = torch.as_tensor(actions)
            old_probs = policy.pd(observations[:-1]).probs.gather(dim=1, index=actions[:-1]).detach()

            # PPO reuses the sample trajectory multiple times
            for _ in range(3):
                new_probs = policy.pd(observations[:-1]).probs.gather(dim=1, index=actions[:-1])
                prop_ratio = new_probs / old_probs
                clipped_prop_ratio = prop_ratio.clamp(1. - clip_epsilon, 1. + clip_epsilon)

                # In 'CartPole-v0' env,
                #   sum() worked better without reward normalization
                #   mean() worked better with reward normalization
                surrogate = torch.min(
                    prop_ratio * gs_normalized,
                    clipped_prop_ratio * gs_normalized)

                # add a regularization term which steers new policy towards 0.5
                entropy = -(new_probs * torch.log(old_probs + 1e-10) +
                            (1. - new_probs) * torch.log(1. - old_probs + 1e-10))

                loss = - torch.mean(surrogate + entropy_beta * entropy)
                optim.zero_grad()
                loss.backward()
                optim.step()

    return policy, scores
