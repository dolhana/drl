from functools import partial

import gym
import numpy as np
import torch

import pong
import ppo
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_train_1_episode():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[])
    _policy, _scores = ppo.train(run_one_episode, policy_network, n_episodes=1)

def test_train_1_epsode_gpu():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[], device=torch.device('cuda'))
    _policy, _scores = ppo.train(run_one_episode, policy_network, n_episodes=1)

def test_train_3k():
    torch.manual_seed(0)
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[16])
    _policy, scores = ppo.train(run_one_episode, policy_network, n_episodes=3000, alpha=1e-3, gamma=1., entropy_beta=0.01, weight_decay=0)
    assert np.mean(scores[-100:]) > 195.

def test_train_3k_gpu():
    torch.manual_seed(0)
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[16], device=torch.device('cuda'))
    _policy, scores = ppo.train(run_one_episode, policy_network, n_episodes=3000, alpha=1e-3, gamma=1., entropy_beta=0.01, weight_decay=0)
    assert np.mean(scores[-100:]) > 195.


def test_train_pong_1_episode():
    env = gym.make('PongDeterministic-v4')
    run_one_episode = partial(pong.run_one_episode, env, render=True)
    policy_network = pong.PolicyNetwork()
    _policy, _scores = ppo.train(run_one_episode, policy_network, n_episodes=1)

def test_train_pong_long(n_episodes=1000, device=device, render=False, **kwargs):
    env = gym.make('PongDeterministic-v4')
    run_one_episode = partial(pong.run_one_episode, env, render=render)
    policy_network = pong.PolicyNetwork(device=device)
    _policy, _scores = ppo.train(run_one_episode, policy_network, n_episodes=n_episodes, **kwargs)

def test_train_exp1():
    test_train_pong_long()

def test_train_exp2():
    test_train_pong_long(gamma=0.99)

def test_train_exp3():
    test_train_pong_long(gamma=0.99, weight_decay=0)

def test_train_exp4():
    test_train_pong_long(gamma=0.99, weight_decay=1e-4)
