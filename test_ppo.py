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

def test_train_pong_long(n_episodes=1000, batchnorm=False, device=device, render=False, **kwargs):
    env = gym.make('PongDeterministic-v4')
    run_one_episode = partial(pong.run_one_episode, env, render=render)
    policy_network = pong.PolicyNetwork(batchnorm=batchnorm, device=device)
    _policy, _scores = ppo.train(run_one_episode, policy_network, n_episodes=n_episodes, **kwargs)

def test_train_exp1():
    test_train_pong_long()

def test_train_exp2():
    test_train_pong_long(gamma=0.99)

def test_train_exp3():
    test_train_pong_long(gamma=0.99, weight_decay=0)

def test_train_exp4():
    test_train_pong_long(gamma=0.99, weight_decay=1e-4)

def test_train_exp5():
    test_train_pong_long(gamma=0.99, weight_decay=0, alpha=1e-4)

def test_train_exp6():
    test_train_pong_long(n_episodes=10000, gamma=0.99, weight_decay=0, alpha=1e-5)
    # failed to learn
    # last 100 episode score mean after 9900 episodes: -20.23

def test_train_exp7():
    test_train_pong_long(gamma=0.99, weight_decay=1e-6, alpha=1e-5)

def test_train_exp8():
    test_train_pong_long(n_episodes=10000, gamma=0.99, weight_decay=1e-6, alpha=1e-5, batchnorm=True)
    # not effectively but learns something
    # last 100 episode score mean after 9900 episodes: 2.16
    # learning accelerated after 6900 episodes
