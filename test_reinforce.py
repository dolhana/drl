from functools import partial

import gym
import numpy as np

import pong
import reinforce
import util


def test_train_simple():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[])
    _policy, _scores = reinforce.train(run_one_episode, policy_network, n_episodes=1, gamma=1.)

def test_train_2000():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    run_one_episode = partial(util.run_one_episode, env)
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[16])
    _policy, scores = reinforce.train(run_one_episode, policy_network, n_episodes=2000, gamma=1., alpha=1e-2)
    assert np.mean(scores[-100:]) > 195.


def test_train_pong_1_episode():
    env = gym.make('PongDeterministic-v4')
    run_one_episode = partial(pong.run_one_episode, env, render=True)
    policy_network = pong.PolicyNetwork()
    _policy, _scores = reinforce.train(run_one_episode, policy_network, n_episodes=1)

def test_train_pong_3k_episode(n_episodes=3000, render=False, **kwargs):
    env = gym.make('PongDeterministic-v4')
    run_one_episode = partial(pong.run_one_episode, env, render=render)
    policy_network = pong.PolicyNetwork()
    _policy, _scores = reinforce.train(run_one_episode, policy_network, n_episodes=n_episodes, **kwargs)
