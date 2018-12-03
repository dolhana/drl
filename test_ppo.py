import numpy as np
import gym
import ppo
import util

def test_train_simple():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[])
    _policy, _scores = ppo.train(env, policy_network, n_episodes=1)

def test_train_3000():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    policy_network = util.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[16])
    _policy, scores = ppo.train(env, policy_network, n_episodes=3000, gamma=1., alpha=1e-3)
    assert np.mean(scores[-100:]) > 195.
