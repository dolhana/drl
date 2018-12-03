import numpy as np
import gym
import reinforce

def test_train_simple():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    policy_network = reinforce.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[])
    _policy, scores = reinforce.train(env, policy_network, 1, 1)

def test_train_2000():
    env = gym.make('CartPole-v0')
    n_state_dims = env.observation_space.shape[0]
    n_action_dims = env.action_space.n
    policy_network = reinforce.PolicyNetwork(n_state_dims, n_action_dims, hidden_units=[16])
    _policy, scores = reinforce.train(env, policy_network, 2000, 1, alpha=1e-2)
    assert np.mean(scores[-100:]) > 195.
