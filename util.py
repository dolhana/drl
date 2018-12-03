import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

def run_episode(env: gym.Env, policy, t_max: int = 1000):
    """Runs one episode and returns the trajectory.
    Returns:
      [(reward, observation, action)]
    """
    observation = env.reset()
    reward = 0

    episode = []
    for _ in range(t_max):
        action = policy.action(observation).detach().cpu().numpy()
        episode.append(([reward], observation, [action]))
        observation, reward, done, _ = env.step(action)
        if done:
            break
    episode.append(([reward], observation, [0]))

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
        return torch.distributions.Categorical(logits=logits)


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
