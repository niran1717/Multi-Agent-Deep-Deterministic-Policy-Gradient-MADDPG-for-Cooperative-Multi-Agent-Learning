import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=256):
        """
        Critic Network for MADDPG.
        Args:
            obs_dim (int): Dimension of the observation space for each agent.
            action_dim (int): Dimension of the action space for each agent.
            num_agents (int): Number of agents in the environment.
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(Critic, self).__init__()

        # Since this is a centralized critic, input is (all states + all actions)
        input_dim = (obs_dim + action_dim) * num_agents

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single output Q-value

        # Initialize weights for better training stability
        self.init_weights()

    def init_weights(self):
        """ Initialize weights of the network for stable training. """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, states, actions):
        """
        Forward pass of the critic network.
        Args:
            states (Tensor): Tensor containing observations of all agents.
            actions (Tensor): Tensor containing actions of all agents.
        Returns:
            Tensor: Estimated Q-value.
        """
        x = torch.cat([states, actions], dim=-1)  # Concatenate states and actions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)  # Q-value output
        return q_value
