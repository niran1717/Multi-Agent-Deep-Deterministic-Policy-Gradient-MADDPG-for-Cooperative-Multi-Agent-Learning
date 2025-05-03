import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        """
        Actor Network for MADDPG.
        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(Actor, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights for better training stability
        self.init_weights()

    def init_weights(self):
        """ Initialize weights of the network for stable training. """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)

    def forward(self, obs):
        """
        Forward pass of the actor network.
        Args:
            obs (Tensor): The input observation.
        Returns:
            Tensor: Action values bounded between [-1, 1] using Tanh.
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Tanh bounds actions between -1 and 1
        return action
