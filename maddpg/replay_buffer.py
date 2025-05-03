import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_dim, action_dim, num_agents, device="cpu"):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.ptr = 0
        self.size = 0

        # Store experiences in numpy arrays for efficiency
        self.states = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Stores a transition (s, a, r, s', done) in the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Move pointer and handle circular buffer
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """Samples a batch of experiences for training."""
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        batch_states = torch.tensor(self.states[indices], dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        batch_rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        batch_next_states = torch.tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def is_ready(self):
        """Checks if buffer has enough samples for training."""
        return self.size >= self.batch_size
