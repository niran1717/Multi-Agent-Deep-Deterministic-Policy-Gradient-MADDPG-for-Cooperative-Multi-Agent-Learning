import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from maddpg.actor import Actor
from maddpg.critic import Critic


class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, num_agents, agent_index,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.95, tau=0.01, hidden_dim=256, device="cpu"):
        """
        Initializes a MADDPG agent with actor-critic networks.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_index = agent_index  # Unique index for each agent
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor and Critic Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(obs_dim, action_dim, num_agents, hidden_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim, num_agents, hidden_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Loss function
        self.criterion = nn.MSELoss()

    def select_action(self, obs, noise=0.05):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]

        # Apply exploration noise
        noise_array = np.random.randn(*action.shape).astype(np.float32) * noise
        action += noise_array

        # Rescale from [-1, 1] → [0, 1]
        action = 0.5 * (action + 1.0)
        return np.clip(action, 0, 1).astype(np.float32)

    def update(self, replay_buffer, batch_size, all_agents):
        if not replay_buffer.is_ready():
            return

        # Sample from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample()

        states = states.to(self.device)         # [batch, num_agents, obs_dim]
        actions = actions.to(self.device)       # [batch, num_agents, action_dim]
        rewards = rewards.to(self.device)       # [batch, num_agents]
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        batch_size = states.shape[0]

        # ----------------------- Critic Update ----------------------- #
        # Get target actions from all agents
        target_actions = []
        for i in range(self.num_agents):
            agent_obs = next_states[:, i, :]
            with torch.no_grad():
                agent_action = all_agents[i].actor_target(agent_obs)
            target_actions.append(agent_action)
        target_actions = torch.cat(target_actions, dim=-1)
        next_states_flat = next_states.view(batch_size, -1)
        
        num_target_samples = 4  # You can tune this value (2–5 usually works well)
        target_q_total = 0
        # Target Q
        # Average Q' over multiple samples of target actions
        for _ in range(num_target_samples):
            target_actions_sample = []
            for i in range(self.num_agents):
                obs = next_states[:, i, :]
                with torch.no_grad():
                    act = all_agents[i].actor_target(obs)
                target_actions_sample.append(act)
            target_actions_concat = torch.cat(target_actions_sample, dim=-1)

            with torch.no_grad():
                q = self.critic_target(next_states_flat, target_actions_concat)
                target_q_total += q

        # Final averaged target Q
        target_q_avg = target_q_total / num_target_samples
        y = rewards[:, self.agent_index].unsqueeze(1) + self.gamma * target_q_avg * (1 - dones[:, self.agent_index].unsqueeze(1))

        # Current Q
        actions_flat = actions.view(batch_size, -1)
        states_flat = states.view(batch_size, -1)
        predicted_q = self.critic(states_flat, actions_flat)

        critic_loss = self.criterion(predicted_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # ----------------------- Actor Update ------------------------ #
        # Recompute current agent's actions
        agent_obs = states[:, self.agent_index, :]
        agent_action = self.actor(agent_obs)

        # Use current agent's action and others' previous actions
        current_actions = []
        for i in range(self.num_agents):
            if i == self.agent_index:
                current_actions.append(agent_action)
            else:
                other_action = actions[:, i, :]
                current_actions.append(other_action)
        current_actions = torch.cat(current_actions, dim=-1)

        # Actor loss = maximize Q (or minimize -Q)
        actor_loss = -self.critic(states_flat, current_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- Soft Update ------------------------ #
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
    
    def soft_update(self, source_net, target_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

