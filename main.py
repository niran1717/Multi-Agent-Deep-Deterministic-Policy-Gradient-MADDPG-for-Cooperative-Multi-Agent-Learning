import os
import numpy as np
import torch
import random
from pettingzoo.mpe import simple_spread_v3
from maddpg.agent import MADDPGAgent
from maddpg.replay_buffer import ReplayBuffer

# Logging and saving directories
os.makedirs("saved_models_1", exist_ok=True)
os.makedirs("logs_1", exist_ok=True)
reward_history = []

# Hyperparameters
EPISODES = 50000
MAX_STEPS = 100   # Increased from 25
BATCH_SIZE = 128
BUFFER_SIZE = 1_000_000
GAMMA = 0.95
TAU = 0.01
ACTOR_LR = 5e-3
CRITIC_LR = 5e-3
SAVE_INTERVAL = 1000
t = 0 #GLOBAL TIMESTEP

# Noise scheduling
initial_noise = 0.5
final_noise = 0.01
noise_decay = 0.9999

# Seeding
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Initialize environment
env = simple_spread_v3.env(continuous_actions=True, render_mode=None)
env.reset()
num_agents = len(env.agents)

obs_dim = env.observation_space(env.agents[0]).shape[0]
action_dim = env.action_space(env.agents[0]).shape[0]

# Initialize agents
agents = [MADDPGAgent(obs_dim, action_dim, num_agents, agent_index=i,
                      actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                      gamma=GAMMA, tau=TAU, device="cpu")
          for i in range(num_agents)]

# Replay Buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, obs_dim, action_dim, num_agents)

# Logging reward history
reward_history = []

print("\n--- Starting MADDPG Training ---\n")

initial_noise = 0.2
final_noise = 0.01
noise_decay = 0.9995  # Decays slowly over time

for episode in range(EPISODES):
    env.reset()
    observations = {agent: env.observe(agent) for agent in env.agents}
    total_rewards = np.zeros(num_agents)

    # Compute decayed noise
    current_noise = max(final_noise, initial_noise * (noise_decay ** episode))

    for step in range(MAX_STEPS):
        actions = {}

        for i, agent in enumerate(env.agents):
            action = agents[i].select_action(observations[agent], noise=current_noise)
            actions[agent] = np.array(action, dtype=np.float32)

        for agent in env.agents:
            env.step(actions[agent])

        next_observations = {agent: env.observe(agent) for agent in env.agents}
        rewards = {agent: env.rewards[agent] for agent in env.agents}
        dones = {agent: env.terminations[agent] or env.truncations[agent] for agent in env.agents}

        replay_buffer.add(
            np.array([observations[agent] for agent in env.agents]),
            np.array([actions[agent] for agent in env.agents]),
            np.array([rewards[agent] for agent in env.agents]),
            np.array([next_observations[agent] for agent in env.agents]),
            np.array([dones[agent] for agent in env.agents])
        )

        total_rewards += np.array([rewards[agent] for agent in env.agents])
        observations = next_observations

        t += 1
        if replay_buffer.is_ready() and t % 100 == 0:
            for agent in agents:
                agent.update(replay_buffer, BATCH_SIZE, agents)  # <- pass all agents

        if all(dones.values()):
            break

    # Log rewards
    reward_history.append(total_rewards.tolist())  # Store as list for numpy saving
    if episode % 100 == 0:
        print(f"Episode {episode}: Average Reward per Agent: {np.mean(total_rewards):.3f}")

    if episode % SAVE_INTERVAL == 0 and episode > 0:
        for i, agent in enumerate(agents):
            torch.save(agent.actor.state_dict(), f"saved_models_1/actor_agent_{i}_ep{episode}.pth")
            torch.save(agent.critic.state_dict(), f"saved_models_1/critic_agent_{i}_ep{episode}.pth")
        print(f"\n--- Models Saved at Episode {episode} ---\n")

# Save rewards at end
np.save("rewards.npy", np.array(reward_history))
print("\n--- Training Completed ---\n")
