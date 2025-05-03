# 🧠 Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for Cooperative Multi-Agent Learning

This repository contains a complete implementation of the MADDPG algorithm using PyTorch and PettingZoo. It includes training, evaluation, plotting, and trajectory logging for cooperative multi-agent environments.

## 📁 Folder Structure

```
maddpg_project/
│   main.py                  # Training entry point
│   requirements.txt         # All dependencies
│
├───maddpg/                  # Core MADDPG components
│   actor.py
│   agent.py
│   critic.py
│   replay_buffer.py
│   utils.py
│   __init__.py
│
├───others/                  # Stored logs and comparison rewards
│   *.npy (reward/loss values for DDPG, DQN, etc.)
│
├───plots/                   # Saved graphs
│   *.png (loss, reward, comparison, trajectories, etc.)
│
├───saved_models/            # Trained model checkpoints
│   *.pth
│
├───train_logs/              # SLURM log files from HPC runs
│   maddpg.err
│   maddpg.out
│
└───trajectory/              # Agent position logs
    ├───training/
    └───evaluation/
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # or use `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## 🏋️‍♂️ Training the Model

To train the MADDPG agents:

```bash
python main.py
```

> Training logs and model checkpoints will be saved in `train_logs/` and `saved_models/`.

## 📊 Plotting & Visualization

You can use the provided plots in the `plots/` folder or generate your own using:

```bash
python maddpg/utils.py
```

## 📦 Trajectory Data

The folder `trajectory/` contains `.npy` files that store per-agent positional data. These are used for plotting training and evaluation movement patterns.

## 📡 SLURM (HPC) Training

If using a cluster like TACC with SLURM:

```bash
sbatch train_logs/maddpg.slurm
```

## 🔗 GitHub Info

**Repository Name:** `maddpg_project`  
**Author:** `Your Name`  
**License:** MIT  
**Status:** 📈 Actively maintained

## ✨ Credits

This project was developed for multi-agent reinforcement learning research and experimentation. Built using PyTorch, NumPy, and PettingZoo.
