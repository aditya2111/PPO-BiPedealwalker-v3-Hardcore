Here is a clean and comprehensive `README.md` / project description file for your **Proximal Policy Optimization (PPO) implementation on BipedalWalker-v3 (hardcore)** using PyTorch and Gymnasium:

---

# 🦿 PPO-BipedalWalker-v3 (Hardcore) — PyTorch Implementation

This repository contains a full implementation of **Proximal Policy Optimization (PPO)** applied to the **BipedalWalker-v3 (Hardcore)** continuous control task from the [Gymnasium](https://gymnasium.farama.org/) environment suite. It includes training, logging, model checkpointing, and reward visualization.

---

## 🚀 Environment

* **Environment**: `BipedalWalker-v3` (with `hardcore=True`)
* **Observation space**: Continuous (24-dimensional)
* **Action space**: Continuous (4-dimensional)
* **Reward goal**: Learn to walk across rough terrain using legged locomotion.

---

## 📦 Requirements

Before running, install the required packages:

```bash
pip install swig
pip install "gymnasium[box2d]"
pip install torch matplotlib pandas
```

---

## 🧠 Algorithm: Proximal Policy Optimization (PPO)

This implementation uses:

* **Actor-Critic architecture** with:

  * A diagonal multivariate Gaussian policy for continuous actions.
  * Separate fully-connected networks for actor and critic.
* **Generalized Advantage Estimation (GAE)** for stable advantage estimation.
* **Entropy bonus** to encourage exploration (annealed over time).
* **Clipped Surrogate Objective** to ensure conservative policy updates.
* **Mini-batch SGD** with shuffled batches from a replay buffer.
* **Action standard deviation decay** for reducing exploration over time.

---

## 🧱 Code Structure

### 1. `ActorCritic` class

* Shared architecture for policy (actor) and value function (critic).
* Outputs a mean action vector and uses a learnable `log_std` for exploration.
* Uses `MultivariateNormal` distribution for sampling and log-probability computation.

### 2. `PPO` class

* Manages policy updates, training loop, GAE advantage calculation, and logging.
* Handles:

  * Clipping of policy updates
  * Entropy bonus scheduling
  * Action standard deviation decay
  * Gradient clipping

### 3. `RolloutBuffer` class

* Temporarily stores state transitions for training.
* Stores:

  * States, actions, rewards, log\_probs, values, and terminal flags.

### 4. Training loop

* Runs until `10M` timesteps.
* Triggers policy updates every `8192` samples.
* Logs average reward and saves checkpoints periodically.
* Decays exploration noise after `500k` steps.

---

## 📈 Logging & Visualization

* Rewards are logged to a `.csv` file.
* Training progress can be visualized using matplotlib:

  * Raw reward plot
  * Smoothed curve (rolling average)

Example output:
![Training Curve](https://i.imgur.com/example_placeholder.png)

---

## 🧪 Hyperparameters

| Hyperparameter        | Value          |
| --------------------- | -------------- |
| Actor LR              | `2.5e-4`       |
| Critic LR             | `5e-4`         |
| Discount factor (γ)   | `0.99`         |
| GAE lambda            | `0.95`         |
| PPO Clip (ε)          | `0.2`          |
| K Epochs              | `10`           |
| Action Std (start)    | `0.6`          |
| Min Action Std        | `0.1`          |
| Batch size            | `8192`         |
| Update frequency      | `2049`         |
| Max episode length    | `600`          |
| Action Std decay rate | `0.0001`       |
| Entropy decay steps   | `3e6`          |
| Save model every      | `20,000` steps |

---

## 💾 Checkpoints

Model checkpoints are saved in:

```
PPO_preTrained/BipedalWalker-v3/PPO_BipedalWalker-v3_<seed>.pth
```

Use `ppo_agent.load(path)` to reload models for evaluation.

---

## 📊 Example Logs & Plotting

CSV logs:

```
episode, timestep, reward
...
```

Plot training reward with:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("path/to/log.csv")
df['smoothed'] = df['reward'].rolling(window=10).mean()

plt.plot(df['timestpe'], df['reward'], label='Reward')
plt.plot(df['timestpe'], df['smoothed'], '--', label='Smoothed')
plt.legend()
plt.grid()
plt.title('Training Reward - PPO on BipedalWalker-v3')
plt.show()
```

---

## 🔑 Key Takeaways

* Suitable as a template for training continuous control agents with PPO.
* Works with environments having continuous, high-dimensional state-action spaces.
* Self-contained and modular implementation—easy to extend or modify.

---

## 📚 References

* Schulman et al. **"Proximal Policy Optimization Algorithms"**, 2017. [\[paper\]](https://arxiv.org/abs/1707.06347)
* OpenAI Gym / Gymnasium Environments
* PyTorch Documentation

---

## 🛠 Future Improvements

* Evaluation and rendering support
* Parallelized rollout collection
* TensorBoard integration
* Support for multi-environment training (e.g. `gym.vector`)

---
