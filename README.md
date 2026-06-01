# Legend of Turtlebot: Breath of the Forest

A reinforcement learning project that trains a TurtleBot3 Burger robot to autonomously navigate procedurally-generated 5×5 grid mazes using Proximal Policy Optimization (PPO). The robot learns to travel from a fixed start position to a goal using a 52-dimensional observation space combining simulated LiDAR sensor readings, CNN-based visual goal detection, and a visited-cell exploration map — all inside a PyBullet physics simulation. A Zelda-themed GUI provides a live dashboard with robot POV camera, bird's-eye path trace, 360° LiDAR visualisation, and real-time training statistics.

---

## Group Information

| | |
|---|---|
| **Group Number** | 10 |
| **Group Composition** | Abisha Nasim, William Sklibosios |

---

## Project Structure

```
project_final/
├── RUN_ME.py                  # Main GUI launcher (recommended entry point)
├── train.py                   # PPO training script (1M timesteps, 10 parallel envs)
├── test.py                    # Evaluation script (3 random maze scenarios)
├── turtlebot3_maze_env.py     # Custom Gymnasium environment (physics, rewards, sensors)
├── maze_grid.py               # Procedural 5×5 maze generator (recursive backtracking)
├── goal_detector.py           # CNN-style visual goal detection from robot POV camera
├── train_metrics.py           # WandB / TensorBoard metrics callback
├── turtlebot3_burger.urdf     # TurtleBot3 robot model definition
├── turtlebot3_description/    # 3D mesh assets for robot rendering
├── assets/                    # GUI backgrounds, title image, background music
└── model/                     # Saved trained model (turtlebot3_maze_model.zip)
```

---

## Install Info

**Requirements:** Python 3.10+

**Option A — CPU only (Mac / Linux / Windows):**
```bash
pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib Pillow pygame wandb
```

**Option B — GPU-accelerated training (NVIDIA only):**
```bash
# Install CUDA-enabled PyTorch first (adjust cu118 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib Pillow pygame wandb
```

> `pygame` is optional — it enables the Zelda background music in the GUI. `wandb` is strongly recommended for tracking training metrics; without it, metrics are only printed to the console.

---

## Run Commands

### Recommended — Launch the full GUI

```bash
python RUN_ME.py
```

Opens a Zelda-themed splash screen followed by a home dashboard. From there you can train the model, test it, or open your WandB experiment logs — all from a live visual interface showing the robot POV, bird's-eye path trace, LiDAR chart, and episode statistics.

---

### Train the model from the command line

```bash
python train.py
```

Trains a PPO agent for **1,000,000 timesteps** across **10 parallel environments**. Each episode spawns a new randomly-generated maze. The trained model is saved to `model/turtlebot3_maze_model.zip`. Training metrics (success rate, collision rate, average reward, average steps to goal) are logged every 10 episodes to TensorBoard and, if configured, to Weights & Biases.

---

### Test the trained model

```bash
python test.py
```

Loads the saved model and runs it on **3 random maze scenarios** with fixed seeds. A live NavigatorGUI window displays the robot navigating each maze in real time. Per-scenario results (total reward, steps taken, distance to goal, outcome) are printed to the console.

---

## How It Works

| Component | Detail |
|---|---|
| **Algorithm** | PPO (Proximal Policy Optimization) via Stable-Baselines3 |
| **Policy** | MlpPolicy — 52-dim observation → continuous 2D wheel velocities |
| **Observation** | 24 LiDAR rays + 3 CNN goal-detection values + 25 visited-cell flags |
| **Action space** | Left/right wheel velocity ∈ [−1, 1], mapped to ±15 rad/s |
| **Maze** | New random 5×5 perfect maze each episode (recursive backtracking DFS) |
| **Simulator** | PyBullet with gravity, friction, and collision detection |
| **Reward** | +2000 goal reach (efficiency-scaled) · −200 collision · +0.5 new cell · step penalties for walls, spinning, reversing, revisiting |
