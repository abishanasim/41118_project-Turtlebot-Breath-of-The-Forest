Legend of TurtleBot: Breath of the Forest
A Python-based TurtleBot3 maze navigation project using PyBullet, Gymnasium, and Stable-Baselines3 PPO. The project trains a reinforcement learning agent to navigate a 5x5 forest maze using simulated LiDAR readings, goal-detection feedback, and visited-cell memory. A Tkinter launcher is included for training, testing, and opening TensorBoard.

# 1. Project Overview
This project simulates a TurtleBot3 Burger robot inside a generated maze environment. The robot must travel from the starting cell to the goal cell while avoiding walls and learning efficient navigation behaviour.

The system includes:
A custom Gymnasium environment for TurtleBot3 maze navigation.
PyBullet physics simulation and TurtleBot3 URDF/mesh assets.
PPO reinforcement learning through Stable-Baselines3.
LiDAR-style wall sensing using ray casts.
Goal-detection feedback using rendered camera images.
Occupancy/visited-cell tracking to encourage exploration.
A Tkinter GUI launcher for training, testing, and TensorBoard.
TensorBoard logging for reward and training metrics.

# 2. Main Features
5x5 maze generation using seeded maze layouts.
TurtleBot3 Burger simulation using PyBullet.
PPO training with Stable-Baselines3.
Continuous wheel velocity control for robot movement.
Observation vector containing:
24 LiDAR ray distances.
CNN-style goal visibility flag.
Goal confidence value.
Goal horizontal offset.
25 visited-cell indicators.
Reward shaping based on:
Goal reaching.
Collision penalty.
Step penalty.
Progress toward goal.
New cell exploration reward.
Revisit penalty.
Wall proximity penalty.
Goal visibility and centering reward.
GUI-based testing over three different maze scenarios.
TensorBoard support for viewing training graphs.

# 3. Project Structure
```text
project/
├── RUN_ME.py                       # Main Tkinter launcher for training, testing, and TensorBoard
├── train.py                        # PPO training script
├── test.py                         # Runs the trained model through test maze scenarios
├── turtlebot3_maze_env.py          # Custom Gymnasium + PyBullet TurtleBot3 maze environment
├── maze_grid.py                    # 5x5 maze generation and grid utilities
├── goal_detector.py                # Goal detection helper using image processing
├── train_metrics.py                # Custom Stable-Baselines3 callback for training metrics
├── turtlebot3_burger.urdf          # TurtleBot3 Burger robot model
├── turtlebot3_description/         # TurtleBot3 mesh assets
├── assets/                         # GUI image and audio assets
│   ├── splash_background.png
│   ├── home_background.png
│   ├── navigator_background.png
│   ├── title_text.png
│   ├── dirt.jpg
│   └── Zelda Main Theme Song.mp3
├── model/                          # Saved trained PPO models
├── maze_tensorboard/               # TensorBoard training logs
├── wandb/                          # Weights & Biases run logs, if enabled
├── portfolio/                      # Web portfolio for the project
│   ├── index.html
│   ├── logo.png
│   └── Graphs.png
└── README.md                       # Project documentation
```
# 4. Requirements
Recommended setup:
Windows 10/11, macOS, or Linux
Python 3.10 to 3.12 recommended
`pip`
A virtual environment
Python packages used by the project:
```text
gymnasium
pybullet
stable-baselines3[extra]
numpy
matplotlib
pillow
tensorboard
wandb
```
`wandb` is optional. The training script will still run without it.

# 5. Setup Instructions
## 5.1 Create a virtual environment
From inside the project folder:
```powershell
python -m venv .venv
```

## 5.2 Activate the virtual environment on Windows PowerShell
If PowerShell blocks activation, first run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate the environment:
```powershell
.\.venv\Scripts\activate
```
After activation, the terminal should look like this:
```text
(.venv) PS C:\Users\User\Documents\ai\project>
```

## 5.3 Install dependencies
Install the required packages into the active virtual environment:
```powershell
python -m pip install --upgrade pip
python -m pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib pillow tensorboard wandb
```
If you do not want Weights & Biases logging, you can omit `wandb`:
```powershell
python -m pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib pillow tensorboard
```

# 6. Running the Project
##6.1 Start the main launcher
```powershell
python maze.py
```
This opens the main GUI titled:
```text
Legend of Turtlebot Breath of the Forest
```
From the launcher, you can:
Start training.
Run testing.
Open TensorBoard.
Stop running processes.

## 6.2 Train the PPO model directly
```powershell
python maze_train.py
```
The training script uses:
```text
TOTAL_TIMESTEPS = 600000
N_ENVS = 4
MODEL_PATH = model/turtlebot3_maze_model_LIDAR
```
After training, the model is saved as:
```text
model/turtlebot3_maze_model_LIDAR.zip
```

## 6.3 Test the trained model directly
```powershell
python maze_test.py
```
The test script loads:
```text
model/turtlebot3_maze_model_LIDAR.zip
```
It then runs the trained robot through three seeded maze scenarios.

## 6.4 Open TensorBoard
```powershell
tensorboard --logdir maze_tensorboard --port 6006
```
Then open this address in a browser:
```text
http://localhost:6006
```

# 7. Training Configuration
The main reward settings are stored in `maze_train.py` inside `REWARD_CONFIG`.
Key values include:
```python
"goal_reward": 2000.0,
"collision_penalty": -200.0,
"step_penalty": -0.05,
"progress_scale": 20.0,
"new_cell_reward": 0.5,
"revisit_penalty": -0.05,
"cnn_visible_reward": 0.5,
"cnn_centering_reward": 1.5,
"front_wall_threshold": 0.50,
"front_wall_penalty": 3.0,
"proximity_threshold": 0.18,
"proximity_penalty": -0.5,
"crash_lidar_threshold": 0.01,
```
PPO hyperparameters include:
```python
learning_rate = 0.0003
n_steps = 1024
batch_size = 512
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
ent_coef = 0.005
clip_range = 0.2
```
# 8. Environment Details
The custom environment is implemented in:
```text
turtlebot3_maze_env.py
```
The robot starts in the bottom-left cell and must reach the top-right goal cell.
The observation size is 52 values:
```text
24 LiDAR values
+ 3 CNN/goal-detection values
+ 25 visited-cell indicators
= 52 total observations
```
The action space is continuous and controls the TurtleBot's wheel velocities.
# 9. Troubleshooting
Error: `No module named 'stable_baselines3'`
Activate the environment and reinstall:
```powershell
.\.venv\Scripts\activate
python -m pip install stable-baselines3[extra]
```
Then test:
```powershell
python -c "from stable_baselines3 import PPO; print('works')"
```

TensorBoard does not open
Make sure logs exist in:
```text
maze_tensorboard/
```
Then run:
```powershell
tensorboard --logdir maze_tensorboard --port 6006
```
Open:
```text
http://localhost:6006
```
Test model not found
The test script expects this file:
```text
model/turtlebot3_maze_model_LIDAR.zip
```
Train the model first:
```powershell
python maze_train.py
```
# 10. Suggested Run Order
For a fresh setup:
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib pillow tensorboard wandb
python -c "from stable_baselines3 import PPO; print('Stable-Baselines3 works')"
python maze.py
```
Then use the GUI to train, test, or open TensorBoard.

# 11. Authors
Created by:
Abisha Nasim and William Sklibosios

