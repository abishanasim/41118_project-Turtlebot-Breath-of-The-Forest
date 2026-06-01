import os 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv 
from turtlebot3_maze_env import TurtleBot3MazeEnv
from train_metrics import MetricsCallback 

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

TOTAL_TIMESTEPS = 1000000
N_ENVS          = 10
MODEL_PATH      = "model/turtlebot3_maze_model"

# ========================================================
# Reward / Penalty Configuration
# Centralised here so tuning only requires changes in one place
# ========================================================
REWARD_CONFIG = {

    "goal_reward":           2000.0,
    "collision_penalty":     -200.0,
    "step_penalty":           -0.05,
    "progress_scale":          20.0,

    # Occupancy / visited-grid exploration signals
    "new_cell_reward":         0.5,    # one-time bonus — small so goal dominates
    "revisit_penalty":        -0.05,   # grows with visit count (see env for scaling)

    # CNN visual guidance (wall-aware — centering reward disabled when front blocked)
    "cnn_visible_reward":      0.5,    # small bonus whenever goal is visible
    "cnn_centering_reward":    1.5,    # bonus when goal centred AND path ahead clear

    # Front-wall early-warning penalty (fraction units × 3 m = actual metres)
    "front_wall_threshold":    0.50,   # 0.50 frac = 1.5 m
    "front_wall_penalty":      4.0,    # was 3.0

    # Near-wall exponential proximity penalty
    "proximity_threshold":     0.22,   # was 0.18 (0.54 m) → now 0.66 m
    "proximity_penalty":      -0.8,    # was -0.5
    "proximity_exponent":      3.0,

    # Emergency fail distance
    "crash_lidar_threshold":   0.01,
}

if __name__ == "__main__":
    # ── WandB initialisation (no-op if wandb not installed) ───────────────
    if _WANDB_AVAILABLE:
        wandb.init(
            project="turtlebot3-maze",
            name="ppo-lidar-cnn-occ",
            config={
                **REWARD_CONFIG,
                "total_timesteps": TOTAL_TIMESTEPS,
                "n_envs":          N_ENVS,
                "algorithm":       "PPO",
                "policy":          "MlpPolicy",
                "obs_size":        52,
                "learning_rate":   0.0003,
                "n_steps":         1024,
                "batch_size":      512,
                "n_epochs":        10,
                "gamma":           0.99,
                "gae_lambda":      0.95,
                "ent_coef":        0.005,
                "clip_range":      0.2,
                "randomize_maze":  True,
            },
        )

    env = make_vec_env(
        TurtleBot3MazeEnv,
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "renders": False,
            "reward_config": REWARD_CONFIG,
            "randomize_maze": True,
        },
    )

    if os.path.exists(MODEL_PATH + ".zip"):
        print("Loading existing model for continued training...")
        model = PPO.load(MODEL_PATH, env=env, device='cuda')
    else:
        print("Training from scratch...")
        model = PPO(
            "MlpPolicy", env,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            clip_range=0.2,
            verbose=1,
            device='cuda',
        )

    callback = MetricsCallback(verbose=1)

    print(f"Training for {TOTAL_TIMESTEPS} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}.zip")
    env.close()

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()