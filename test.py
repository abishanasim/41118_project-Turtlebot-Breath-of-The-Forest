import time 
import random 
from stable_baselines3 import PPO
from turtlebot3_maze_env import TurtleBot3MazeEnv 
from RUN_ME import NavigatorGUI
from maze_grid import generate_5x5_maze

# To use fixed seeds instead, comment out the random lines and
# uncomment the lines below:
# SEED_1, SEED_2, SEED_3 = 8, 9, 10

SEED_1 = random.randint(0, 99)
SEED_2 = random.randint(0, 99)
SEED_3 = random.randint(0, 99)

print(f"Test seeds: {SEED_1}, {SEED_2}, {SEED_3}")

SCENARIOS = [
    {
        "name":  f"MAZE 1 — Seed {SEED_1}",
        "walls": generate_5x5_maze(seed=SEED_1),
    },
    {
        "name":  f"MAZE 2 — Seed {SEED_2}",
        "walls": generate_5x5_maze(seed=SEED_2),
    },
    {
        "name":  f"MAZE 3 — Seed {SEED_3}",
        "walls": generate_5x5_maze(seed=SEED_3),
    },
]

MODEL_PATH = "model/turtlebot3_maze_model"


def test():
    env  = TurtleBot3MazeEnv(renders=False, walls=SCENARIOS[0]["walls"])
    model = PPO.load(MODEL_PATH, env=env)
    gui   = NavigatorGUI(env)

    for ep, scenario in enumerate(SCENARIOS):
        is_last = ep == len(SCENARIOS) - 1
        print(f"\n--- Episode {ep + 1}: {scenario['name']} ---")

        env.close()
        env   = TurtleBot3MazeEnv(renders=False, walls=scenario["walls"])
        model = PPO.load(MODEL_PATH, env=env)
        gui.env = env

        gui.new_episode(ep + 1, scenario["name"])

        obs, _       = env.reset()
        done         = False
        total_reward = 0.0
        status       = "RUNNING"

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if info["reached_goal"]:
                status = "REACHED GOAL"
            elif info["collision"]:
                status = "COLLISION"
            elif truncated:
                status = "TIMEOUT"

            lidar = obs[:24]
            gui.update(
                steps      = env._step_count,
                reward     = total_reward,
                dist       = info["dist"],
                status     = status,
                lidar      = lidar,
                visit_map  = env.get_visit_map(),
            )
            time.sleep(1 / 60)

        print(f"Result: {status} | Total Reward: {total_reward:.2f}")

        gui.show_result(
            status        = status,
            reward        = total_reward,
            steps         = env._step_count,
            dist          = info["dist"],
            scenario_name = scenario["name"],
            is_last       = is_last,
        )
        gui.wait_for_next()

    time.sleep(1)
    gui.close()
    env.close()


if __name__ == "__main__":
    test()
