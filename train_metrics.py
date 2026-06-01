import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class MetricsCallback(BaseCallback):
    """
    Tracks and logs every 10 completed episodes to WandB:
      - Success rate         (% episodes reaching goal)
      - Collision rate       (% episodes ending in collision)
      - Average episode reward
      - Average steps to goal (successful episodes only)

    WandB logging activates automatically whenever a wandb run is active
    (i.e. wandb.init() has been called before training starts).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_steps   = []
        self._successes       = []
        self._collisions      = []
        self._current_rewards = {}
        self._current_steps   = {}

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals["infos"]):
            if i not in self._current_rewards:
                self._current_rewards[i] = 0.0
                self._current_steps[i]   = 0

            self._current_rewards[i] += self.locals["rewards"][i]
            self._current_steps[i]   += 1

            if self.locals["dones"][i]:
                self._episode_rewards.append(self._current_rewards[i])
                self._episode_steps.append(self._current_steps[i])
                self._successes.append(float(info.get("reached_goal", False)))
                self._collisions.append(float(info.get("collision",   False)))
                self._current_rewards[i] = 0.0
                self._current_steps[i]   = 0

        # Log every 10 completed episodes
        if len(self._successes) >= 10:
            success_rate   = np.mean(self._successes)
            collision_rate = np.mean(self._collisions)
            avg_reward     = np.mean(self._episode_rewards)
            success_steps  = [s for s, g in
                               zip(self._episode_steps, self._successes) if g]
            avg_steps      = np.mean(success_steps) if success_steps else 0.0

            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "metrics/success_rate":      success_rate,
                    "metrics/collision_rate":    collision_rate,
                    "metrics/avg_reward":        avg_reward,
                    "metrics/avg_steps_to_goal": avg_steps,
                }, step=self.num_timesteps)

            if self.verbose:
                print(f"\n[Metrics @ {self.num_timesteps} steps]")
                print(f"  Success rate:    {success_rate*100:.1f}%")
                print(f"  Collision rate:  {collision_rate*100:.1f}%")
                print(f"  Avg reward:      {avg_reward:.2f}")
                print(f"  Avg steps/goal:  {avg_steps:.0f}")

            self._episode_rewards = []
            self._episode_steps   = []
            self._successes       = []
            self._collisions      = []

        return True
