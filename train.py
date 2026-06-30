import os
import threading
import time
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from clone_hero_env import CloneHeroEnv

# Import new config
from config import (
    SAVE_EVERY_N_STEPS, SAVE_SYNCHRONOUS,
    CURRICULUM_ENABLED, CURRICULUM_STAGES,
    USE_VEC_FRAME_STACK, N_FRAME_STACK,
    REWARD_SCORE_SCALE, REWARD_GHOST_PENALTY, REWARD_FAIL_PENALTY,
    REWARD_SURVIVAL_BONUS, REWARD_HIT_BONUS, REWARD_COMBO_BONUS,
)

# ─────────────────────────────────────────────────────────────
# SAVE + PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────

class VerboseCheckpointCallback(BaseCallback):

    def __init__(self, save_every: int, save_path: str, name_prefix: str, synchronous: bool = True):
        super().__init__()
        self.save_every   = save_every
        self.save_path    = save_path
        self.name_prefix  = name_prefix
        self.last_save_at = 0
        self.synchronous  = synchronous

        # reward tracking
        self.episode_rewards  = []
        self.current_reward   = 0.0
        self.best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # accumulate reward
        self.current_reward += float(self.locals["rewards"][0])

        # episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            recent = self.episode_rewards[-10:]
            avg    = sum(recent) / len(recent)

            print(
                f"  [step {step:>8,}] "
                f"ep_reward: {self.current_reward:>8.1f}  "
                f"avg(10): {avg:>8.1f}  "
                f"episodes: {len(self.episode_rewards)}"
            )

            # flag improvement
            if avg > self.best_mean_reward:
                self.best_mean_reward = avg
                best_path = os.path.join(self.save_path, f"{self.name_prefix}_BEST")
                self.model.save(best_path)
                print(f"  ★ new best avg reward {avg:.1f} — saved BEST model")

            self.current_reward = 0.0

        # checkpoint save
        if step - self.last_save_at >= self.save_every:
            filename = f"{self.name_prefix}_{step}_steps"
            filepath = os.path.join(self.save_path, filename)

            if self.synchronous:
                # Save in main thread - safer for pydirectinput
                self.model.save(filepath)
                print(f"\n{'─'*55}")
                print(f"  CHECKPOINT SAVED  step {step:,}")
                print(f"  → models/{filename}.zip")
                print(f"{'─'*55}\n")
            else:
                # Background thread (original behavior)
                threading.Thread(
                    target=self.model.save,
                    args=(filepath,),
                    daemon=True
                ).start()
                self.last_save_at = step
                print(f"\n{'─'*55}")
                print(f"  CHECKPOINT SAVED  step {step:,} (background)")
                print(f"  → models/{filename}.zip")
                print(f"{'─'*55}\n")

            self.last_save_at = step

        return True


# ─────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Adjust environment difficulty based on training progress."""

    def __init__(self, curriculum_stages, env):
        super().__init__()
        self.stages = curriculum_stages
        self.env = env
        self.current_stage = 0
        self.stage_start_step = 0

    def _on_step(self) -> bool:
        if not CURRICULUM_ENABLED:
            return True

        step = self.num_timesteps

        # Check if we should advance to next stage
        if self.current_stage < len(self.stages) - 1:
            current_stage_config = self.stages[self.current_stage]
            min_steps = current_stage_config.get("min_steps", 0)

            if step - self.stage_start_step >= min_steps:
                # Check if performance is good enough to advance
                # (could add performance check here)
                self.current_stage += 1
                self.stage_start_step = step
                new_stage = self.stages[self.current_stage]
                print(f"\n{'='*55}")
                print(f"  CURRICULUM: Advancing to stage '{new_stage['name']}'")
                print(f"  NPS range: {new_stage['min_nps']} - {new_stage['max_nps']}")
                print(f"{'='*55}\n")
                # Note: In practice, you'd need to change the song/chart here
                # This is a placeholder for curriculum logic

        return True


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def make_env(debug=False, curriculum_stage=0):
    """Factory function for creating env (needed for VecEnv)."""
    def _init():
        return Monitor(CloneHeroEnv(debug=debug, curriculum_stage=curriculum_stage))
    return _init


def train(resume_from: str = None):
    print("Please launch Clone Hero and navigate to the song.")
    for i in range(4, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # Create vectorized environment
    env = DummyVecEnv([make_env(debug=False, curriculum_stage=0)])

    # Apply VecFrameStack for proper frame stacking (replaces manual deque)
    if USE_VEC_FRAME_STACK:
        env = VecFrameStack(env, n_stack=N_FRAME_STACK, channels_order="last")
        print(f"Using VecFrameStack with {N_FRAME_STACK} frames")

    if resume_from:
        # ── resume from checkpoint ────────────────────────────
        print(f"\nResuming from {resume_from}")
        model = PPO.load(resume_from, env=env)
        print("Model loaded. Continuing training...\n")
    else:
        # ── fresh training run ────────────────────────────────
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,          # 0 = silence SB3's own table, our callback handles output
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,      # encourages exploration — important early on
            tensorboard_log="./logs/",
            # Better hyperparameters for this task
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
        )
        print("Fresh model created.\n")

    callback = VerboseCheckpointCallback(
        save_every  = SAVE_EVERY_N_STEPS,
        save_path   = "./models/",
        name_prefix = "ppo_clonehero",
        synchronous = SAVE_SYNCHRONOUS,
    )

    # Add curriculum callback if enabled
    callbacks = [callback]
    if CURRICULUM_ENABLED:
        callbacks.append(CurriculumCallback(CURRICULUM_STAGES, env))

    run_name = f"PPO_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Training started — saving every {SAVE_EVERY_N_STEPS:,} steps")
    print(f"Synchronous save: {SAVE_SYNCHRONOUS}")
    print(f"Tensorboard: tensorboard --logdir ./logs/\n")

    try:
        model.learn(
            total_timesteps  = 500_000,
            callback         = callbacks,
            tb_log_name      = run_name,
            log_interval     = 99999,   # silence SB3 table — our callback handles it
            reset_num_timesteps = not bool(resume_from)
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # save final
    final_step = callback.num_timesteps
    final_path = f"models/ppo_clonehero_FINAL_{final_step}_steps"
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")
    env.close()


if __name__ == "__main__":

    # Usage:
    #   python train.py                                      ← fresh run
    #   python train.py models/ppo_clonehero_4096_steps     ← resume
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_from=resume)