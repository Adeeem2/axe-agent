import os
import threading
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from clone_hero_env import CloneHeroEnv


# ─────────────────────────────────────────────────────────────
# SAVE + PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────

SAVE_EVERY_N_STEPS = 4096


class VerboseCheckpointCallback(BaseCallback):

    def __init__(self, save_every: int, save_path: str, name_prefix: str):
        super().__init__()
        self.save_every   = save_every
        self.save_path    = save_path
        self.name_prefix  = name_prefix
        self.last_save_at = 0

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

        # checkpoint save in background thread so env keeps running
        if step - self.last_save_at >= self.save_every:
            filename = f"{self.name_prefix}_{step}_steps"
            filepath = os.path.join(self.save_path, filename)
            threading.Thread(
                target=self.model.save,
                args=(filepath,),
                daemon=True
            ).start()
            self.last_save_at = step
            print(f"\n{'─'*55}")
            print(f"  CHECKPOINT SAVED  step {step:,}")
            print(f"  → models/{filename}.zip")
            print(f"{'─'*55}\n")

        return True


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def train(resume_from: str = None):
    print("Please launch Clone Hero and navigate to the song.")
    for i in range(4, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # Wrap in Monitor then DummyVecEnv — SB3 works best with VecEnv
    env = DummyVecEnv([lambda: Monitor(CloneHeroEnv(debug=False))])

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
            tensorboard_log="./logs/"
        )
        print("Fresh model created.\n")

    callback = VerboseCheckpointCallback(
        save_every  = SAVE_EVERY_N_STEPS,
        save_path   = "./models/",
        name_prefix = "ppo_clonehero"
    )

    run_name = f"PPO_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Training started — saving every {SAVE_EVERY_N_STEPS:,} steps")
    print(f"Tensorboard: tensorboard --logdir ./logs/\n")

    try:
        model.learn(
            total_timesteps  = 500_000,
            callback         = callback,
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
    import sys

    # Usage:
    #   python train.py                                      ← fresh run
    #   python train.py models/ppo_clonehero_4096_steps     ← resume
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_from=resume)