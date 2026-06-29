import sys
import os
import re
import time
from clone_hero_env import CloneHeroEnv
from train import train


def get_latest_checkpoint(models_dir: str = "./models") -> str | None:
    """
    Scans models/ for checkpoint files named ppo_clonehero_FINAL_<N>_steps.zip
    and returns the path of the one with the highest step count.
    Returns None if no checkpoints exist yet.
    """
    if not os.path.isdir(models_dir):
        return None

    pattern = re.compile(r"ppo_clonehero_FINAL_(\d+)_steps\.zip$")
    best_path  = None
    best_steps = -1

    for fname in os.listdir(models_dir):
        match = pattern.match(fname)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path  = os.path.join(models_dir, fname)

    return best_path


def test_env():
    print("=" * 55)
    print("  ENV TEST — random actions for 20 steps")
    print("  Switch to Clone Hero NOW")
    print("=" * 55)

    for i in range(5, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    env = CloneHeroEnv(debug=True)
    obs, info = env.reset()

    hit_count  = 0
    miss_count = 0

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        action_name = ["Green", "Red", "Yellow", "Blue", "Orange", "Nothing"][action]

        if reward > 0:
            hit_count += 1
            tag = "HIT  ✓"
        elif reward < -0.01:
            miss_count += 1
            tag = "MISS ✗"
        else:
            tag = "     ·"

        print(f"  Step {i:>2} | {tag} | Action: {action_name:<7} | Reward: {reward:>6.2f}")

        if terminated:
            print("  Episode ended early (song failed)")
            break

        time.sleep(0.05)

    env.close()

    print()
    print(f"  Hits:   {hit_count}")
    print(f"  Misses: {miss_count}")
    print()
    print("  Check debug_obs.png   → what the agent sees (grayscale 84x84)")
    print("  Check debug_color.png → raw color capture of the highway")
    print()
    if hit_count == 0:
        print("  ⚠  No hits detected — your LANE_COLORS_BGR or STRIKEBAR_Y")
        print("     in clone_hero_env.py probably needs calibration.")
        print("     Run: python calibrate.py")
    else:
        print("  ✓ Env is working. Ready to train.")


def play(model_path: str):
    from stable_baselines3 import PPO

    print(f"Loading model: {model_path}")
    env = CloneHeroEnv(debug=False)
    model = PPO.load(model_path, env=env)

    print("Switch to Clone Hero NOW — playing in 5 seconds...")
    time.sleep(5)

    obs, _ = env.reset()
    episode      = 0
    total_reward = 0.0

    print("Playing. Press Ctrl+C to stop.\n")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated:
                episode += 1
                print(f"  Episode {episode} finished | Total reward: {total_reward:.1f}")
                total_reward = 0.0
                time.sleep(1)
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "test":
            test_env()

        elif cmd == "play":
            if len(sys.argv) < 3:
                # no path given — try BEST model, then latest checkpoint
                best = "./models/ppo_clonehero_BEST.zip"
                if os.path.exists(best):
                    play(best)
                else:
                    latest = get_latest_checkpoint()
                    if latest:
                        play(latest)
                    else:
                        print("No model found. Train first: python main.py")
            else:
                play(sys.argv[2])

        else:
            # explicit path passed — resume from it
            train(resume_from=sys.argv[1])

    else:
        # ── no args: auto-detect latest checkpoint ────────────
        latest = get_latest_checkpoint()

        if latest:
            step_count = re.search(r"_(\d+)_steps", latest).group(1)
            print("=" * 55)
            print(f"  Found checkpoint at step {int(step_count):,}")
            print(f"  → {latest}")
            print(f"  Resuming automatically...")
            print("=" * 55)
            train(resume_from=latest)
        else:
            print("=" * 55)
            print("  No checkpoint found — starting fresh training run")
            print("=" * 55)
            train()