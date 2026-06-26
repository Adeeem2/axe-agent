import sys
from train import train
from clone_hero_env import CloneHeroEnv
import time

def test_env():
    print("Testing Environment with random actions...")
    env = CloneHeroEnv(debug=True)
    obs, info = env.reset()
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i} | Reward: {reward:.2f} | Action: {action}")
        time.sleep(0.05)
    
    env.close()
    print("Test complete. Check processed_obs_debug.png for visual verification.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_env()
    else:
        train()
