import os
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from clone_hero_env import CloneHeroEnv

def train():
    # Countdown to switch to the game
    print("Please launch Clone Hero and navigate to the song.")
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    # Initialize environment
    env = Monitor(CloneHeroEnv(debug=False))

    # Create directories for models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Define the model (PPO with CNN policy is ideal for image input)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./logs/"
    )

    # Callback to save model every 5000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ppo_clonehero'
    )

    print("Training started. Press Ctrl+C to stop.")
    run_name = f"PPO_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        # Total timesteps can be adjusted. 100,000 is a good "small training" start.
        model.learn(
            total_timesteps=20000,
            callback=checkpoint_callback,
            tb_log_name=run_name,
            log_interval=1
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # Save the final model
    model.save("models/ppo_clonehero_final")
    print("Training finished. Final model saved.")
    
    env.close()

if __name__ == "__main__":
    train()
