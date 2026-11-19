import gymnasium as gym
import ale_py
from datetime import datetime
from pathlib import Path

# Import the DQN agent and Atari helpers from Stable-Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# Change these values for a shorter training run or to skip watching afterward.
TOTAL_TIMESTEPS = 500_000  # Smoke test; use the approved budget for experiments.
CHECKPOINT_FREQUENCY = 100_000

# A new folder for each run protects previously saved models.
SAVE_DIR = Path(__file__).resolve().parent / "models" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
SAVE_DIR.mkdir(parents=True, exist_ok=False)
print(f"Saving models to: {SAVE_DIR}", flush=True)

# ==============================
# STEP 1: CREATE THE ENVIRONMENT
print("Creating training environment...", flush=True)
# TODO: Create your chosen ALE environment using the classroom example.
# Use frameskip=1, render_mode=None, frame_skip=4, terminal_on_life_loss=False.
# Pass seed=SEED to make_atari_env for reproducible setup.
SEED = 42

env = make_atari_env(
    "ALE/BattleZone-v5",
    n_envs=1,
    seed=SEED,
    env_kwargs={
        "frameskip": 1,
        "render_mode": None
    },
    wrapper_kwargs={
        "frame_skip": 4,
        "terminal_on_life_loss": False
    }
)

if env is None:
    raise NotImplementedError("Complete the environment setup before training.")

# Give the agent 4 frames so it can see the ball moving.
env = VecFrameStack(env, n_stack=4)
print("Environment ready!", flush=True)

try:
    # ==============================
    # STEP 2: CREATE THE AGENT
    agent = DQN(
        "CnnPolicy",
        env,
        learning_rate=0.0001,
        buffer_size=50_000,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=1,
        seed=SEED,
    )
    agent.save(str(SAVE_DIR / "battlezone_initial"))
    print("Initial model saved!", flush=True)

    # ==============================
    # STEP 3: TRAIN THE AGENT
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQUENCY,
        save_path=str(SAVE_DIR),
        name_prefix="battlezone",
        verbose=2,
    )
    print("Starting training...", flush=True)
    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        log_interval=1,  # Print training statistics after every episode.
    )

    # ==============================
    # STEP 4: SAVE THE TRAINED AGENT
    agent.save(str(SAVE_DIR / "battlezone_final"))
    print(f"Training complete! Final model saved to: {SAVE_DIR / 'battlezone_final.zip'}", flush=True)
finally:
    env.close()

