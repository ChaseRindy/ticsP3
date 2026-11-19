from pathlib import Path
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# Copy your final saved model here and name the copy my_model.zip.
MODEL_PATH = Path(__file__).resolve().parent / "models/run_20261119_122106_235524/battlezone_400000_steps.zip"
WATCH_STEPS = 2000

agent = DQN.load(MODEL_PATH, device="cpu")

watch_env = make_atari_env(
    "ALE/BattleZone-v5",
    n_envs=1,
    seed=42,
    env_kwargs={
        "frameskip": 1,
        "render_mode": "human"
    },
    wrapper_kwargs={
        "frame_skip": 4,
        "terminal_on_life_loss": False
    }
)
if watch_env is None:
    raise NotImplementedError("Complete the viewing environment setup.")
watch_env = VecFrameStack(watch_env, n_stack=4)
try:
    observation = watch_env.reset()
    for step in range(WATCH_STEPS):
        action, _ = agent.predict(observation, deterministic=True)
        observation, rewards, dones, infos = watch_env.step(action)
finally:
    watch_env.close()
