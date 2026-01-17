import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from pathlib import Path

# Add project root to path to import src
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.core.dataset_loader import DatasetLoader

# Define a custom environment for our dataset
class DatasetEnv(gym.Env):
    def __init__(self, df):
        super(DatasetEnv, self).__init__()

        # Assume all features are numeric for simplicity and scale to [0, 1]
        # In practice, you would handle categorical features and scaling appropriately
        self.dataset = df.select_dtypes(include=[np.number]).drop(columns=['label']).apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.labels = df['label']
        self.current_step = 0

        # Define action and observation space
        # Actions: 0 (normal), 1 (alternative)
        self.action_space = spaces.Discrete(2)
        # Observation: features from the dataset
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.dataset.shape[1],), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the environment to the beginning
        self.current_step = 0
        obs = self.dataset.iloc[self.current_step].values
        info = {}
        return obs, info

    def step(self, action):
        # Compare the action with the actual label, reward if correct
        reward = 1 if action == self.labels.iloc[self.current_step] else -1

        self.current_step += 1
        terminated = self.current_step >= len(self.labels)
        truncated = False  # Episode can be truncated if max steps reached

        # Set placeholder for info
        info = {}

        # Provide the next state
        if terminated:
            next_state = np.zeros(self.dataset.shape[1], dtype=np.float32)
        else:
            next_state = self.dataset.iloc[self.current_step % len(self.labels)].values

        return next_state, reward, terminated, truncated, info

# Prepare the environment
loader = DatasetLoader(data_dir='datasets')
df = loader.load_ton_iot()
env = DatasetEnv(df)
check_env(env, warn=True)

# Make environment vectorized for stability
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize the agent
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

mean_reward, std_reward
