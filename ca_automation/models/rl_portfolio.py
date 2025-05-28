import numpy as np
import gym
from gym import spaces

class PortfolioEnv(gym.Env):
    def __init__(self):
        super(PortfolioEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.returns = np.array([0.12, 0.10, 0.08])
        self.cov_matrix = np.array([[0.1, 0.01, 0.02], [0.01, 0.08, 0.01], [0.02, 0.01, 0.07]])
        self.current_step = 0
        self.max_steps = 100
        self.weights = np.array([1/3, 1/3, 1/3])

    def reset(self):
        self.current_step = 0
        self.weights = np.array([1/3, 1/3, 1/3])
        return self._get_obs()

    def _get_obs(self):
        market_conditions = np.random.normal(0, 0.1, 3)
        return np.concatenate([self.weights, market_conditions])

    def step(self, action):
        self.weights = action / np.sum(action)
        portfolio_return = np.sum(self.returns * self.weights)
        portfolio_risk = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        reward = portfolio_return - 0.5 * portfolio_risk
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self._get_obs(), reward, done, {}

def optimize_portfolio_rl():
    try:
        env = PortfolioEnv()
        state = env.reset()
        for _ in range(100):
            action = env.action_space.sample()  # Placeholder: Use DQN in production
            state, reward, done, _ = env.step(action)
            if done:
                break
        return env.weights
    except Exception as e:
        print(f"Error in optimize_portfolio_rl: {e}")
        return [0.33, 0.33, 0.34]