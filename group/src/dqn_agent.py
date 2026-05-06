from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn


@dataclass
class DQNConfig:
    episodes: int = 300
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    memory_size: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992
    target_update_steps: int = 50
    transaction_cost: float = 0.001
    hidden_dim: int = 64
    window_size: int = 64


@dataclass
class TrainedDQN:
    agent: "DQNAgent"
    scaler: MinMaxScaler
    feature_columns: list[str]
    training_rewards: list[float]
    config: DQNConfig


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TradingEnv:
    def __init__(self, features: np.ndarray, prices: np.ndarray, dates: list[pd.Timestamp], transaction_cost: float) -> None:
        if len(prices) < 2:
            raise ValueError("TradingEnv requires at least two rows of market data.")
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.dates = dates
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.cash = 1.0
        self.shares = 0.0
        return self._state()

    def _equity(self, price: float) -> float:
        return float(self.cash + self.shares * price)

    def _state(self) -> np.ndarray:
        equity = max(self._equity(float(self.prices[self.idx])), 1e-9)
        position = 1.0 if self.shares > 0 else 0.0
        cash_ratio = self.cash / equity
        return np.concatenate([self.features[self.idx], np.array([position, cash_ratio], dtype=np.float32)]).astype(
            np.float32
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | str]]:
        price_t = float(self.prices[self.idx])
        equity_before = max(self._equity(price_t), 1e-9)

        if action == 1 and self.shares == 0.0:
            investable_cash = self.cash * (1.0 - self.transaction_cost)
            self.shares = investable_cash / max(price_t, 1e-9)
            self.cash = 0.0
        elif action == 2 and self.shares > 0.0:
            self.cash = self.shares * price_t * (1.0 - self.transaction_cost)
            self.shares = 0.0

        next_idx = min(self.idx + 1, len(self.prices) - 1)
        next_price = float(self.prices[next_idx])
        equity_after = max(self._equity(next_price), 1e-9)
        reward = float(np.log(equity_after / equity_before))

        self.idx = next_idx
        done = self.idx >= len(self.prices) - 1
        info = {"equity": equity_after, "date": str(self.dates[self.idx].date())}
        return self._state(), reward, done, info


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.config = config
        self.policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(config.memory_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1).values
            targets = rewards_t + self.config.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


def train_dqn(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    config: DQNConfig | None = None,
) -> TrainedDQN:
    config = config or DQNConfig()
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(train_df[feature_columns])
    prices = train_df["close"].to_numpy(dtype=np.float32)
    dates = train_df["date"].tolist()

    state_dim = len(feature_columns) + 2
    agent = DQNAgent(state_dim=state_dim, action_dim=3, config=config, seed=seed)
    rewards: list[float] = []
    epsilon = config.epsilon_start
    rng = np.random.default_rng(seed)

    for _ in range(config.episodes):
        if len(train_df) > config.window_size + 2:
            start_idx = int(rng.integers(0, len(train_df) - config.window_size - 1))
            end_idx = start_idx + config.window_size + 1
        else:
            start_idx = 0
            end_idx = len(train_df)

        env = TradingEnv(
            features=scaled_features[start_idx:end_idx],
            prices=prices[start_idx:end_idx],
            dates=dates[start_idx:end_idx],
            transaction_cost=config.transaction_cost,
        )

        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

    return TrainedDQN(agent=agent, scaler=scaler, feature_columns=feature_columns, training_rewards=rewards, config=config)


def evaluate_dqn(trained: TrainedDQN, test_df: pd.DataFrame) -> pd.DataFrame:
    scaled_features = trained.scaler.transform(test_df[trained.feature_columns])
    env = TradingEnv(
        features=scaled_features,
        prices=test_df["close"].to_numpy(dtype=np.float32),
        dates=test_df["date"].tolist(),
        transaction_cost=trained.config.transaction_cost,
    )

    rows = [
        {
            "date": pd.to_datetime(test_df.iloc[0]["date"]),
            "equity": 1.0,
            "action": "START",
        }
    ]

    state = env.reset()
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    done = False
    while not done:
        action = trained.agent.select_action(state, epsilon=0.0)
        next_state, _, done, info = env.step(action)
        rows.append(
            {
                "date": pd.to_datetime(info["date"]),
                "equity": float(info["equity"]),
                "action": action_names[action],
            }
        )
        state = next_state

    return pd.DataFrame(rows)


def evaluate_buy_and_hold(test_df: pd.DataFrame, transaction_cost: float = 0.001) -> pd.DataFrame:
    first_price = float(test_df.iloc[0]["close"])
    shares = (1.0 - transaction_cost) / max(first_price, 1e-9)
    rows = [
        {
            "date": pd.to_datetime(test_df.iloc[0]["date"]),
            "equity": 1.0,
            "action": "BUY",
        }
    ]
    for idx, row in test_df.iloc[1:].iterrows():
        equity = shares * float(row["close"])
        rows.append(
            {
                "date": pd.to_datetime(row["date"]),
                "equity": equity,
                "action": "HOLD",
            }
        )
    return pd.DataFrame(rows)
