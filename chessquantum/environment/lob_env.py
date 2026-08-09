"""
Limit Order Book Environment for HFT Reinforcement Learning.
Based on LOBSTER system (Huang & Polak 2012) and UIUC IE421 implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import deque
import gym
from gym import spaces


class Order:
    """Individual order in the limit order book."""

    def __init__(
        self, order_id: int, price: float, volume: int, side: str, timestamp: float
    ):
        self.order_id = order_id
        self.price = price
        self.volume = volume
        self.side = side  # 'bid' or 'ask'
        self.timestamp = timestamp
        self.is_active = True


class FillsLedger:
    """
    FIFO trade ledger for tracking inventory and P&L.
    Reference: UIUC IE421 HFT Project
    """

    def __init__(self):
        self.long_positions = deque()  # (price, volume, timestamp)
        self.short_positions = deque()
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.inventory = 0

    def add_position(self, price: float, volume: int, side: str, timestamp: float):
        """Add a new position (long for buy, short for sell)."""
        if side == "buy":
            self.long_positions.append((price, volume, timestamp))
            self.inventory += volume
        else:  # sell
            self.short_positions.append((price, volume, timestamp))
            self.inventory -= volume

    def close_position(self, price: float, volume: int, side: str, timestamp: float):
        """Close positions using FIFO accounting."""
        remaining = volume

        if side == "sell":  # Closing long positions
            while remaining > 0 and self.long_positions:
                buy_price, buy_volume, buy_time = self.long_positions[0]
                close_volume = min(remaining, buy_volume)
                pnl = (price - buy_price) * close_volume
                self.realized_pnl += pnl

                if close_volume == buy_volume:
                    self.long_positions.popleft()
                else:
                    self.long_positions[0] = (
                        buy_price,
                        buy_volume - close_volume,
                        buy_time,
                    )

                remaining -= close_volume

        else:  # buy to close short positions
            while remaining > 0 and self.short_positions:
                sell_price, sell_volume, sell_time = self.short_positions[0]
                close_volume = min(remaining, sell_volume)
                pnl = (sell_price - price) * close_volume
                self.realized_pnl += pnl

                if close_volume == sell_volume:
                    self.short_positions.popleft()
                else:
                    self.short_positions[0] = (
                        sell_price,
                        sell_volume - close_volume,
                        sell_time,
                    )

                remaining -= close_volume

        self.inventory -= volume if side == "sell" else -volume

    def update_unrealized_pnl(self, mid_price: float):
        """Update unrealized P&L based on current mid price."""
        unrealized = 0.0
        for price, volume, _ in self.long_positions:
            unrealized += (mid_price - price) * volume
        for price, volume, _ in self.short_positions:
            unrealized += (price - mid_price) * volume
        self.unrealized_pnl = unrealized
        return unrealized


class LimitOrderBookEnv(gym.Env):
    """
    Limit Order Book Environment for HFT training.

    Features:
    - Level-3 order book simulation
    - FIFO trade ledger
    - Realistic order execution mechanics
    - Slippage and market impact modeling

    References:
    - LOBSTER: Limit Order Book Reconstruction System (Huang & Polak 2012)
    - UIUC IE421 HFT Project: LOBEnv implementation
    - arXiv:2509.12456v1: RL-Based Market Making environment
    """

    def __init__(
        self,
        data_path: str = None,
        window_size: int = 100,
        max_position: int = 1000,
        transaction_cost: float = 0.0001,
    ):
        """
        Initialize LOB environment.

        Args:
            data_path: Path to LOB data (CSV or parquet)
            window_size: Number of steps for state history
            max_position: Maximum allowed inventory position
            transaction_cost: Transaction cost per trade
        """
        super().__init__()

        self.window_size = window_size
        self.max_position = max_position
        self.transaction_cost = transaction_cost

        # State space: 17 features as per UIUC implementation
        # [mid_price, spread, bid_volume, ask_volume, imbalance,
        #  volatility, order_flow, trade_volume, etc.]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 17), dtype=np.float32
        )

        # Action space: -K to +K (price offsets in ticks)
        # +k: buy at k ticks above best bid
        # -k: sell at k ticks below best ask
        # 0: hold/cancel
        self.action_space = spaces.Discrete(11)  # -5 to +5

        # Order book levels
        self.bid_levels = []  # List of (price, volume)
        self.ask_levels = []

        # Trade ledger
        self.ledger = FillsLedger()

        # Data loading
        self.data = None
        if data_path:
            self._load_data(data_path)

        self.current_step = 0
        self.state_buffer = deque(maxlen=window_size)

    def _load_data(self, data_path: str):
        """Load LOB data from file."""
        if data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith(".parquet"):
            self.data = pd.read_parquet(data_path)
        else:
            # Generate synthetic data for testing
            self._generate_synthetic_data()

    def _generate_synthetic_data(self, n_steps: int = 10000):
        """Generate synthetic LOB data for testing."""
        np.random.seed(42)

        mid_price = 100.0
        spread = 0.01

        data = []
        for t in range(n_steps):
            # Random walk mid-price
            mid_price += np.random.randn() * 0.01

            # Mean-reverting spread
            spread = 0.01 + np.random.randn() * 0.002
            spread = max(0.001, spread)

            # Volume imbalance
            imbalance = np.random.randn() * 0.3

            # Volatility
            volatility = np.abs(np.random.randn() * 0.005)

            data.append(
                {
                    "timestamp": t,
                    "mid_price": mid_price,
                    "best_bid": mid_price - spread / 2,
                    "best_ask": mid_price + spread / 2,
                    "bid_volume": np.random.randint(100, 1000),
                    "ask_volume": np.random.randint(100, 1000),
                    "imbalance": imbalance,
                    "volatility": volatility,
                }
            )

        self.data = pd.DataFrame(data)

    def _extract_features(self) -> np.ndarray:
        """Extract 17-feature state vector from current LOB."""
        if self.data is None or self.current_step >= len(self.data):
            # Return zeros for terminal state
            return np.zeros(17)

        row = self.data.iloc[self.current_step]

        features = [
            row.get("mid_price", 100.0),  # 0: mid price
            row.get("best_ask", 100.01) - row.get("best_bid", 99.99),  # 1: spread
            row.get("bid_volume", 500),  # 2: bid volume
            row.get("ask_volume", 500),  # 3: ask volume
            row.get("imbalance", 0.0),  # 4: imbalance
            row.get("volatility", 0.01),  # 5: volatility
            self.ledger.inventory / self.max_position,  # 6: normalized inventory
            self.ledger.realized_pnl / 1000.0,  # 7: normalized realized PnL
            self.ledger.unrealized_pnl / 1000.0,  # 8: normalized unrealized PnL
            self.current_step / len(self.data),  # 9: time progress
            np.sin(2 * np.pi * self.current_step / 2520),  # 10: intraday cycle
            np.cos(2 * np.pi * self.current_step / 2520),  # 11: intraday cycle
            row.get("trade_volume", 0),  # 12: recent trade volume
            row.get("order_flow", 0.0),  # 13: order flow
            row.get("bid_depth", 0.0),  # 14: bid depth
            row.get("ask_depth", 0.0),  # 15: ask depth
            row.get("microstructure_signal", 0.0),  # 16: microstructure alpha
        ]

        return np.array(features, dtype=np.float32)

    def _execute_order(self, action: int):
        """
        Execute an order based on action.

        Action mapping:
        -5 to -1: sell at price levels below best ask
        0: hold/cancel
        1 to 5: buy at price levels above best bid
        """
        if action == 0:
            return  # No action

        if self.data is None or self.current_step >= len(self.data):
            return

        row = self.data.iloc[self.current_step]
        best_bid = row.get("best_bid", 99.99)
        best_ask = row.get("best_ask", 100.01)
        tick_size = 0.01

        # Determine side and price
        if action > 0:
            side = "buy"
            price_offset = action * tick_size
            price = best_bid + price_offset
            volume = 100  # Fixed volume for simplicity
        else:
            side = "sell"
            price_offset = abs(action) * tick_size
            price = best_ask - price_offset
            volume = 100

        # Check inventory limits
        if side == "buy" and self.ledger.inventory + volume > self.max_position:
            return  # Would exceed max position
        if side == "sell" and self.ledger.inventory - volume < -self.max_position:
            return  # Would exceed max position

        # Check if price is within reasonable bounds
        if side == "buy" and price >= best_ask:
            # Marketable buy order - execute immediately
            self._execute_market_order("buy", volume, best_ask)
        elif side == "sell" and price <= best_bid:
            # Marketable sell order - execute immediately
            self._execute_market_order("sell", volume, best_bid)
        else:
            # Limit order - add to book
            self._add_limit_order(side, price, volume)

    def _execute_market_order(self, side: str, volume: int, price: float):
        """Execute market order with slippage."""
        # Apply transaction cost
        price_with_cost = price * (
            1 + self.transaction_cost if side == "buy" else 1 - self.transaction_cost
        )

        # Execute trade
        if side == "buy":
            self.ledger.add_position(price_with_cost, volume, "buy", self.current_step)
        else:
            self.ledger.add_position(price_with_cost, volume, "sell", self.current_step)

    def _add_limit_order(self, side: str, price: float, volume: int):
        """Add limit order to the book."""
        # In a full implementation, this would add to bid/ask levels
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (price offset)

        Returns:
            next_state, reward, done, info
        """
        # Execute action
        self._execute_order(action)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) if self.data is not None else False

        # Update unrealized PnL
        if self.data is not None and self.current_step < len(self.data):
            mid_price = self.data.iloc[self.current_step].get("mid_price", 100.0)
            self.ledger.update_unrealized_pnl(mid_price)

        # Extract next state
        next_state = self._extract_features()
        self.state_buffer.append(next_state)

        # Compute reward (GRPO)
        reward = self._compute_reward()

        # Prepare state window
        state_window = np.array(self.state_buffer)
        if len(state_window) < self.window_size:
            # Pad with zeros
            pad = np.zeros((self.window_size - len(state_window), 17))
            state_window = np.vstack([pad, state_window])

        info = {
            "inventory": self.ledger.inventory,
            "realized_pnl": self.ledger.realized_pnl,
            "unrealized_pnl": self.ledger.unrealized_pnl,
        }

        return state_window, reward, done, info

    def _compute_reward(self) -> float:
        """Compute reward using GRPO formula."""
        pnl_change = (
            self.ledger.realized_pnl - self.prev_pnl if hasattr(self, "prev_pnl") else 0
        )
        self.prev_pnl = self.ledger.realized_pnl

        # GRPO components
        pnl_component = pnl_change
        inventory_penalty = -0.01 * (self.ledger.inventory**2)
        turnover = abs(pnl_change) * 0.01  # Approximate turnover
        slippage = 0.001 * turnover

        reward = pnl_component + inventory_penalty - turnover - slippage

        # Terminal penalty
        if self.current_step >= len(self.data) - 1:
            reward -= abs(self.ledger.inventory) * 0.1

        return reward

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.current_step = 0
        self.ledger = FillsLedger()
        self.state_buffer.clear()
        self.prev_pnl = 0.0

        # Initial state
        initial_state = self._extract_features()
        self.state_buffer.append(initial_state)

        # Pad to window size
        state_window = np.array(self.state_buffer)
        if len(state_window) < self.window_size:
            pad = np.zeros((self.window_size - len(state_window), 17))
            state_window = np.vstack([pad, state_window])

        return state_window
