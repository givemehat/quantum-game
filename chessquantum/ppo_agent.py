"""
Proximal Policy Optimization (PPO) Agent for HFT with Dual-LSTM Architecture.
Based on UIUC IE421 HFT Project and arXiv:2509.12456v1.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from typing import List, Tuple, Dict, Optional


class DualLSTMNetwork(nn.Module):
    """
    Dual-LSTM architecture for actor and critic.

    Uses separate LSTM stacks to capture temporal patterns in LOB data.
    Reference: UIUC IE421 HFT Project - Algorithmic Trading System with RL
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # Actor LSTM
        self.actor_lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, 3)  # Buy, Sell, Hold

        # Critic LSTM
        self.critic_lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        actor_hidden: Optional[Tuple] = None,
        critic_hidden: Optional[Tuple] = None,
    ):
        """
        Forward pass with optional hidden state for sequential processing.

        Args:
            x: Input tensor (batch, seq_len, features)
            actor_hidden: Previous actor LSTM hidden state
            critic_hidden: Previous critic LSTM hidden state

        Returns:
            action_logits, value, (new_actor_hidden, new_critic_hidden)
        """
        # Extract features
        features = self.feature_extractor(x)

        # Actor LSTM
        actor_out, actor_hidden = self.actor_lstm(features, actor_hidden)
        action_logits = self.actor_head(actor_out[:, -1, :])

        # Critic LSTM
        critic_out, critic_hidden = self.critic_lstm(features, critic_hidden)
        value = self.critic_head(critic_out[:, -1, :])

        return action_logits, value, (actor_hidden, critic_hidden)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple:
        """Initialize LSTM hidden states."""
        actor_hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        )
        critic_hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        )
        return (actor_hidden, critic_hidden)


class PPOTradingAgent:
    """
    Proximal Policy Optimization agent for HFT.

    Implements:
    - Dual-LSTM for temporal pattern recognition
    - Clipped surrogate objective
    - GRPO reward function (Generalized Return-Potential-Opportunity)

    References:
    - UIUC GitLab: Algorithmic Trading System with RL (GRPO reward)
    - arXiv:2509.12456v1: RL-Based Market Making as Stochastic Control
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions (buy/sell/hold)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dual-LSTM network
        self.policy = DualLSTMNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.hidden_states = []

        # Current hidden state
        self.current_hidden = None

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action based on current policy.

        Args:
            state: Current market state
            training: Whether to use exploration

        Returns:
            action, log_prob, value
        """
        # Convert to tensor
        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Initialize hidden state if needed
        if self.current_hidden is None:
            self.current_hidden = self.policy.init_hidden(1, self.device)

        # Forward pass
        with torch.no_grad():
            action_logits, value, self.current_hidden = self.policy(
                state_tensor, *self.current_hidden
            )
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)

            if training:
                action = dist.sample()
            else:
                action = torch.argmax(action_probs)

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

        if done:
            self.current_hidden = None

    def compute_grpo_reward(
        self,
        pnl: float,
        inventory: float,
        turnover: float,
        slippage: float,
        opportunity_cost: float = 0.0,
    ) -> float:
        """
        Compute Generalized Return-Potential-Opportunity (GRPO) reward.

        Reference: UIUC IE421 HFT Project
        Balanced reward combining:
        - Realized PnL
        - Inventory risk penalty (quadratic)
        - Turnover cost
        - Slippage estimate
        - Opportunity cost

        Args:
            pnl: Profit/Loss from trades
            inventory: Current inventory position
            turnover: Total trading volume
            slippage: Estimated slippage cost
            opportunity_cost: Cost of missed opportunities

        Returns:
            GRPO reward value
        """
        # PnL component
        pnl_reward = pnl

        # Inventory penalty (quadratic)
        inventory_penalty = -0.01 * (inventory**2)

        # Turnover cost (linear)
        turnover_cost = -0.001 * turnover

        # Slippage penalty
        slippage_penalty = -0.5 * slippage

        # Opportunity cost (optional)
        opportunity_penalty = -0.1 * opportunity_cost

        # Combined reward
        reward = (
            pnl_reward
            + inventory_penalty
            + turnover_cost
            + slippage_penalty
            + opportunity_penalty
        )

        return reward

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float = 0.0,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            last_value: Value of last state
            gae_lambda: GAE lambda parameter

        Returns:
            advantages, returns
        """
        advantages = []
        returns = []

        gae = 0
        next_value = last_value

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * gae_lambda * gae

            advantage = gae
            advantage = (advantage - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])

            next_value = values[t]

        return advantages, returns

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO clipped objective.

        Returns:
            Dictionary of loss metrics
        """
        if len(self.states) == 0:
            return {}

        # Convert buffers to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)

        # Compute returns and advantages
        last_value = values[-1].item() if not self.dones[-1] else 0.0
        advantages, returns = self.compute_gae(
            self.rewards, self.values, self.dones, last_value
        )

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Forward pass with current policy
        # Reshape for LSTM: (batch, seq_len, features)
        states_reshaped = states.unsqueeze(1)

        action_logits, current_values, _ = self.policy(states_reshaped)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(current_values.squeeze(), returns)

        # Total loss
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.hidden_states = []

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
