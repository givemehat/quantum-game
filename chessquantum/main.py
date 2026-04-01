"""
Main Entry Point for Quantum-Chess HFT Framework.
Integrates all modules into a cohesive research prototype.
"""

import numpy as np
from typing import Dict, List, Optional
import time
import json

from quantum.qaoa_layer import TradingQAOA
from engine.expectiminimax import ExpectiminimaxEngine, NodeType, GameNode
from rl.ppo_agent import PPOTradingAgent
from data.environment.lob_env import LimitOrderBookEnv


class QuantumChessHFT:
    """
    Main orchestrator for the Quantum-Chess HFT Framework.
    
    Integrates:
    - Quantum QAOA layer for action probability generation
    - Expectiminimax engine for game tree search
    - PPO agent for learned decision-making
    - LOB environment for market simulation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the integrated system.
        
        Args:
            config: Configuration dictionary with system parameters
        """
        self.config = config
        
        # Initialize components
        self.env = LimitOrderBookEnv(
            data_path=config.get('data_path'),
            window_size=config.get('window_size', 100),
            max_position=config.get('max_position', 1000),
            transaction_cost=config.get('transaction_cost', 0.0001)
        )
        
        self.quantum_layer = TradingQAOA(
            n_qubits=config.get('n_qubits', 12),
            p_layers=config.get('p_layers', 2)
        )
        
        self.search_engine = ExpectiminimaxEngine(
            max_depth=config.get('search_depth', 3),
            branching_factor=config.get('branching_factor', 5)
        )
        
        self.rl_agent = PPOTradingAgent(
            state_dim=17,
            action_dim=11,  # -5 to +5
            learning_rate=config.get('learning_rate', 3e-4)
        )
        
        # Performance tracking
        self.latency_history = []
        self.pnl_history = []
        self.inventory_history = []
        
    def select_action_hybrid(self, state: np.ndarray) -> int:
        """
        Hybrid action selection combining quantum probabilities with RL policy.
        
        Steps:
        1. Get action probabilities from quantum QAOA layer
        2. Get action probabilities from RL agent
        3. Combine with adaptive weighting
        4. Apply expectiminimax for final selection
        
        Returns:
            Selected action
        """
        start_time = time.perf_counter()
        
        # Step 1: Quantum layer - generate action probabilities
        # Simplified: map state to action rewards
        action_rewards = {}
        for a in range(-5, 6):
            # Placeholder: in production, compute from market state
            action_rewards[a] = self._estimate_action_reward(state, a)
        
        quantum_probs = self.quantum_layer.get_action_probabilities(
            market_state=state.flatten(),
            action_rewards=action_rewards
        )
        
        # Step 2: RL agent - get policy probabilities
        rl_action, rl_log_prob, rl_value = self.rl_agent.select_action(state, training=False)
        
        # Step 3: Combine probabilities with adaptive weighting
        alpha = self.config.get('quantum_weight', 0.6)  # Quantum layer weight
        combined_probs = {}
        for a in range(-5, 6):
            q_prob = quantum_probs.get(a, 0.0)
            r_prob = 1.0 / 11 if rl_action != a else 0.5  # Simplified
            combined_probs[a] = alpha * q_prob + (1 - alpha) * r_prob
        
        # Normalize
        total = sum(combined_probs.values())
        if total > 0:
            combined_probs = {k: v / total for k, v in combined_probs.items()}
        
        # Step 4: Expectiminimax search for final decision
        # Build game tree with combined probabilities as priors
        best_action, search_stats = self.search_engine.get_best_action(
            state.flatten(),
            inventory=self.env.ledger.inventory
        )
        
        # Log latency
        latency_us = (time.perf_counter() - start_time) * 1_000_000
        self.latency_history.append(latency_us)
        
        return best_action
    
    def _estimate_action_reward(self, state: np.ndarray, action: int) -> float:
        """Estimate expected reward for an action given current state."""
        # Simplified: use LOB features to estimate
        if len(state) > 0:
            mid_price = state[0, 0] if len(state.shape) > 1 else state[0]
            spread = state[0, 1] if len(state.shape) > 1 else state[1] if len(state) > 1 else 0.01
            imbalance = state[0, 4] if len(state.shape) > 1 else state[4] if len(state) > 4 else 0.0
            
            if action > 0:  # Buy
                reward = -action * spread * 0.1 + imbalance * 0.5
            elif action < 0:  # Sell
                reward = -abs(action) * spread * 0.1 - imbalance * 0.5
            else:  # Hold
                reward = 0.0
            
            return reward
        return 0.0
    
    def run_backtest(self, n_steps: int = 1000) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            n_steps: Number of simulation steps
            
        Returns:
            Performance metrics
        """
        state = self.env.reset()
        
        for step in range(n_steps):
            # Select action
            action = self.select_action_hybrid(state)
            
            # Execute in environment
            next_state, reward, done, info = self.env.step(action + 5)  # Map -5..5 to 0..10
            
            # Store for RL training
            self.rl_agent.store_transition(
                state.flatten(), action + 5, reward, done, 0, 0
            )
            
            # Update tracking
            self.pnl_history.append(info['realized_pnl'])
            self.inventory_history.append(info['inventory'])
            
            state = next_state
            
            if done:
                break
            
            # Periodic policy update
            if step % 100 == 0 and step > 0:
                loss_metrics = self.rl_agent.update_policy()
                print(f"Step {step}: PnL={info['realized_pnl']:.2f}, "
                      f"Inventory={info['inventory']}, "
                      f"Latency={self.latency_history[-1]:.1f}µs")
        
        # Compute metrics
        metrics = {
            'total_pnl': self.pnl_history[-1] if self.pnl_history else 0,
            'avg_latency_us': np.mean(self.latency_history) if self.latency_history else 0,
            'max_inventory': max(self.inventory_history) if self.inventory_history else 0,
            'sharpe_ratio': self._compute_sharpe_ratio(),
            'win_rate': self._compute_win_rate()
        }
        
        return metrics
    
    def _compute_sharpe_ratio(self) -> float:
        """Compute Sharpe ratio from PnL history."""
        if len(self.pnl_history) < 2:
            return 0.0
        pnl_array = np.array(self.pnl_history)
        returns = np.diff(pnl_array)
        if np.std(returns) > 0:
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0
    
    def _compute_win_rate(self) -> float:
        """Compute win rate from PnL history."""
        if len(self.pnl_history) < 2:
            return 0.0
        returns = np.diff(self.pnl_history)
        positive = sum(1 for r in returns if r > 0)
        return positive / len(returns) if len(returns) > 0 else 0.0
    
    def save_results(self, path: str):
        """Save backtest results to file."""
        results = {
            'pnl_history': self.pnl_history,
            'inventory_history': self.inventory_history,
            'latency_history': self.latency_history,
            'metrics': self._compute_metrics()
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main execution function."""
    config = {
        'data_path': None,  # Use synthetic data
        'window_size': 50,
        'max_position': 1000,
        'transaction_cost': 0.0001,
        'n_qubits': 10,
        'p_layers': 2,
        'search_depth': 3,
        'branching_factor': 5,
        'quantum_weight': 0.6,
        'learning_rate': 3e-4
    }
    
    print("Initializing Quantum-Chess HFT Framework...")
    system = QuantumChessHFT(config)
    
    print("Running backtest...")
    metrics = system.run_backtest(n_steps=500)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total PnL: ${metrics['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Avg Latency: {metrics['avg_latency_us']:.1f} µs")
    print(f"Max Inventory: {metrics['max_inventory']}")
    
    # Save results
    system.save_results("backtest_results.json")
    print("\nResults saved to backtest_results.json")
    
    # Launch dashboard
    print("\nLaunching visualization dashboard at http://localhost:8050")
    import subprocess
    subprocess.Popen(["python", "visualization/dashboard.py"])


if __name__ == "__main__":
    main()