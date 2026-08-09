"""
Expectiminimax Game Tree Search for HFT Decision Making
Adapted for stochastic market environments with alpha-beta pruning.
Based on DM828 course materials and 2048 AI implementation.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import math


class NodeType(Enum):
    MAX = "MAX"  # Agent's turn (profit maximization)
    MIN = "MIN"  # Market adversary turn (worst-case scenario)
    CHANCE = "CHANCE"  # Random market events


@dataclass
class GameNode:
    """Represents a node in the expectiminimax game tree."""

    state: np.ndarray  # Market state vector
    node_type: NodeType  # MAX, MIN, or CHANCE
    depth: int  # Depth from root
    parent_action: Optional[int] = None
    children: List["GameNode"] = None
    value: float = 0.0
    alpha: float = -math.inf
    beta: float = math.inf


class ExpectiminimaxEngine:
    """
    Expectiminimax search engine for HFT decision-making.

    Implements:
    - Depth-limited search with alpha-beta pruning
    - Stochastic node handling for market uncertainty
    - Dynamic depth based on market volatility

    References:
    - DM828 Introduction to AI (Syddansk Universitet): Expectiminimax algorithm
    - 2048 AI Player (GitHub): Alpha-beta pruning implementation
    """

    def __init__(self, max_depth: int = 3, branching_factor: int = 5):
        """
        Initialize search engine.

        Args:
            max_depth: Maximum search depth
            branching_factor: Maximum number of actions to consider per node
        """
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.node_count = 0
        self.pruned_count = 0

    def get_available_actions(self, state: np.ndarray) -> List[int]:
        """
        Get available actions for current state.
        Actions are price levels relative to best bid/ask.

        Args:
            state: Current market state

        Returns:
            List of possible actions (price offsets in ticks)
        """
        # Simplified: actions from -K to +K with step 1
        K = 5  # 5 ticks above/below best price
        actions = list(range(-K, K + 1))
        return actions

    def get_market_responses(
        self, state: np.ndarray, action: int
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get possible market responses after an action.
        Each response has a probability weight.

        Args:
            state: Current market state
            action: Agent's action

        Returns:
            List of (next_state, probability) tuples
        """
        # Simplified: 3 possible market responses
        # In practice, this would use a learned market dynamics model
        responses = []

        # Response 1: Market moves favorably (30% probability)
        state_favorable = state + np.random.normal(0.01, 0.005, size=state.shape)
        responses.append((state_favorable, 0.3))

        # Response 2: Market moves against (30% probability)
        state_adverse = state - np.random.normal(0.01, 0.005, size=state.shape)
        responses.append((state_adverse, 0.3))

        # Response 3: Market stays neutral (40% probability)
        responses.append((state, 0.4))

        return responses

    def evaluate_state(self, state: np.ndarray, inventory: float = 0.0) -> float:
        """
        Evaluation function for a market state.

        Combines:
        - Expected profit (PnL)
        - Risk penalty (inventory squared)
        - Microstructure alpha

        Args:
            state: Market state vector
            inventory: Current inventory position

        Returns:
            Evaluation score
        """
        # Extract features from state (simplified)
        # Assuming state contains: [mid_price, spread, volume_imbalance, volatility]
        if len(state) >= 4:
            mid_price = state[0]
            spread = state[1]
            imbalance = state[2]
            volatility = state[3]
        else:
            mid_price = 100.0  # Default
            spread = 0.01
            imbalance = 0.0
            volatility = 0.001

        # Profit component: expected return from mid-price momentum
        expected_return = imbalance * spread * 10.0

        # Risk penalty: quadratic inventory cost
        risk_penalty = -0.01 * (inventory**2)

        # Microstructure alpha: order flow imbalance signal
        alpha = imbalance * 0.1

        # Combined evaluation
        evaluation = expected_return + risk_penalty + alpha

        return evaluation

    def expectiminimax(
        self, node: GameNode, alpha: float = -math.inf, beta: float = math.inf
    ) -> float:
        """
        Expectiminimax search with alpha-beta pruning.

        Handles:
        - MAX nodes: maximize over child values
        - MIN nodes: minimize over child values
        - CHANCE nodes: expected value over probabilistic outcomes

        Args:
            node: Current game node
            alpha: Alpha value for pruning
            beta: Beta value for pruning

        Returns:
            Node value
        """
        self.node_count += 1
        node.alpha = alpha
        node.beta = beta

        # Terminal condition: max depth reached or terminal state
        if node.depth >= self.max_depth:
            node.value = self.evaluate_state(node.state)
            return node.value

        if node.node_type == NodeType.MAX:
            return self._max_value(node, alpha, beta)
        elif node.node_type == NodeType.MIN:
            return self._min_value(node, alpha, beta)
        elif node.node_type == NodeType.CHANCE:
            return self._chance_value(node)
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

    def _max_value(self, node: GameNode, alpha: float, beta: float) -> float:
        """
        Process MAX node: maximize over child actions.
        """
        value = -math.inf
        actions = self.get_available_actions(node.state)

        # Limit branching factor
        actions = actions[: self.branching_factor]

        for action in actions:
            # Get possible market responses after action
            responses = self.get_market_responses(node.state, action)

            for next_state, prob in responses:
                # Create MIN node for market response
                child = GameNode(
                    state=next_state,
                    node_type=NodeType.MIN,
                    depth=node.depth + 1,
                    parent_action=action,
                )
                child_value = self.expectiminimax(child, alpha, beta)
                value = max(value, child_value)

                # Alpha-beta pruning
                if value >= beta:
                    self.pruned_count += 1
                    break
                alpha = max(alpha, value)

            if value >= beta:
                break

        node.value = value
        return value

    def _min_value(self, node: GameNode, alpha: float, beta: float) -> float:
        """
        Process MIN node: minimize over child market outcomes.
        """
        value = math.inf
        actions = self.get_available_actions(node.state)
        actions = actions[: self.branching_factor]

        for action in actions:
            responses = self.get_market_responses(node.state, action)

            for next_state, prob in responses:
                # Create CHANCE node for random market events
                child = GameNode(
                    state=next_state,
                    node_type=NodeType.CHANCE,
                    depth=node.depth + 1,
                    parent_action=action,
                )
                child_value = self.expectiminimax(child, alpha, beta)
                value = min(value, child_value)

                # Alpha-beta pruning
                if value <= alpha:
                    self.pruned_count += 1
                    break
                beta = min(beta, value)

            if value <= alpha:
                break

        node.value = value
        return value

    def _chance_value(self, node: GameNode) -> float:
        """
        Process CHANCE node: expected value over probabilistic outcomes.
        """
        expected_value = 0.0
        actions = self.get_available_actions(node.state)
        actions = actions[: self.branching_factor]

        total_prob = 0.0

        for action in actions:
            responses = self.get_market_responses(node.state, action)

            for next_state, prob in responses:
                # Create MAX node for next agent decision
                child = GameNode(
                    state=next_state,
                    node_type=NodeType.MAX,
                    depth=node.depth + 1,
                    parent_action=action,
                )
                child_value = self.expectiminimax(child)
                expected_value += prob * child_value
                total_prob += prob

        if total_prob > 0:
            expected_value /= total_prob

        node.value = expected_value
        return expected_value

    def get_best_action(
        self, state: np.ndarray, inventory: float = 0.0
    ) -> Tuple[int, Dict]:
        """
        Get best action from current state using expectiminimax.

        Args:
            state: Current market state
            inventory: Current inventory position

        Returns:
            Best action and search statistics
        """
        self.node_count = 0
        self.pruned_count = 0

        root = GameNode(state=state, node_type=NodeType.MAX, depth=0)

        best_value = -math.inf
        best_action = 0

        actions = self.get_available_actions(state)
        actions = actions[: self.branching_factor]

        for action in actions:
            responses = self.get_market_responses(state, action)
            action_value = 0.0

            for next_state, prob in responses:
                child = GameNode(
                    state=next_state,
                    node_type=NodeType.MIN,
                    depth=1,
                    parent_action=action,
                )
                child_value = self.expectiminimax(child)
                action_value += prob * child_value

            if action_value > best_value:
                best_value = action_value
                best_action = action

        stats = {
            "nodes_expanded": self.node_count,
            "pruned_nodes": self.pruned_count,
            "best_value": best_value,
        }

        return best_action, stats
