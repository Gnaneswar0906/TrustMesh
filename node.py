"""
node.py — Node model for TrustMesh simulation.

Each node has an ID, a trust score, and a type (good / faulty / malicious).
Voting behavior varies by type:
  - Good:      Always votes correctly
  - Malicious: Votes randomly (50% chance of being wrong)
  - Faulty:    70% correct, may be slow or unresponsive
"""

import random


class Node:
    """Represents a single node in the distributed system."""

    def __init__(self, node_id: int, node_type: str = "good", trust_score: float = 5.0):
        """
        Args:
            node_id:     Unique identifier for the node.
            node_type:   One of 'good', 'faulty', 'malicious'.
            trust_score: Initial trust score (0–10 scale).
        """
        if node_type not in ("good", "faulty", "malicious"):
            raise ValueError(f"Invalid node_type: {node_type}")

        self.id = node_id
        self.node_type = node_type
        self.trust_score = max(0.0, min(10.0, trust_score))
        self.is_leader = False
        self.response_time = 0.0          # milliseconds for last vote
        self.vote_history: list[dict] = []
        self.trust_history: list[float] = [self.trust_score]

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def vote(self, correct_answer: str) -> dict:
        """
        Cast a vote based on node type.

        Returns a dict:
            {
                "node_id":       int,
                "vote":          str | None,    # "YES" / "NO" / None (no response)
                "is_correct":    bool | None,
                "responded":     bool,
                "was_slow":      bool,
                "response_time": float          # ms
            }
        """
        opposite = "NO" if correct_answer == "YES" else "YES"

        if self.node_type == "good":
            return self._cast(correct_answer, correct_answer, slow_chance=0.0, drop_chance=0.0)

        elif self.node_type == "malicious":
            # 95% chance of voting WRONG (strongly adversarial)
            chosen = opposite if random.random() < 0.95 else correct_answer
            return self._cast(chosen, correct_answer, slow_chance=0.0, drop_chance=0.0)

        else:  # faulty
            # 50 % correct, 25 % slow, 20 % no-response
            chosen = correct_answer if random.random() < 0.5 else opposite
            return self._cast(chosen, correct_answer, slow_chance=0.25, drop_chance=0.20)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cast(self, chosen: str, correct_answer: str,
              slow_chance: float, drop_chance: float) -> dict:
        """Build the vote result dict with optional slowness / drop."""
        roll = random.random()

        # No response
        if roll < drop_chance:
            result = {
                "node_id": self.id,
                "vote": None,
                "is_correct": None,
                "responded": False,
                "was_slow": False,
                "response_time": 0.0,
            }
            self.vote_history.append(result)
            return result

        # Slow response
        was_slow = roll < (drop_chance + slow_chance)
        base_time = random.uniform(1, 5)        # normal latency ms
        slow_extra = random.uniform(50, 150) if was_slow else 0.0
        self.response_time = base_time + slow_extra

        result = {
            "node_id": self.id,
            "vote": chosen,
            "is_correct": (chosen == correct_answer),
            "responded": True,
            "was_slow": was_slow,
            "response_time": self.response_time,
        }
        self.vote_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def clamp_trust(self):
        """Keep trust in [0, 10]."""
        self.trust_score = max(0.0, min(10.0, self.trust_score))

    def record_trust(self):
        """Snapshot current trust into history."""
        self.trust_history.append(self.trust_score)

    def __repr__(self):
        return (f"Node(id={self.id}, type={self.node_type}, "
                f"trust={self.trust_score:.1f}, leader={self.is_leader})")
