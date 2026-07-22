"""
trust.py — Dynamic trust score management.

Rules:
    Correct vote   → +1
    Wrong vote     → −2
    Slow response  → −1
    No response    → −3

Trust is clamped to [0, 10].
"""

from node import Node


def update_trust(node: Node, vote_result: dict) -> float:
    """
    Adjust a node's trust score based on its most recent vote.

    Args:
        node:        The node whose trust to update.
        vote_result: The dict returned by node.vote().

    Returns:
        The new trust score.
    """
    if not vote_result["responded"]:
        # No response at all → heavy penalty
        node.trust_score -= 3
    else:
        # Correct / wrong
        if vote_result["is_correct"]:
            node.trust_score += 1
        else:
            node.trust_score -= 2

        # Slow penalty (additive)
        if vote_result["was_slow"]:
            node.trust_score -= 1

    node.clamp_trust()
    node.record_trust()
    return node.trust_score


def elect_leader(nodes: list[Node]) -> Node:
    """
    Leader election by trust: the node with the highest trust becomes leader.
    Ties are broken by lowest node ID.

    Returns:
        The newly elected leader node.
    """
    # Reset all leader flags
    for n in nodes:
        n.is_leader = False

    # Pick highest trust (lowest id on tie)
    leader = max(nodes, key=lambda n: (n.trust_score, -n.id))
    leader.is_leader = True
    return leader


def get_trust_snapshot(nodes: list[Node]) -> dict[int, float]:
    """Return {node_id: trust_score} for every node."""
    return {n.id: round(n.trust_score, 2) for n in nodes}
