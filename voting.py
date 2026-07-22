"""
voting.py — Raft and TrustMesh consensus engines.

Raft:      Simple majority vote (each node = 1 vote).
TrustMesh: Trust-weighted vote (each node's vote weighted by trust score).
"""

from node import Node


def raft_vote(nodes: list[Node], correct_answer: str) -> dict:
    """
    Traditional Raft-style majority vote.

    Returns:
        {
            "system":        "Raft",
            "decision":      str,         # "YES" or "NO"
            "is_correct":    bool,
            "vote_details":  list[dict],  # per-node vote records
            "message_count": int,         # total messages exchanged
            "yes_count":     int,
            "no_count":      int,
        }
    """
    vote_details = []
    yes_count = 0
    no_count = 0
    messages = 0

    for node in nodes:
        result = node.vote(correct_answer)
        vote_details.append(result)

        if result["responded"]:
            messages += 2                     # request + response
            if result["vote"] == "YES":
                yes_count += 1
            else:
                no_count += 1
        else:
            messages += 1                     # request only (no response)

    decision = "YES" if yes_count >= no_count else "NO"
    is_correct = (decision == correct_answer)

    return {
        "system": "Raft",
        "decision": decision,
        "is_correct": is_correct,
        "vote_details": vote_details,
        "message_count": messages,
        "yes_count": yes_count,
        "no_count": no_count,
    }


def trustmesh_vote(nodes: list[Node], correct_answer: str) -> dict:
    """
    TrustMesh trust-weighted vote.

    Instead of counting votes, we sum the trust scores of YES-voters and
    NO-voters.  The side with the higher total trust wins.

    Returns:
        {
            "system":         "TrustMesh",
            "decision":       str,
            "is_correct":     bool,
            "vote_details":   list[dict],
            "message_count":  int,
            "yes_trust":      float,
            "no_trust":       float,
        }
    """
    vote_details = []
    yes_trust = 0.0
    no_trust = 0.0
    messages = 0

    for node in nodes:
        result = node.vote(correct_answer)
        vote_details.append(result)

        if result["responded"]:
            messages += 2
            if result["vote"] == "YES":
                yes_trust += node.trust_score
            else:
                no_trust += node.trust_score
        else:
            messages += 1

    decision = "YES" if yes_trust >= no_trust else "NO"
    is_correct = (decision == correct_answer)

    return {
        "system": "TrustMesh",
        "decision": decision,
        "is_correct": is_correct,
        "vote_details": vote_details,
        "message_count": messages,
        "yes_trust": yes_trust,
        "no_trust": no_trust,
    }
