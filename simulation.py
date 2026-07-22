"""
simulation.py — Multi-round simulation engine.

Runs both Raft and TrustMesh side-by-side for a configurable number of
rounds and collects all per-round data for metrics and visualization.
"""

import copy
import random
import time

from node import Node
from voting import raft_vote, trustmesh_vote
from trust import update_trust, elect_leader, get_trust_snapshot


# ──────────────────────────────────────────────────────────────
# Node factory
# ──────────────────────────────────────────────────────────────

def create_nodes(num_nodes: int,
                 malicious_pct: float = 0.0,
                 faulty_pct: float = 0.0) -> list[Node]:
    """
    Create a list of nodes with the given composition.

    Args:
        num_nodes:     Total number of nodes.
        malicious_pct: Fraction of nodes that are malicious (0.0–1.0).
        faulty_pct:    Fraction of nodes that are faulty (0.0–1.0).

    Returns:
        List[Node] with the specified type distribution.
    """
    num_malicious = int(num_nodes * malicious_pct)
    num_faulty = int(num_nodes * faulty_pct)
    num_good = num_nodes - num_malicious - num_faulty

    nodes: list[Node] = []
    node_id = 0

    for _ in range(num_good):
        nodes.append(Node(node_id, "good"))
        node_id += 1
    for _ in range(num_malicious):
        nodes.append(Node(node_id, "malicious"))
        node_id += 1
    for _ in range(num_faulty):
        nodes.append(Node(node_id, "faulty"))
        node_id += 1

    random.shuffle(nodes)
    return nodes


# ──────────────────────────────────────────────────────────────
# Single-round runner
# ──────────────────────────────────────────────────────────────

def _run_round(nodes: list[Node], correct_answer: str, vote_fn, system_name: str) -> dict:
    """Run one consensus round and return per-round data."""
    start = time.perf_counter()
    result = vote_fn(nodes, correct_answer)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Update trust for TrustMesh nodes only
    if system_name == "TrustMesh":
        for vd in result["vote_details"]:
            node = next(n for n in nodes if n.id == vd["node_id"])
            update_trust(node, vd)

    # Leader election
    old_leader_id = next((n.id for n in nodes if n.is_leader), None)
    leader = elect_leader(nodes)
    leader_changed = (leader.id != old_leader_id) if old_leader_id is not None else True

    return {
        "system": system_name,
        "decision": result["decision"],
        "is_correct": result["is_correct"],
        "latency_ms": elapsed_ms,
        "message_count": result["message_count"],
        "leader_id": leader.id,
        "leader_changed": leader_changed,
        "trust_snapshot": get_trust_snapshot(nodes),
        "vote_details": result["vote_details"],
    }


# ──────────────────────────────────────────────────────────────
# Full simulation
# ──────────────────────────────────────────────────────────────

def run_simulation(num_nodes: int = 10,
                   num_rounds: int = 20,
                   malicious_pct: float = 0.0,
                   faulty_pct: float = 0.0,
                   seed: int | None = 42) -> dict:
    """
    Run a full simulation (both Raft and TrustMesh).

    Both systems start with the SAME node configuration and SAME initial
    random seed, but run independently so trust updates in TrustMesh don't
    affect Raft's randomness or vice-versa.

    Returns:
        {
            "config": { ... },
            "raft_rounds":      [round_data, ...],
            "trustmesh_rounds": [round_data, ...],
            "raft_nodes":       [Node, ...],
            "trustmesh_nodes":  [Node, ...],
        }
    """
    if seed is not None:
        random.seed(seed)

    correct_answer = "YES"

    # Create two independent copies of nodes
    base_nodes = create_nodes(num_nodes, malicious_pct, faulty_pct)
    raft_nodes = copy.deepcopy(base_nodes)
    tm_nodes = copy.deepcopy(base_nodes)

    raft_rounds = []
    tm_rounds = []

    # Use separate Random objects so each system gets independent randomness
    raft_rng = random.Random(seed)
    tm_rng = random.Random(seed)

    for r in range(num_rounds):
        # Raft round — use raft_rng
        random.seed(raft_rng.randint(0, 2**31))
        raft_data = _run_round(raft_nodes, correct_answer, raft_vote, "Raft")
        raft_data["round"] = r + 1
        raft_rounds.append(raft_data)

        # TrustMesh round — use tm_rng
        random.seed(tm_rng.randint(0, 2**31))
        tm_data = _run_round(tm_nodes, correct_answer, trustmesh_vote, "TrustMesh")
        tm_data["round"] = r + 1
        tm_rounds.append(tm_data)

    return {
        "config": {
            "num_nodes": num_nodes,
            "num_rounds": num_rounds,
            "malicious_pct": malicious_pct,
            "faulty_pct": faulty_pct,
            "seed": seed,
        },
        "raft_rounds": raft_rounds,
        "trustmesh_rounds": tm_rounds,
        "raft_nodes": raft_nodes,
        "trustmesh_nodes": tm_nodes,
    }


# ──────────────────────────────────────────────────────────────
# Pre-defined scenarios
# ──────────────────────────────────────────────────────────────

SCENARIOS = [
    {"name": "All Good (0%)",   "malicious_pct": 0.0, "faulty_pct": 0.0},
    {"name": "10% Malicious",   "malicious_pct": 0.1, "faulty_pct": 0.0},
    {"name": "20% Malicious",   "malicious_pct": 0.2, "faulty_pct": 0.0},
    {"name": "30% Malicious",   "malicious_pct": 0.3, "faulty_pct": 0.0},
    {"name": "40% Malicious",   "malicious_pct": 0.4, "faulty_pct": 0.0},
    {"name": "50% Malicious",   "malicious_pct": 0.5, "faulty_pct": 0.0},
    {"name": "60% Malicious",   "malicious_pct": 0.6, "faulty_pct": 0.0},
    {"name": "80% Malicious",   "malicious_pct": 0.8, "faulty_pct": 0.0},
]


def run_all_scenarios(num_nodes: int = 10, num_rounds: int = 20, seed: int = 42) -> list[dict]:
    """Run every pre-defined scenario and return a list of results."""
    results = []
    for sc in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  Running: {sc['name']}")
        print(f"  Nodes: {num_nodes} | Rounds: {num_rounds}")
        print(f"  Malicious: {sc['malicious_pct']*100:.0f}% | Faulty: {sc['faulty_pct']*100:.0f}%")
        print(f"{'='*60}")

        sim = run_simulation(
            num_nodes=num_nodes,
            num_rounds=num_rounds,
            malicious_pct=sc["malicious_pct"],
            faulty_pct=sc["faulty_pct"],
            seed=seed,
        )
        sim["scenario_name"] = sc["name"]
        results.append(sim)

    return results
