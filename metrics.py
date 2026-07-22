"""
metrics.py — Performance metrics for Raft vs TrustMesh.

Six metrics:
    1. Consensus Accuracy
    2. Fault Tolerance (accuracy vs % malicious)
    3. Latency (avg ms per round)
    4. Leader Stability (# of leader changes)
    5. Message Overhead (total messages)
    6. Throughput (decisions per second)
"""

from simulation import run_simulation


# ──────────────────────────────────────────────────────────────
# Per-simulation metric extraction
# ──────────────────────────────────────────────────────────────

def compute_metrics(sim_result: dict) -> dict:
    """
    Given a simulation result dict, compute all six metrics for both systems.

    Returns:
        {
            "Raft":      { metric_name: value, ... },
            "TrustMesh": { metric_name: value, ... },
        }
    """
    out = {}

    for system, rounds_key in [("Raft", "raft_rounds"), ("TrustMesh", "trustmesh_rounds")]:
        rounds = sim_result[rounds_key]
        n = len(rounds)

        correct = sum(1 for r in rounds if r["is_correct"])
        total_latency = sum(r["latency_ms"] for r in rounds)
        leader_changes = sum(1 for r in rounds if r["leader_changed"])
        total_messages = sum(r["message_count"] for r in rounds)
        total_time_s = total_latency / 1000.0 if total_latency > 0 else 1e-6

        out[system] = {
            "consensus_accuracy": round(correct / n * 100, 2),
            "avg_latency_ms": round(total_latency / n, 4),
            "leader_changes": leader_changes,
            "message_overhead": total_messages,
            "throughput_dps": round(n / total_time_s, 2),
        }

    return out


# ──────────────────────────────────────────────────────────────
# Fault-tolerance sweep
# ──────────────────────────────────────────────────────────────

def fault_tolerance_sweep(num_nodes: int = 10,
                          num_rounds: int = 20,
                          seed: int = 42,
                          steps: int = 11) -> dict:
    """
    Sweep malicious_pct from 0% to 100% and record consensus accuracy
    for both systems at each step.

    Returns:
        {
            "malicious_pcts": [0, 10, 20, ...],
            "raft_accuracy":  [100.0, 95.0, ...],
            "tm_accuracy":    [100.0, 100.0, ...],
        }
    """
    pcts = []
    raft_acc = []
    tm_acc = []

    for i in range(steps):
        pct = i / (steps - 1)
        sim = run_simulation(num_nodes, num_rounds, malicious_pct=pct, seed=seed)
        m = compute_metrics(sim)

        pcts.append(round(pct * 100))
        raft_acc.append(m["Raft"]["consensus_accuracy"])
        tm_acc.append(m["TrustMesh"]["consensus_accuracy"])

        print(f"  Malicious {pct*100:5.1f}%  →  Raft {m['Raft']['consensus_accuracy']:6.1f}%"
              f"  |  TrustMesh {m['TrustMesh']['consensus_accuracy']:6.1f}%")

    return {
        "malicious_pcts": pcts,
        "raft_accuracy": raft_acc,
        "tm_accuracy": tm_acc,
    }


# ──────────────────────────────────────────────────────────────
# Pretty-print helper
# ──────────────────────────────────────────────────────────────

def print_metrics(scenario_name: str, metrics: dict):
    """Print a nicely formatted metrics table."""
    print(f"\n┌{'─'*62}┐")
    print(f"│  {scenario_name:^58}  │")
    print(f"├{'─'*30}┬{'─'*15}┬{'─'*15}┤")
    print(f"│ {'Metric':<28} │ {'Raft':^13} │ {'TrustMesh':^13} │")
    print(f"├{'─'*30}┼{'─'*15}┼{'─'*15}┤")

    labels = [
        ("Consensus Accuracy (%)", "consensus_accuracy"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("Leader Changes", "leader_changes"),
        ("Message Overhead", "message_overhead"),
        ("Throughput (dec/s)", "throughput_dps"),
    ]

    for label, key in labels:
        rv = metrics["Raft"][key]
        tv = metrics["TrustMesh"][key]
        print(f"│ {label:<28} │ {rv:>13} │ {tv:>13} │")

    print(f"└{'─'*30}┴{'─'*15}┴{'─'*15}┘")
