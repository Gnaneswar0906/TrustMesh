"""
graphs.py — Matplotlib visualizations for TrustMesh simulation results.

Generates publication-quality charts and saves them to results/.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────
COLORS = {
    "raft":      "#e74c3c",     # red
    "trustmesh": "#2ecc71",     # green
    "good":      "#3498db",     # blue
    "malicious": "#e74c3c",     # red
    "faulty":    "#f39c12",     # orange
}

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#e0e0e0",
    "axes.labelcolor":  "#e0e0e0",
    "text.color":       "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "grid.color":       "#2a2a4a",
    "grid.alpha":       0.5,
    "font.size":        11,
    "figure.dpi":       150,
})


# ──────────────────────────────────────────────────────────────
# 1. Trust Score Evolution
# ──────────────────────────────────────────────────────────────

def plot_trust_evolution(sim_result: dict, filename: str = "trust_evolution.png"):
    """Line chart showing each node's trust score over rounds."""
    nodes = sim_result["trustmesh_nodes"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for node in nodes:
        color = COLORS.get(node.node_type, "#95a5a6")
        style = "-" if node.node_type == "good" else ("--" if node.node_type == "malicious" else ":")
        ax.plot(node.trust_history, label=f"Node {node.id} ({node.node_type})",
                color=color, linestyle=style, linewidth=2, alpha=0.85)

    ax.set_xlabel("Round")
    ax.set_ylabel("Trust Score")
    ax.set_title("Trust Score Evolution (TrustMesh)", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.5, 10.5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 2. Consensus Accuracy Comparison
# ──────────────────────────────────────────────────────────────

def plot_accuracy_comparison(all_metrics: list[dict], scenario_names: list[str],
                             filename: str = "accuracy_comparison.png"):
    """Grouped bar chart: Raft vs TrustMesh accuracy across scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_names))
    width = 0.35

    raft_vals = [m["Raft"]["consensus_accuracy"] for m in all_metrics]
    tm_vals = [m["TrustMesh"]["consensus_accuracy"] for m in all_metrics]

    bars1 = ax.bar(x - width/2, raft_vals, width, label="Raft", color=COLORS["raft"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, tm_vals, width, label="TrustMesh", color=COLORS["trustmesh"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Consensus Accuracy (%)")
    ax.set_title("Consensus Accuracy — Raft vs TrustMesh", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9, rotation=30, ha='right')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color="#e74c3c")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9, color="#2ecc71")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 3. Latency Comparison
# ──────────────────────────────────────────────────────────────

def plot_latency_comparison(all_metrics: list[dict], scenario_names: list[str],
                            filename: str = "latency_comparison.png"):
    """Bar chart: average latency per round."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_names))
    width = 0.35

    raft_vals = [m["Raft"]["avg_latency_ms"] for m in all_metrics]
    tm_vals = [m["TrustMesh"]["avg_latency_ms"] for m in all_metrics]

    ax.bar(x - width/2, raft_vals, width, label="Raft", color=COLORS["raft"], edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, tm_vals, width, label="TrustMesh", color=COLORS["trustmesh"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Average Latency per Round", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 4. Fault Tolerance Curve
# ──────────────────────────────────────────────────────────────

def plot_fault_tolerance(sweep_data: dict, filename: str = "fault_tolerance.png"):
    """Line chart: accuracy vs % malicious nodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sweep_data["malicious_pcts"], sweep_data["raft_accuracy"],
            "o-", color=COLORS["raft"], linewidth=2.5, markersize=7, label="Raft")
    ax.plot(sweep_data["malicious_pcts"], sweep_data["tm_accuracy"],
            "s-", color=COLORS["trustmesh"], linewidth=2.5, markersize=7, label="TrustMesh")

    ax.set_xlabel("Malicious Nodes (%)")
    ax.set_ylabel("Consensus Accuracy (%)")
    ax.set_title("Fault Tolerance — Accuracy vs Malicious Node %", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=12)
    ax.grid(True)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 5. Message Overhead
# ──────────────────────────────────────────────────────────────

def plot_message_overhead(all_metrics: list[dict], scenario_names: list[str],
                          filename: str = "message_overhead.png"):
    """Grouped bar chart for total messages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_names))
    width = 0.35

    raft_vals = [m["Raft"]["message_overhead"] for m in all_metrics]
    tm_vals = [m["TrustMesh"]["message_overhead"] for m in all_metrics]

    ax.bar(x - width/2, raft_vals, width, label="Raft", color=COLORS["raft"], edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, tm_vals, width, label="TrustMesh", color=COLORS["trustmesh"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Total Messages")
    ax.set_title("Message Overhead Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# 6. Throughput
# ──────────────────────────────────────────────────────────────

def plot_throughput(all_metrics: list[dict], scenario_names: list[str],
                    filename: str = "throughput.png"):
    """Grouped bar chart for throughput (decisions/sec)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenario_names))
    width = 0.35

    raft_vals = [m["Raft"]["throughput_dps"] for m in all_metrics]
    tm_vals = [m["TrustMesh"]["throughput_dps"] for m in all_metrics]

    ax.bar(x - width/2, raft_vals, width, label="Raft", color=COLORS["raft"], edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, tm_vals, width, label="TrustMesh", color=COLORS["trustmesh"], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Decisions per Second")
    ax.set_title("Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis="y")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────
# Master plot function
# ──────────────────────────────────────────────────────────────

def generate_all_graphs(scenario_results: list[dict],
                        all_metrics: list[dict],
                        sweep_data: dict):
    """Generate every chart and save to results/."""
    scenario_names = [r["scenario_name"] for r in scenario_results]

    print("\n📊 Generating graphs...")

    # Trust evolution — use the most adversarial scenario
    worst = max(scenario_results, key=lambda r: r["config"]["malicious_pct"])
    plot_trust_evolution(worst, "trust_evolution.png")

    # Comparison charts
    plot_accuracy_comparison(all_metrics, scenario_names)
    plot_latency_comparison(all_metrics, scenario_names)
    plot_message_overhead(all_metrics, scenario_names)
    plot_throughput(all_metrics, scenario_names)

    # Fault tolerance
    plot_fault_tolerance(sweep_data)

    print("\n✅ All graphs saved to results/\n")
