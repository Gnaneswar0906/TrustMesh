"""
main.py — TrustMesh Interactive Demo.

Interactive CLI that lets the user configure nodes, run Raft vs TrustMesh
voting side-by-side, and see detailed explanations with comparison graphs.
"""

import os
import sys
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Graph Styling ─────────────────────────────────────────────
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
    "font.size":        12,
    "figure.dpi":       150,
})

RAFT_COLOR = "#e74c3c"
TM_COLOR   = "#2ecc71"
GOOD_COLOR = "#3498db"
MAL_COLOR  = "#e74c3c"
FAULTY_COLOR = "#f39c12"


# ══════════════════════════════════════════════════════════════
#  NODE CLASS
# ══════════════════════════════════════════════════════════════

class Node:
    def __init__(self, name, node_type, trust_score=5.0):
        self.name = name
        self.node_type = node_type  # "good", "malicious", "faulty"
        self.trust_score = max(0.0, min(10.0, trust_score))
        self.trust_history = [self.trust_score]

    def vote(self, correct_answer):
        """Cast a vote based on node type."""
        opposite = "NO" if correct_answer == "YES" else "YES"

        if self.node_type == "good":
            return correct_answer
        elif self.node_type == "malicious":
            # 95% chance of voting WRONG
            return opposite if random.random() < 0.95 else correct_answer
        else:  # faulty
            # 50% correct, with chance of no response
            if random.random() < 0.15:  # 15% no response
                return None
            return correct_answer if random.random() < 0.5 else opposite

    def __repr__(self):
        return f"{self.name}({self.node_type}, trust={self.trust_score:.1f})"


# ══════════════════════════════════════════════════════════════
#  VOTING SYSTEMS
# ══════════════════════════════════════════════════════════════

def raft_voting(nodes, correct_answer):
    """Traditional Raft: each node = 1 vote (equal weight)."""
    votes = {}
    for node in nodes:
        v = node.vote(correct_answer)
        votes[node.name] = v

    yes_count = sum(1 for v in votes.values() if v == "YES")
    no_count = sum(1 for v in votes.values() if v == "NO")
    no_response = sum(1 for v in votes.values() if v is None)

    decision = "YES" if yes_count > no_count else "NO"
    is_correct = (decision == correct_answer)

    return {
        "votes": votes,
        "yes_count": yes_count,
        "no_count": no_count,
        "no_response": no_response,
        "decision": decision,
        "is_correct": is_correct,
    }


def trustmesh_voting(nodes, correct_answer):
    """TrustMesh: votes weighted by trust score."""
    votes = {}
    for node in nodes:
        v = node.vote(correct_answer)
        votes[node.name] = v

    yes_trust = 0.0
    no_trust = 0.0
    yes_details = []
    no_details = []
    no_response_list = []

    for node in nodes:
        v = votes[node.name]
        if v == "YES":
            yes_trust += node.trust_score
            yes_details.append((node.name, node.trust_score))
        elif v == "NO":
            no_trust += node.trust_score
            no_details.append((node.name, node.trust_score))
        else:
            no_response_list.append(node.name)

    decision = "YES" if yes_trust >= no_trust else "NO"
    is_correct = (decision == correct_answer)

    return {
        "votes": votes,
        "yes_trust": yes_trust,
        "no_trust": no_trust,
        "yes_details": yes_details,
        "no_details": no_details,
        "no_response": no_response_list,
        "decision": decision,
        "is_correct": is_correct,
    }


# ══════════════════════════════════════════════════════════════
#  TRUST UPDATE
# ══════════════════════════════════════════════════════════════

def update_trust(nodes, votes, correct_answer):
    """Update trust scores based on voting behavior."""
    for node in nodes:
        v = votes.get(node.name)
        if v is None:
            node.trust_score -= 3  # No response penalty
        elif v == correct_answer:
            node.trust_score += 1  # Correct vote reward
        else:
            node.trust_score -= 2  # Wrong vote penalty

        node.trust_score = max(0.0, min(10.0, node.trust_score))
        node.trust_history.append(node.trust_score)


# ══════════════════════════════════════════════════════════════
#  DISPLAY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def print_header(text, char="═"):
    width = 65
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_subheader(text):
    print(f"\n  ── {text} ──")


def print_node_config(nodes):
    """Display node configuration."""
    print_subheader("Node Configuration")
    print(f"  {'Name':<8} {'Type':<12} {'Trust Score':<12} {'Status'}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*20}")
    for node in nodes:
        type_icon = "🟢" if node.node_type == "good" else ("🔴" if node.node_type == "malicious" else "🟡")
        status = ""
        if node.node_type == "good":
            status = "Always votes correctly"
        elif node.node_type == "malicious":
            status = "Votes WRONG 95% of time"
        else:
            status = "Unreliable (50% correct)"
        print(f"  {type_icon} {node.name:<6} {node.node_type:<12} {node.trust_score:<12.1f} {status}")


def explain_raft_round(raft_result, nodes, round_num):
    """Detailed explanation of Raft voting."""
    print_subheader(f"RAFT VOTING (Round {round_num}) — Equal Weight System")
    print(f"  Rule: Every computer gets exactly 1 vote (equal weight)")
    print()

    # Show each node's vote
    print(f"  Voting Results:")
    for node in nodes:
        v = raft_result["votes"][node.name]
        type_tag = f"({node.node_type})"
        if v is None:
            print(f"    {node.name} {type_tag:<14} → ⚠️  NO RESPONSE")
        elif v == "YES":
            print(f"    {node.name} {type_tag:<14} → ✅ YES  (1 vote)")
        else:
            print(f"    {node.name} {type_tag:<14} → ❌ NO   (1 vote)")

    # Tally
    print()
    print(f"  Count:  {raft_result['yes_count']} YES  vs  {raft_result['no_count']} NO", end="")
    if raft_result['no_response'] > 0:
        print(f"  ({raft_result['no_response']} no response)", end="")
    print()

    # Decision
    d = raft_result["decision"]
    icon = "✅" if raft_result["is_correct"] else "❌"
    print(f"  Decision: {d}  {icon} {'CORRECT' if raft_result['is_correct'] else 'WRONG!'}")

    if not raft_result["is_correct"]:
        print(f"  ⚠️  Problem: Bad computers affected the decision!")
        print(f"     Malicious/faulty nodes had equal voting power as good nodes.")


def explain_trustmesh_round(tm_result, nodes, round_num):
    """Detailed explanation of TrustMesh voting."""
    print_subheader(f"TRUSTMESH VOTING (Round {round_num}) — Trust-Weighted System")
    print(f"  Rule: Vote weight = Trust Score (trusted nodes matter MORE)")
    print()

    # Show trust scores
    print(f"  Trust Scores:")
    for node in nodes:
        bar_len = int(node.trust_score)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        label = "⭐ HIGH" if node.trust_score >= 7 else ("⚠️  LOW" if node.trust_score <= 3 else "── MED")
        print(f"    {node.name} = {node.trust_score:>5.1f}  [{bar}]  {label}")

    # Show weighted votes
    print(f"\n  Weighted Voting:")
    for node in nodes:
        v = tm_result["votes"][node.name]
        if v is None:
            print(f"    {node.name} (trust={node.trust_score:.1f}) → ⚠️  NO RESPONSE  (weight = 0)")
        elif v == "YES":
            print(f"    {node.name} (trust={node.trust_score:.1f}) → ✅ YES  (weight = {node.trust_score:.1f})")
        else:
            print(f"    {node.name} (trust={node.trust_score:.1f}) → ❌ NO   (weight = {node.trust_score:.1f})")

    # Show calculation
    print()
    if tm_result["yes_details"]:
        yes_parts = " + ".join(f"{t:.1f}" for _, t in tm_result["yes_details"])
        print(f"  YES total = {yes_parts} = {tm_result['yes_trust']:.1f}")
    else:
        print(f"  YES total = 0.0")

    if tm_result["no_details"]:
        no_parts = " + ".join(f"{t:.1f}" for _, t in tm_result["no_details"])
        print(f"  NO  total = {no_parts} = {tm_result['no_trust']:.1f}")
    else:
        print(f"  NO  total = 0.0")

    if tm_result["no_response"]:
        print(f"  No response: {', '.join(tm_result['no_response'])}")

    # Decision
    print()
    d = tm_result["decision"]
    icon = "✅" if tm_result["is_correct"] else "❌"
    print(f"  Decision: {d}  {icon} {'CORRECT' if tm_result['is_correct'] else 'WRONG!'}")

    if tm_result["is_correct"] and not False:
        print(f"  ✨ TrustMesh works! Trusted nodes have MORE influence.")


def explain_comparison(raft_result, tm_result, round_num):
    """Side-by-side comparison."""
    print_subheader(f"COMPARISON — Round {round_num}")

    raft_icon = "✅ CORRECT" if raft_result["is_correct"] else "❌ WRONG"
    tm_icon = "✅ CORRECT" if tm_result["is_correct"] else "❌ WRONG"

    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │  System       │  Decision  │  Result              │")
    print(f"  ├───────────────┼────────────┼──────────────────────┤")
    print(f"  │  Raft         │  {raft_result['decision']:<8}  │  {raft_icon:<20} │")
    print(f"  │  TrustMesh    │  {tm_result['decision']:<8}  │  {tm_icon:<20} │")
    print(f"  └───────────────┴────────────┴──────────────────────┘")

    if not raft_result["is_correct"] and tm_result["is_correct"]:
        print(f"\n  🏆 TrustMesh WINS this round!")
        print(f"     → Raft failed because bad nodes had equal voting power.")
        print(f"     → TrustMesh succeeded because it weighted votes by trust.")
    elif raft_result["is_correct"] and tm_result["is_correct"]:
        print(f"\n  🤝 Both systems got the correct answer this round.")
    elif not raft_result["is_correct"] and not tm_result["is_correct"]:
        print(f"\n  ⚠️  Both systems failed! Too many malicious nodes.")
    else:
        print(f"\n  📌 Raft got it right, TrustMesh didn't (unusual case).")


def explain_trust_update(nodes, votes, correct_answer):
    """Show how trust scores change."""
    print_subheader("TRUST UPDATE (TrustMesh learns!)")
    print(f"  Rules: Correct vote → +1  |  Wrong vote → -2  |  No response → -3")
    print()

    for node in nodes:
        old_trust = node.trust_history[-2] if len(node.trust_history) >= 2 else node.trust_history[-1]
        new_trust = node.trust_history[-1]
        diff = new_trust - old_trust

        v = votes.get(node.name)
        if v is None:
            reason = "no response → -3"
        elif v == correct_answer:
            reason = "correct vote → +1"
        else:
            reason = "wrong vote → -2"

        arrow = "📈" if diff > 0 else ("📉" if diff < 0 else "━━")
        print(f"    {node.name}: {old_trust:>5.1f} → {new_trust:>5.1f}  ({'+' if diff >= 0 else ''}{diff:.1f})  {arrow}  {reason}")


# ══════════════════════════════════════════════════════════════
#  GRAPH GENERATION
# ══════════════════════════════════════════════════════════════

def generate_comparison_graph(raft_results, tm_results, num_rounds):
    """Generate accuracy comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    rounds = list(range(1, num_rounds + 1))
    raft_correct = [1 if r["is_correct"] else 0 for r in raft_results]
    tm_correct = [1 if r["is_correct"] else 0 for r in tm_results]

    # ── 1. Per-Round Correctness (scatter for clarity) ──
    ax = axes[0][0]
    ax.scatter(rounds, raft_correct, color=RAFT_COLOR, s=80, marker="x",
               label="Raft", zorder=5, linewidths=2)
    ax.scatter(rounds, [c + 0.05 for c in tm_correct], color=TM_COLOR, s=80,
               marker="o", label="TrustMesh", zorder=5, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Correct (1) / Wrong (0)")
    ax.set_title("Per-Round Decision Correctness", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["❌ Wrong", "✅ Correct"])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── 2. Cumulative Accuracy ──
    ax = axes[0][1]
    raft_cum_acc = []
    tm_cum_acc = []
    for i in range(num_rounds):
        r_total = sum(raft_correct[:i+1])
        t_total = sum(tm_correct[:i+1])
        raft_cum_acc.append(r_total / (i+1) * 100)
        tm_cum_acc.append(t_total / (i+1) * 100)

    ax.plot(rounds, raft_cum_acc, "o-", color=RAFT_COLOR, linewidth=2.5, markersize=6, label="Raft")
    ax.plot(rounds, tm_cum_acc, "s-", color=TM_COLOR, linewidth=2.5, markersize=6, label="TrustMesh")
    ax.fill_between(rounds, raft_cum_acc, tm_cum_acc, alpha=0.1, color=TM_COLOR)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Accuracy (%)")
    ax.set_title("Cumulative Accuracy Over Rounds", fontsize=13, fontweight="bold")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add final accuracy labels
    if raft_cum_acc:
        ax.annotate(f"{raft_cum_acc[-1]:.0f}%", xy=(num_rounds, raft_cum_acc[-1]),
                    fontsize=11, fontweight="bold", color=RAFT_COLOR,
                    xytext=(5, 5), textcoords="offset points")
        ax.annotate(f"{tm_cum_acc[-1]:.0f}%", xy=(num_rounds, tm_cum_acc[-1]),
                    fontsize=11, fontweight="bold", color=TM_COLOR,
                    xytext=(5, -15), textcoords="offset points")

    # ── 3. Trust Score Evolution ──
    ax = axes[1][0]
    if hasattr(generate_comparison_graph, '_nodes'):
        nodes = generate_comparison_graph._nodes
        for node in nodes:
            color = GOOD_COLOR if node.node_type == "good" else (MAL_COLOR if node.node_type == "malicious" else FAULTY_COLOR)
            style = "-" if node.node_type == "good" else ("--" if node.node_type == "malicious" else ":")
            ax.plot(node.trust_history, label=f"{node.name} ({node.node_type})",
                    color=color, linestyle=style, linewidth=2, alpha=0.85)

    ax.set_xlabel("Round")
    ax.set_ylabel("Trust Score")
    ax.set_title("Trust Score Evolution (TrustMesh)", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.5, 10.5)
    # Keep legend INSIDE the plot to avoid exploding figure size
    ax.legend(fontsize=7, loc="right", ncol=1)
    ax.grid(True, alpha=0.3)

    # ── 4. Final Accuracy Comparison ──
    ax = axes[1][1]
    raft_total_acc = sum(raft_correct) / num_rounds * 100
    tm_total_acc = sum(tm_correct) / num_rounds * 100

    bars = ax.bar(["Raft\n(Equal Voting)", "TrustMesh\n(Trust-Weighted)"],
                  [raft_total_acc, tm_total_acc],
                  color=[RAFT_COLOR, TM_COLOR],
                  edgecolor="white", linewidth=1.5, width=0.5)

    # Value labels on bars
    for bar, val in zip(bars, [raft_total_acc, tm_total_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=16, fontweight="bold",
                color=RAFT_COLOR if val == raft_total_acc else TM_COLOR)

    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Final Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    # Add improvement label
    if tm_total_acc > raft_total_acc:
        improvement = tm_total_acc - raft_total_acc
        ax.text(0.5, 105, f"TrustMesh is {improvement:.1f}% better!",
                ha="center", fontsize=12, color=TM_COLOR, fontweight="bold",
                transform=ax.get_xaxis_transform())

    fig.suptitle("TrustMesh vs Raft — Performance Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "trustmesh_vs_raft_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  📊 Comparison graph saved: {path}")
    return path


def generate_fault_tolerance_graph(nodes_template, num_rounds, seed):
    """Generate fault tolerance sweep graph."""
    fig, ax = plt.subplots(figsize=(12, 7))

    pcts = list(range(0, 101, 10))
    raft_accs = []
    tm_accs = []

    total_nodes = len(nodes_template)

    for pct in pcts:
        num_mal = int(total_nodes * pct / 100)
        num_good = total_nodes - num_mal

        # Create nodes for this sweep point
        sweep_nodes = []
        for i in range(num_good):
            sweep_nodes.append(Node(f"G{i}", "good", trust_score=8.0))
        for i in range(num_mal):
            sweep_nodes.append(Node(f"M{i}", "malicious", trust_score=2.0))

        random.seed(seed)
        raft_correct = 0
        tm_correct = 0

        for _ in range(num_rounds):
            # Raft
            rv = raft_voting(sweep_nodes, "YES")
            if rv["is_correct"]:
                raft_correct += 1

            # TrustMesh
            tv = trustmesh_voting(sweep_nodes, "YES")
            if tv["is_correct"]:
                tm_correct += 1

            # Update trust for next round
            update_trust(sweep_nodes, tv["votes"], "YES")

        raft_accs.append(raft_correct / num_rounds * 100)
        tm_accs.append(tm_correct / num_rounds * 100)

        # Reset trust for next sweep point
        for n in sweep_nodes:
            n.trust_score = 5.0
            n.trust_history = [5.0]

    ax.plot(pcts, raft_accs, "o-", color=RAFT_COLOR, linewidth=3, markersize=8, label="Raft (Equal Voting)")
    ax.plot(pcts, tm_accs, "s-", color=TM_COLOR, linewidth=3, markersize=8, label="TrustMesh (Trust-Weighted)")

    # Fill area between
    ax.fill_between(pcts, raft_accs, tm_accs, alpha=0.15, color=TM_COLOR)

    ax.set_xlabel("Malicious Nodes (%)", fontsize=12)
    ax.set_ylabel("Consensus Accuracy (%)", fontsize=12)
    ax.set_title("Fault Tolerance — Accuracy vs Malicious Node %", fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 115)
    ax.set_xlim(-2, 102)
    ax.legend(fontsize=12, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Annotate key point
    for i, pct in enumerate(pcts):
        if raft_accs[i] < 80 and tm_accs[i] > 80:
            ax.annotate(f"TrustMesh advantage\nstarts here!",
                        xy=(pct, tm_accs[i]), fontsize=10, color=TM_COLOR,
                        fontweight="bold",
                        xytext=(pct + 10, tm_accs[i] - 20),
                        arrowprops=dict(arrowstyle="->", color=TM_COLOR))
            break

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "fault_tolerance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Fault tolerance graph saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════
#  INPUT FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_user_nodes():
    """Get node configuration from user."""
    print_header("🔷 TRUSTMESH — Raft vs Trust-Weighted Consensus")

    print("""
  This demo shows WHY TrustMesh is better than traditional Raft.

  In Raft:   Every node gets EQUAL voting power (1 vote each)
  In TrustMesh: Trusted nodes get MORE voting power (weighted by trust)

  Node types:
    🟢 Good      — Always votes correctly
    🔴 Malicious — Votes WRONG 95% of the time (bad actor!)
    🟡 Faulty    — Unreliable (50% correct, may not respond)
    """)

    # Ask for input mode
    print("  Choose setup mode:")
    print("    1. Quick setup (preset scenarios)")
    print("    2. Custom setup (configure each node)")
    print()

    choice = input("  Enter choice (1 or 2): ").strip()

    if choice == "2":
        return get_custom_nodes()
    else:
        return get_preset_nodes()


def get_preset_nodes():
    """Preset scenario selection."""
    print()
    print("  Preset Scenarios:")
    print("    1. 5 nodes: 3 Good + 2 Malicious  (Raft struggles)")
    print("    2. 7 nodes: 3 Good + 4 Malicious  (Raft breaks, TrustMesh wins)")
    print("    3. 5 nodes: 2 Good + 2 Malicious + 1 Faulty")
    print("    4. 10 nodes: 4 Good + 6 Malicious (extreme adversary)")
    print("    5. Example from presentation (A,B,C,D,E)")
    print()

    sc = input("  Choose scenario (1-5): ").strip()

    if sc == "1":
        return [
            Node("A", "good", trust_score=9.0),
            Node("B", "good", trust_score=8.0),
            Node("C", "good", trust_score=7.0),
            Node("D", "malicious", trust_score=3.0),
            Node("E", "malicious", trust_score=2.0),
        ]
    elif sc == "2":
        return [
            Node("A", "good", trust_score=10.0),
            Node("B", "good", trust_score=9.0),
            Node("C", "good", trust_score=8.0),
            Node("D", "malicious", trust_score=3.0),
            Node("E", "malicious", trust_score=2.0),
            Node("F", "malicious", trust_score=1.5),
            Node("G", "malicious", trust_score=1.0),
        ]
    elif sc == "3":
        return [
            Node("A", "good", trust_score=9.0),
            Node("B", "good", trust_score=8.0),
            Node("C", "malicious", trust_score=2.0),
            Node("D", "malicious", trust_score=1.5),
            Node("E", "faulty", trust_score=4.0),
        ]
    elif sc == "4":
        return [
            Node("A", "good", trust_score=10.0),
            Node("B", "good", trust_score=9.5),
            Node("C", "good", trust_score=9.0),
            Node("D", "good", trust_score=8.5),
            Node("E", "malicious", trust_score=2.0),
            Node("F", "malicious", trust_score=1.5),
            Node("G", "malicious", trust_score=1.0),
            Node("H", "malicious", trust_score=0.5),
            Node("I", "malicious", trust_score=1.0),
            Node("J", "malicious", trust_score=0.5),
        ]
    else:  # scenario 5 - presentation example
        return [
            Node("A", "good", trust_score=10.0),
            Node("B", "good", trust_score=9.0),
            Node("C", "good", trust_score=8.0),
            Node("D", "malicious", trust_score=2.0),
            Node("E", "malicious", trust_score=1.0),
        ]


def get_custom_nodes():
    """Custom node configuration from user."""
    print()
    num = int(input("  How many nodes? ").strip())
    nodes = []

    names = [chr(65 + i) for i in range(min(num, 26))]  # A, B, C, ...
    if num > 26:
        names = [f"N{i}" for i in range(num)]

    for i in range(num):
        name = names[i]
        print(f"\n  Node {name}:")
        print(f"    Type options: good, malicious, faulty")
        ntype = input(f"    Type for {name}: ").strip().lower()
        if ntype not in ("good", "malicious", "faulty"):
            print(f"    Invalid type, defaulting to 'good'")
            ntype = "good"

        trust = input(f"    Trust score for {name} (0-10, default=5): ").strip()
        trust = float(trust) if trust else 5.0
        trust = max(0.0, min(10.0, trust))

        nodes.append(Node(name, ntype, trust))

    return nodes


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    nodes = get_user_nodes()

    print()
    num_rounds = input("  Number of rounds to simulate (default=10): ").strip()
    num_rounds = int(num_rounds) if num_rounds else 10

    seed_input = input("  Random seed (default=42): ").strip()
    seed = int(seed_input) if seed_input else 42
    random.seed(seed)

    correct_answer = "YES"  # The "correct" answer for all rounds

    # Show configuration
    print_node_config(nodes)

    # ── Run Simulation ──
    print_header("🏁 SIMULATION — Running Raft vs TrustMesh", "─")

    # Create separate copies for each system (so trust updates don't cross-contaminate)
    import copy
    raft_nodes = copy.deepcopy(nodes)
    tm_nodes = copy.deepcopy(nodes)

    raft_results = []
    tm_results = []

    for r in range(1, num_rounds + 1):
        random.seed(seed * 1000 + r * 7)

        # Run both systems
        raft_result = raft_voting(raft_nodes, correct_answer)
        raft_results.append(raft_result)

        random.seed(seed * 1000 + r * 7)  # Same seed for fair comparison
        tm_result = trustmesh_voting(tm_nodes, correct_answer)
        tm_results.append(tm_result)

        # Detailed output for first few rounds and interesting rounds
        show_detail = (r <= 3) or (r == num_rounds) or \
                      (raft_result["is_correct"] != tm_result["is_correct"])

        if show_detail:
            print_header(f"ROUND {r} of {num_rounds}")

            explain_raft_round(raft_result, raft_nodes, r)
            explain_trustmesh_round(tm_result, tm_nodes, r)
            explain_comparison(raft_result, tm_result, r)

            # Update trust (TrustMesh only)
            update_trust(tm_nodes, tm_result["votes"], correct_answer)
            explain_trust_update(tm_nodes, tm_result["votes"], correct_answer)
        else:
            # Still update trust, just don't print details
            update_trust(tm_nodes, tm_result["votes"], correct_answer)

            # Print summary line
            raft_icon = "✅" if raft_result["is_correct"] else "❌"
            tm_icon = "✅" if tm_result["is_correct"] else "❌"
            print(f"  Round {r:>2}: Raft={raft_result['decision']} {raft_icon}  |  TrustMesh={tm_result['decision']} {tm_icon}")

    # ── Final Summary ──
    raft_correct_total = sum(1 for r in raft_results if r["is_correct"])
    tm_correct_total = sum(1 for r in tm_results if r["is_correct"])

    print_header("📊 FINAL RESULTS")

    raft_acc = raft_correct_total / num_rounds * 100
    tm_acc = tm_correct_total / num_rounds * 100

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                    FINAL ACCURACY REPORT                    │
  ├────────────────────┬───────────────┬────────────────────────┤
  │  System            │  Correct/Total│  Accuracy              │
  ├────────────────────┼───────────────┼────────────────────────┤
  │  Raft (Equal)      │  {raft_correct_total:>4}/{num_rounds:<4}     │  {raft_acc:>6.1f}%                 │
  │  TrustMesh (Trust) │  {tm_correct_total:>4}/{num_rounds:<4}     │  {tm_acc:>6.1f}%                 │
  └────────────────────┴───────────────┴────────────────────────┘""")

    if tm_acc > raft_acc:
        print(f"""
  🏆 TrustMesh outperformed Raft by {tm_acc - raft_acc:.1f}%!

  WHY TrustMesh is better:
  ─────────────────────────
  ✅ Adaptive:        Trust scores evolve — bad actors get marginalized
  ✅ Weighted voting:  High-trust nodes dominate decisions
  ✅ Stable leadership: Leaders elected based on trust, not just majority
  ✅ Self-healing:     System learns from node behavior over time

  WHY Raft fails:
  ──────────────────
  ❌ Equal voting:    Malicious nodes get same weight as honest ones
  ❌ No memory:       System doesn't learn from past bad behavior
  ❌ Majority attack: >50% malicious nodes can take over decisions
""")
    elif tm_acc == raft_acc:
        print(f"\n  🤝 Both systems performed equally.")
        print(f"     TrustMesh advantage appears with MORE malicious nodes.")
    else:
        print(f"\n  📌 Unusual result — try increasing rounds or malicious nodes.")

    # ── Final trust scores ──
    print_subheader("Final Trust Scores (TrustMesh)")
    for node in tm_nodes:
        bar_len = int(node.trust_score)
        bar = "█" * bar_len + "░" * (10 - bar_len)
        change = node.trust_score - node.trust_history[0]
        arrow = "📈" if change > 0 else ("📉" if change < 0 else "━━")
        print(f"    {node.name} ({node.node_type:<10}): {node.trust_score:>5.1f}  [{bar}]  {arrow} {'+' if change >= 0 else ''}{change:.1f} from start")

    # ── Generate Graphs ──
    print_header("📈 GENERATING COMPARISON GRAPHS")
    generate_comparison_graph._nodes = tm_nodes
    generate_comparison_graph(raft_results, tm_results, num_rounds)
    generate_fault_tolerance_graph(nodes, num_rounds, seed)

    print(f"\n  All graphs saved to: {RESULTS_DIR}")
    print(f"\n{'═' * 65}")
    print(f"  ✅ DONE — Open 'results/' folder to view graphs!")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
