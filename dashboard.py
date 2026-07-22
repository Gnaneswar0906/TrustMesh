"""
dashboard.py — TrustMesh: Trust-Weighted Consensus in Distributed Systems.

A professional interactive dashboard demonstrating how trust-based
consensus outperforms traditional majority-based (Raft) consensus
in the presence of Byzantine (malicious) and crash (faulty) faults.

Distributed System Concepts:
  - Byzantine Fault Tolerance (BFT)
  - Consensus Protocols (Raft vs Trust-Weighted)
  - Trust/Reputation Systems
  - Leader Election
  - Message Passing & Overhead
  - Fault Tolerance Analysis

Run:   python dashboard.py
Visit: http://localhost:5000
"""

import sys
import json
import random
import math
import os

from flask import Flask, render_template_string, jsonify, request, send_from_directory

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

app = Flask(__name__)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ══════════════════════════════════════════════════════════════
#  DISTRIBUTED SYSTEM CONSTANTS
# ══════════════════════════════════════════════════════════════

TRUST_ALPHA = 7             # EMA smoothing factor as integer (used as alpha/10 = 0.7)
SIGMOID_K = 1.8            # Sigmoid steepness for trust→weight conversion
SIGMOID_THRESHOLD = 5      # Sigmoid midpoint (trust score where weight = 50)
NUM_SIM_ROUNDS = 20        # History simulation rounds for trust building
INITIAL_TRUST = 5          # Default starting trust (integer 0-10)


# ══════════════════════════════════════════════════════════════
#  CORE ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════

def sigmoid_weight(trust_score):
    """
    Convert a raw trust score (0–10) into a voting weight (0–100)
    using a sigmoid function. Returns integer weight.

    Why sigmoid?
    - Provides smooth, non-linear transition (no abrupt cutoffs)
    - Nodes near threshold get moderate weight (~50)
    - Very trusted nodes → weight ~100
    - Very untrusted nodes → weight ~0
    - Matches real reputation systems (e.g., PageRank diminishing returns)
    """
    raw = 1.0 / (1.0 + math.exp(-SIGMOID_K * (trust_score - SIGMOID_THRESHOLD)))
    return int(round(raw * 100))


def compute_round_score(is_correct, response_time_ms, was_slow, responded):
    """
    Compute a nuanced round score (0–10) based on multiple factors,
    NOT just a binary correct/incorrect.

    Factors:
      1. Vote correctness (primary factor)
      2. Response latency (secondary factor)
      3. Whether node responded at all
    """
    if not responded:
        # Node failed to respond — crash/network fault
        return int(round(random.uniform(1.5, 3.0)))

    if is_correct:
        # Correct vote — base score 7-10 depending on speed
        base = 8.5
        if was_slow:
            # Correct but slow — moderate score (network delay)
            return int(round(random.uniform(5.5, 7.5)))
        else:
            # Fast and correct — high score
            latency_bonus = max(0, 1.5 - (response_time_ms / 10.0))
            return int(round(min(10.0, base + latency_bonus + random.uniform(-0.5, 0.5))))
    else:
        # Wrong vote — low score, but not always zero
        # Could be due to network partition, stale data, or malicious intent
        return int(round(random.uniform(0.5, 2.8)))


def simulate_node_behavior(node_type, correct_answer, rng):
    """
    Simulate a node's behavior in a single round based on its type.
    Returns (vote, responded, was_slow, response_time_ms).
    """
    opposite = "NO" if correct_answer == "YES" else "YES"

    if node_type == "good":
        # Good nodes: always vote correctly, fast response
        response_time = rng.uniform(1.0, 8.0)
        return correct_answer, True, False, response_time

    elif node_type == "malicious":
        # Malicious nodes: intentionally vote wrong most of the time
        # But not ALWAYS — smart attackers sometimes vote correctly to
        # maintain some trust (Sybil-like behavior)
        if rng.random() < 0.85:
            vote = opposite
        else:
            vote = correct_answer  # Camouflage behavior
        response_time = rng.uniform(2.0, 15.0)
        was_slow = rng.random() < 0.1
        return vote, True, was_slow, response_time

    else:  # faulty
        # Faulty nodes: unreliable — may not respond, may be slow, may be wrong
        if rng.random() < 0.15:
            return None, False, False, 0.0  # Crash fault — no response
        was_slow = rng.random() < 0.25
        response_time = rng.uniform(3.0, 50.0) if was_slow else rng.uniform(2.0, 12.0)
        vote = correct_answer if rng.random() < 0.55 else opposite
        return vote, True, was_slow, response_time


def analyze_votes(nodes_config, correct_answer="YES", num_rounds=None):
    # Use user-provided rounds or fall back to default constant
    num_rounds = num_rounds if num_rounds is not None else NUM_SIM_ROUNDS
    """
    Given user-configured nodes, run full distributed consensus analysis.

    Args:
        nodes_config: [{"name": "A", "type": "good", "trust": 5.0}, ...]
        correct_answer: the ground-truth value

    Returns:
        Complete analysis with Raft result, TrustMesh scoring,
        BFT analysis, evaluation metrics, and explanations.
    """
    n = len(nodes_config)
    rng = random.Random(42)

    # ── Initialize nodes ──────────────────────────────────
    nodes = []
    for nc in nodes_config:
        nodes.append({
            "name": nc["name"],
            "type": nc["type"],
            "trust": int(nc.get("trust", INITIAL_TRUST)),
            "trust_history": [int(nc.get("trust", INITIAL_TRUST))],
            "round_scores": [],
            "weights": [],
            "votes": [],
            "correct_count": 0,
            "wrong_count": 0,
            "timeout_count": 0,
            "total_latency": 0.0,
            "classification": "unknown",
        })

    # ── STEP 1: Raft Analysis (simple majority) ───────────
    raft_votes = {}
    raft_latencies = {}
    raft_messages = 0

    for node in nodes:
        vote, responded, was_slow, resp_time = simulate_node_behavior(
            node["type"], correct_answer, rng
        )
        raft_votes[node["name"]] = vote
        raft_latencies[node["name"]] = resp_time
        raft_messages += 2 if responded else 1  # request + response (or just request)

    raft_yes = sum(1 for v in raft_votes.values() if v == "YES")
    raft_no = sum(1 for v in raft_votes.values() if v == "NO")
    raft_timeout = sum(1 for v in raft_votes.values() if v is None)
    raft_decision = "YES" if raft_yes >= raft_no else "NO"
    raft_correct = (raft_decision == correct_answer)

    raft_analysis = {
        "decision": raft_decision,
        "is_correct": raft_correct,
        "yes_count": raft_yes,
        "no_count": raft_no,
        "timeout_count": raft_timeout,
        "votes": raft_votes,
        "latencies": raft_latencies,
        "messages": raft_messages,
        "avg_latency": round(sum(raft_latencies.values()) / max(1, len(raft_latencies)), 2),
    }

    # ── STEP 2: TrustMesh — Multi-round trust building ───
    history_log = []
    tm_messages = 0
    leader_changes = 0
    current_leader = None

    for rnd in range(1, num_rounds + 1):
        round_rng = random.Random(42 * 1000 + rnd * 7)
        round_data = {"round": rnd, "node_data": {}}

        for node in nodes:
            vote, responded, was_slow, resp_time = simulate_node_behavior(
                node["type"], correct_answer, round_rng
            )

            is_correct = (vote == correct_answer) if vote is not None else False

            # Compute nuanced round score
            round_score = compute_round_score(is_correct, resp_time, was_slow, responded)
            node["round_scores"].append(round_score)

            # Update trust using Exponential Moving Average (EMA)
            # alpha is stored as integer (7), used as 7/10 = 0.7
            old_trust = node["trust"]
            alpha_frac = TRUST_ALPHA / 10.0
            new_trust = alpha_frac * old_trust + (1 - alpha_frac) * round_score
            node["trust"] = int(round(max(0, min(10, new_trust))))
            node["trust_history"].append(node["trust"])

            # Track statistics
            if not responded:
                node["timeout_count"] += 1
            elif is_correct:
                node["correct_count"] += 1
            else:
                node["wrong_count"] += 1

            node["total_latency"] += resp_time
            node["votes"].append(vote)

            # Compute weight from current trust
            w = sigmoid_weight(node["trust"])
            node["weights"].append(w)

            tm_messages += 2 if responded else 1

            round_data["node_data"][node["name"]] = {
                "vote": vote,
                "round_score": round_score,
                "trust": node["trust"],
                "weight": round(w, 4),
                "responded": responded,
                "was_slow": was_slow,
            }

        # Leader election — highest trust, tie-break by latest round score
        new_leader = max(nodes, key=lambda nd: (nd["trust"], nd["round_scores"][-1] if nd["round_scores"] else 0))["name"]
        if new_leader != current_leader:
            leader_changes += 1
            current_leader = new_leader

        round_data["leader"] = current_leader
        history_log.append(round_data)

    # ── STEP 3: Classify nodes based on final trust ──────
    for node in nodes:
        t = node["trust"]
        accuracy = node["correct_count"] / max(1, num_rounds) * 100

        if t >= 7:
            node["classification"] = "good"
            node["class_reason"] = (
                f"High trust ({t}/10). Voted correctly in {node['correct_count']}/{num_rounds} rounds "
                f"with {accuracy:.0f}% accuracy. Consistent, reliable behavior indicates a trustworthy node."
            )
        elif t >= 4:
            node["classification"] = "faulty"
            node["class_reason"] = (
                f"Medium trust ({t}/10). Mixed behavior — {node['correct_count']} correct, "
                f"{node['wrong_count']} wrong, {node['timeout_count']} timeouts out of {num_rounds} rounds. "
                f"Likely experiencing network issues or intermittent faults."
            )
        else:
            node["classification"] = "malicious"
            node["class_reason"] = (
                f"Low trust ({t}/10). Voted incorrectly in {node['wrong_count']}/{num_rounds} rounds. "
                f"Consistent adversarial behavior — likely a Byzantine node attempting to corrupt consensus."
            )

    # ── STEP 4: TrustMesh Weighted Voting ────────────────
    # Final vote using trust-weighted consensus
    yes_weighted = 0.0
    no_weighted = 0.0
    yes_parts = []
    no_parts = []
    timeout_parts = []

    final_rng = random.Random(42 * 2000)
    for node in nodes:
        vote, responded, was_slow, resp_time = simulate_node_behavior(
            node["type"], correct_answer, final_rng
        )
        weight = sigmoid_weight(node["trust"])
        node["final_vote"] = vote
        node["final_weight"] = weight

        if vote == "YES":
            yes_weighted += weight
            yes_parts.append({"name": node["name"], "trust": node["trust"], "weight": weight})
        elif vote == "NO":
            no_weighted += weight
            no_parts.append({"name": node["name"], "trust": node["trust"], "weight": weight})
        else:
            timeout_parts.append({"name": node["name"], "trust": node["trust"]})

    tm_decision = "YES" if yes_weighted >= no_weighted else "NO"
    tm_correct = (tm_decision == correct_answer)

    tm_analysis = {
        "decision": tm_decision,
        "is_correct": tm_correct,
        "yes_weighted": int(round(yes_weighted)),
        "no_weighted": int(round(no_weighted)),
        "yes_parts": yes_parts,
        "no_parts": no_parts,
        "timeout_parts": timeout_parts,
        "messages": tm_messages,
        "avg_latency": round(sum(nd["total_latency"] for nd in nodes) / max(1, n * num_rounds), 2),
        "leader_changes": leader_changes,
        "final_leader": current_leader,
    }

    # ── STEP 5: BFT Analysis ────────────────────────────
    # Use USER-CONFIGURED type for fault counting, not system-detected classification
    num_malicious = sum(1 for nd in nodes if nd["type"] == "malicious")
    num_faulty_nodes = sum(1 for nd in nodes if nd["type"] == "faulty")
    # Also track detected counts for display
    det_malicious = sum(1 for nd in nodes if nd["classification"] == "malicious")
    det_faulty = sum(1 for nd in nodes if nd["classification"] == "faulty")
    bft_threshold = (n - 1) // 3  # max faults tolerable: f in 3f+1
    total_faults = num_malicious + num_faulty_nodes
    bft_safe = total_faults <= bft_threshold

    bft_analysis = {
        "total_nodes": n,
        "byzantine_count": num_malicious,
        "crash_fault_count": num_faulty_nodes,
        "total_faults": total_faults,
        "detected_byzantine": det_malicious,
        "detected_faulty": det_faulty,
        "bft_threshold": bft_threshold,
        "bft_formula": f"f ≤ (n-1)/3 = ({n}-1)/3 = {bft_threshold}",
        "is_safe": bft_safe,
        "raft_threshold": n // 2,  # Raft needs simple majority
        "raft_formula": f"n/2 = {n}/2 = {n // 2}",
        "good_count": sum(1 for nd in nodes if nd["type"] == "good"),
    }

    # ── STEP 6: Evaluation Metrics ──────────────────────

    # Compute TrustMesh multi-round accuracy
    tm_round_correct = 0
    for rnd_data in history_log:
        rnd_yes_w = 0
        rnd_no_w = 0
        for name, nd_data in rnd_data["node_data"].items():
            if nd_data["vote"] == "YES":
                rnd_yes_w += nd_data["weight"]
            elif nd_data["vote"] == "NO":
                rnd_no_w += nd_data["weight"]
        rnd_decision = "YES" if rnd_yes_w >= rnd_no_w else "NO"
        if rnd_decision == correct_answer:
            tm_round_correct += 1

    # Raft multi-round accuracy (simulate)
    raft_round_correct = 0
    for rnd in range(1, num_rounds + 1):
        rr = random.Random(42 * 1000 + rnd * 7)
        r_yes = 0
        r_no = 0
        for node in nodes:
            v, resp, _, _ = simulate_node_behavior(node["type"], correct_answer, rr)
            if v == "YES":
                r_yes += 1
            elif v == "NO":
                r_no += 1
        if ("YES" if r_yes >= r_no else "NO") == correct_answer:
            raft_round_correct += 1

    raft_accuracy_pct = round(raft_round_correct / num_rounds * 100, 1)
    tm_accuracy_pct = round(tm_round_correct / num_rounds * 100, 1)

    # Effective throughput = raw decisions/sec × accuracy rate
    # A wrong decision is wasted work, so only correct decisions count
    raft_raw_throughput = 1000.0 / max(0.001, raft_analysis["avg_latency"])
    tm_raw_throughput = 1000.0 / max(0.001, tm_analysis["avg_latency"])
    raft_throughput = round(raft_raw_throughput * (raft_accuracy_pct / 100.0), 2)
    tm_throughput = round(tm_raw_throughput * (tm_accuracy_pct / 100.0), 2)

    eval_metrics = {
        "consensus_accuracy": {
            "raft": raft_accuracy_pct,
            "trustmesh": tm_accuracy_pct,
            "description": "Percentage of rounds where the system reached the correct consensus decision.",
        },
        "fault_tolerance": {
            "raft": f"{n // 2} of {n} nodes (simple majority)",
            "trustmesh": f"Dynamic — adapts via trust. Currently handling {total_faults} faults",
            "description": "Maximum number of faulty/malicious nodes the system can tolerate.",
        },
        "avg_latency": {
            "raft": f"{raft_analysis['avg_latency']:.2f} ms",
            "trustmesh": f"{tm_analysis['avg_latency']:.2f} ms",
            "description": "Average time per consensus round (request to decision).",
        },
        "leader_stability": {
            "raft": "N/A (random election)",
            "trustmesh": f"{leader_changes} changes in {num_rounds} rounds",
            "description": "Number of leader changes. Fewer changes = more stable system.",
        },
        "message_overhead": {
            "raft": f"{raft_analysis['messages']} messages",
            "trustmesh": f"{tm_analysis['messages']} messages",
            "description": "Total network messages exchanged (request + response per node per round).",
        },
        "throughput": {
            "raft": f"{raft_throughput} correct decisions/sec",
            "trustmesh": f"{tm_throughput} correct decisions/sec",
            "description": "Effective throughput — only correct consensus decisions count as useful work.",
        },
    }

    # ── STEP 7: Build node details ──────────────────────
    node_details = []
    for node in nodes:
        node_details.append({
            "name": node["name"],
            "type": node["type"],
            "final_vote": node.get("final_vote"),
            "trust": node["trust"],
            "initial_trust": node["trust_history"][0],
            "classification": node["classification"],
            "class_reason": node["class_reason"],
            "trust_history": node["trust_history"],
            "round_scores": node["round_scores"],
            "weights": node["weights"],
            "final_weight": node.get("final_weight", 0),
            "correct_count": node["correct_count"],
            "wrong_count": node["wrong_count"],
            "timeout_count": node["timeout_count"],
            "avg_latency": round(node["total_latency"] / max(1, num_rounds), 2),
        })

    # ── STEP 8: Build explanation ────────────────────────
    explanation = []
    if not raft_correct and tm_correct:
        explanation.append("TrustMesh reaches the CORRECT consensus while Raft fails.")
        explanation.append("Raft failed because Byzantine nodes had equal voting power — one node, one vote.")
        explanation.append("TrustMesh succeeded by learning trust profiles over time and weighting votes via sigmoid function.")
    elif raft_correct and tm_correct:
        explanation.append("Both systems reach the correct consensus.")
        explanation.append("The honest majority is strong enough for even equal-weight voting to work.")
        explanation.append("TrustMesh still provides value: it identifies which nodes to trust for future rounds.")
    elif not raft_correct and not tm_correct:
        explanation.append("Both systems failed — the adversarial majority is too strong.")
        explanation.append("When Byzantine nodes exceed the BFT threshold, no consensus protocol can guarantee correctness.")
    else:
        explanation.append("Raft succeeded while TrustMesh didn't — a rare edge case due to trust evolution timing.")

    # ── Sigmoid curve data for visualization ─────────────
    sigmoid_data = []
    for i in range(11):
        sigmoid_data.append({"trust": i, "weight": sigmoid_weight(i)})

    # ── STEP 9: Per-round Raft vs TrustMesh comparison ───
    round_comparison = []
    raft_cumulative_correct = 0
    tm_cumulative_correct = 0
    for rnd_idx, rnd_data in enumerate(history_log):
        rnd_num = rnd_idx + 1
        # TrustMesh decision for this round
        rnd_yes_w = 0
        rnd_no_w = 0
        for name, nd_data in rnd_data["node_data"].items():
            if nd_data["vote"] == "YES":
                rnd_yes_w += nd_data["weight"]
            elif nd_data["vote"] == "NO":
                rnd_no_w += nd_data["weight"]
        tm_rnd_decision = "YES" if rnd_yes_w >= rnd_no_w else "NO"
        tm_rnd_correct = (tm_rnd_decision == correct_answer)
        if tm_rnd_correct:
            tm_cumulative_correct += 1

        # Raft decision for this round (same RNG as used in trust building)
        rr = random.Random(42 * 1000 + rnd_num * 7)
        r_yes = 0
        r_no = 0
        for node in nodes:
            v, resp, _, _ = simulate_node_behavior(node["type"], correct_answer, rr)
            if v == "YES":
                r_yes += 1
            elif v == "NO":
                r_no += 1
        raft_rnd_decision = "YES" if r_yes >= r_no else "NO"
        raft_rnd_correct = (raft_rnd_decision == correct_answer)
        if raft_rnd_correct:
            raft_cumulative_correct += 1

        round_comparison.append({
            "round": rnd_num,
            "raft_decision": raft_rnd_decision,
            "raft_correct": raft_rnd_correct,
            "raft_yes": r_yes,
            "raft_no": r_no,
            "raft_accuracy": int(round(raft_cumulative_correct / rnd_num * 100)),
            "tm_decision": tm_rnd_decision,
            "tm_correct": tm_rnd_correct,
            "tm_yes_w": int(round(rnd_yes_w)),
            "tm_no_w": int(round(rnd_no_w)),
            "tm_accuracy": int(round(tm_cumulative_correct / rnd_num * 100)),
        })

    return {
        "correct_answer": correct_answer,
        "num_nodes": n,
        "num_sim_rounds": num_rounds,
        "raft": raft_analysis,
        "trustmesh": tm_analysis,
        "nodes": node_details,
        "history_log": history_log,
        "bft": bft_analysis,
        "eval_metrics": eval_metrics,
        "explanation": explanation,
        "sigmoid_data": sigmoid_data,
        "sigmoid_params": {"k": SIGMOID_K, "threshold": SIGMOID_THRESHOLD},
        "trust_params": {"alpha": TRUST_ALPHA, "initial": INITIAL_TRUST},
        "round_comparison": round_comparison,
    }



# ══════════════════════════════════════════════════════════════
#  API Routes
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Run the full analysis on user-submitted node configuration."""
    data = request.json
    nodes_config = data.get("nodes", [])
    correct_answer = data.get("correct_answer", "YES")
    num_rounds = data.get("num_rounds", NUM_SIM_ROUNDS)
    num_rounds = max(5, min(100, int(num_rounds)))

    if len(nodes_config) < 2:
        return jsonify({"error": "Need at least 2 nodes"}), 400

    result = analyze_votes(nodes_config, correct_answer, num_rounds=num_rounds)
    return jsonify(result)


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    """Serve PNG images from the results directory."""
    return send_from_directory(RESULTS_DIR, filename)


# ══════════════════════════════════════════════════════════════
#  HTML TEMPLATE — Professional Dark UI
# ══════════════════════════════════════════════════════════════

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrustMesh — Trust-Weighted Distributed Consensus</title>
<meta name="description" content="Interactive demonstration of Trust-Weighted Consensus vs Raft in distributed systems with Byzantine Fault Tolerance analysis.">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
/* ══════════════════════════════════════════════════════════ */
/*  CSS DESIGN SYSTEM                                        */
/* ══════════════════════════════════════════════════════════ */
:root {
  --bg-primary: #06060e;
  --bg-secondary: #0a0a16;
  --bg-surface: #0f0f1e;
  --bg-card: #13132a;
  --bg-card-hover: #191940;
  --bg-elevated: #1c1c3a;

  --accent-primary: #00d4aa;
  --accent-secondary: #48dbfb;
  --accent-tertiary: #a29bfe;
  --accent-gradient: linear-gradient(135deg, #00d4aa 0%, #48dbfb 50%, #a29bfe 100%);

  --danger: #ff4757;
  --danger-dim: rgba(255,71,87,0.12);
  --warning: #ffa502;
  --warning-dim: rgba(255,165,2,0.12);
  --success: #00d4aa;
  --success-dim: rgba(0,212,170,0.12);
  --info: #48dbfb;
  --info-dim: rgba(72,219,251,0.12);
  --purple: #a29bfe;
  --purple-dim: rgba(162,155,254,0.12);

  --text-primary: #e8e8f4;
  --text-secondary: #9898b8;
  --text-muted: #5a5a7a;

  --border: rgba(255,255,255,0.06);
  --border-active: rgba(0,212,170,0.3);
  --glass: rgba(15,15,30,0.7);

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 20px;

  --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 20px rgba(0,0,0,0.4);
  --shadow-lg: 0 8px 40px rgba(0,0,0,0.5);
  --shadow-glow: 0 0 30px rgba(0,212,170,0.15);

  --mono: 'JetBrains Mono', 'Consolas', monospace;
  --sans: 'Inter', system-ui, sans-serif;
  --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

* { margin:0; padding:0; box-sizing:border-box; }
html { scroll-behavior: smooth; }

body {
  font-family: var(--sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  min-height: 100vh;
  line-height: 1.6;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a5a; }

/* ── HEADER ── */
.header {
  position: relative;
  background: linear-gradient(180deg, #0d1b3e 0%, #0a0a16 100%);
  padding: 2.5rem 2rem;
  text-align: center;
  border-bottom: 1px solid var(--border);
  overflow: hidden;
}
.header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at 30% 50%, rgba(0,212,170,0.04) 0%, transparent 50%),
              radial-gradient(circle at 70% 50%, rgba(72,219,251,0.03) 0%, transparent 50%);
  animation: headerPulse 8s ease-in-out infinite;
}
@keyframes headerPulse {
  0%,100% { opacity: 0.5; }
  50% { opacity: 1; }
}

.header-content { position: relative; z-index: 1; }

.header h1 {
  font-size: 2.4rem;
  font-weight: 900;
  letter-spacing: -0.5px;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.3rem;
}
.header-subtitle {
  color: var(--text-secondary);
  font-size: 0.95rem;
  font-weight: 400;
}
.header-tags {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 0.8rem;
  flex-wrap: wrap;
}
.header-tag {
  font-size: 0.65rem;
  padding: 3px 10px;
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

/* ── CONTAINER ── */
.container { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; }

/* ── GLASS CARD ── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: var(--shadow-md);
  transition: var(--transition);
}
.glass-card:hover {
  border-color: rgba(255,255,255,0.1);
  box-shadow: var(--shadow-lg);
}

/* ── BUTTONS ── */
button {
  border: none;
  border-radius: var(--radius-sm);
  padding: 10px 20px;
  font-weight: 600;
  font-size: 0.85rem;
  cursor: pointer;
  transition: var(--transition);
  font-family: var(--sans);
}
button:active { transform: scale(0.97); }

.btn-primary {
  background: var(--accent-gradient);
  color: #06060e;
  font-weight: 700;
  padding: 12px 28px;
  border-radius: var(--radius-md);
  font-size: 0.95rem;
  box-shadow: 0 4px 15px rgba(0,212,170,0.2);
}
.btn-primary:hover {
  box-shadow: 0 6px 30px rgba(0,212,170,0.35);
  transform: translateY(-2px);
}
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }

.btn-ghost {
  background: transparent;
  color: var(--text-secondary);
  border: 1px solid var(--border);
  padding: 8px 16px;
  font-size: 0.8rem;
}
.btn-ghost:hover { border-color: var(--accent-primary); color: var(--accent-primary); }
.btn-ghost.active {
  background: rgba(0,212,170,0.1);
  border-color: var(--accent-primary);
  color: var(--accent-primary);
}

/* ── INPUT SECTION ── */
.input-section {
  padding: 2rem;
  margin-bottom: 2rem;
}
.input-section h2 {
  font-size: 1.3rem;
  font-weight: 800;
  margin-bottom: 0.4rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.input-section .section-desc {
  color: var(--text-secondary);
  font-size: 0.85rem;
  margin-bottom: 1.5rem;
  line-height: 1.7;
}

.config-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
}
.config-row label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-secondary);
}
.config-row input[type="number"] {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  border-radius: var(--radius-sm);
  padding: 8px 14px;
  width: 80px;
  font-size: 1rem;
  outline: none;
  text-align: center;
  font-family: var(--mono);
  transition: var(--transition);
}
.config-row input:focus { border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(0,212,170,0.1); }

/* ── NODE GRID ── */
.node-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.node-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  transition: var(--transition);
}
.node-card:hover { border-color: rgba(255,255,255,0.12); }
.node-card .nc-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}
.node-card .nc-name {
  font-size: 1.1rem;
  font-weight: 800;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.node-card .nc-id {
  font-size: 0.65rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.node-card .nc-field {
  margin-bottom: 0.7rem;
}
.node-card .nc-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 0.3rem;
  font-weight: 600;
}

/* Type selector */
.type-selector {
  display: flex;
  gap: 4px;
}
.type-btn {
  flex: 1;
  padding: 6px 8px;
  font-size: 0.72rem;
  font-weight: 700;
  border-radius: 6px;
  text-align: center;
  border: 1.5px solid var(--border);
  background: transparent;
  cursor: pointer;
  transition: var(--transition);
  color: var(--text-muted);
}
.type-btn.good:hover, .type-btn.good.active { background: var(--success-dim); border-color: var(--success); color: var(--success); }
.type-btn.malicious:hover, .type-btn.malicious.active { background: var(--danger-dim); border-color: var(--danger); color: var(--danger); }
.type-btn.faulty:hover, .type-btn.faulty.active { background: var(--warning-dim); border-color: var(--warning); color: var(--warning); }

/* Trust slider */
.trust-slider-wrap {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.trust-slider {
  -webkit-appearance: none;
  appearance: none;
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
  outline: none;
  cursor: pointer;
}
.trust-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--text-primary);
  box-shadow: 0 0 6px rgba(0,0,0,0.5);
  cursor: pointer;
}
.trust-val {
  font-family: var(--mono);
  font-size: 0.85rem;
  font-weight: 700;
  min-width: 32px;
  text-align: right;
}

/* Presets */
.presets-section {
  margin-bottom: 1.5rem;
}
.presets-section .ps-label {
  font-size: 0.7rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 0.5rem;
  font-weight: 600;
}
.presets-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.analyze-row {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-top: 1rem;
}

/* ══════════════════════════════════════════════════════════ */
/*  RESULTS                                                   */
/* ══════════════════════════════════════════════════════════ */
.results-section { animation: fadeUp 0.6s cubic-bezier(0.4, 0, 0.2, 1); }
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(30px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Tabs */
.tabs-nav {
  display: flex;
  gap: 4px;
  margin-bottom: 1.5rem;
  padding: 4px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  overflow-x: auto;
}
.tab-btn {
  padding: 10px 18px;
  font-size: 0.8rem;
  font-weight: 600;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-muted);
  white-space: nowrap;
  border: none;
  cursor: pointer;
  transition: var(--transition);
}
.tab-btn:hover { color: var(--text-secondary); background: rgba(255,255,255,0.03); }
.tab-btn.active {
  color: var(--accent-primary);
  background: rgba(0,212,170,0.1);
  box-shadow: 0 0 10px rgba(0,212,170,0.1);
}
.tab-content { display: none; animation: fadeUp 0.4s ease; }
.tab-content.active { display: block; }

/* Section card */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  margin-bottom: 1.5rem;
  overflow: hidden;
}
.result-card-header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-weight: 700;
  font-size: 0.95rem;
}
.result-card-header .step-badge {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 800;
  flex-shrink: 0;
}
.step-badge.accent { background: var(--success-dim); color: var(--success); }
.step-badge.danger { background: var(--danger-dim); color: var(--danger); }
.step-badge.warning { background: var(--warning-dim); color: var(--warning); }
.step-badge.info { background: var(--info-dim); color: var(--info); }
.step-badge.purple { background: var(--purple-dim); color: var(--purple); }
.result-card-body { padding: 1.5rem; }

/* Vote rows */
.vote-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  margin-bottom: 3px;
  font-family: var(--mono);
  font-size: 0.85rem;
  background: var(--bg-surface);
  transition: var(--transition);
}
.vote-row:hover { background: var(--bg-elevated); }
.vr-name { font-weight: 700; min-width: 24px; }
.vr-arrow { color: var(--text-muted); }
.vr-vote { font-weight: 700; }
.vr-vote.yes { color: var(--success); }
.vr-vote.no { color: var(--danger); }
.vr-vote.timeout { color: var(--warning); }
.vr-meta { font-size: 0.75rem; color: var(--text-muted); margin-left: auto; }

/* Tally box */
.tally-box {
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  margin-top: 1rem;
  font-family: var(--mono);
  border: 1px solid var(--border);
}
.tally-line { margin-bottom: 0.4rem; font-size: 0.88rem; color: var(--text-secondary); }
.tally-line .val { font-weight: 700; color: var(--text-primary); }
.tally-decision {
  font-size: 1.2rem;
  font-weight: 800;
  margin-top: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px solid var(--border);
}
.tally-decision.correct { color: var(--success); }
.tally-decision.wrong { color: var(--danger); }
.tally-note {
  font-size: 0.82rem;
  margin-top: 0.5rem;
  padding: 0.6rem 0.8rem;
  border-radius: var(--radius-sm);
  font-family: var(--sans);
}
.tally-note.good { background: var(--success-dim); color: var(--success); }
.tally-note.bad { background: var(--danger-dim); color: var(--danger); }

/* Node classification grid */
.classify-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}
.classify-card {
  background: var(--bg-surface);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  position: relative;
  overflow: hidden;
  transition: var(--transition);
}
.classify-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.classify-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
}
.classify-card.good { border-color: rgba(0,212,170,0.25); }
.classify-card.good::before { background: var(--success); }
.classify-card.malicious { border-color: rgba(255,71,87,0.25); }
.classify-card.malicious::before { background: var(--danger); }
.classify-card.faulty { border-color: rgba(255,165,2,0.25); }
.classify-card.faulty::before { background: var(--warning); }
.cc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.cc-name { font-weight: 800; font-size: 1.05rem; }
.cc-badge {
  font-size: 0.6rem;
  padding: 3px 10px;
  border-radius: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
}
.cc-badge.good { background: var(--success-dim); color: var(--success); }
.cc-badge.malicious { background: var(--danger-dim); color: var(--danger); }
.cc-badge.faulty { background: var(--warning-dim); color: var(--warning); }

.cc-trust-val { font-size: 1.8rem; font-weight: 800; text-align: center; margin: 0.3rem 0; }
.cc-trust-bar { height: 6px; background: var(--bg-card); border-radius: 3px; overflow: hidden; margin-bottom: 0.4rem; }
.cc-trust-fill { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }

.cc-stats {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.3rem;
  margin: 0.5rem 0;
  font-size: 0.72rem;
  text-align: center;
}
.cc-stat {
  padding: 4px;
  border-radius: 4px;
  background: var(--bg-card);
}
.cc-stat-val { font-weight: 700; font-family: var(--mono); }
.cc-stat-label { color: var(--text-muted); font-size: 0.6rem; text-transform: uppercase; }
.cc-reason { font-size: 0.75rem; color: var(--text-secondary); line-height: 1.5; margin-top: 0.4rem; }

/* Comparison grid */
.comp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
@media(max-width:700px) { .comp-grid { grid-template-columns: 1fr; } }
.comp-card {
  border-radius: var(--radius-md);
  padding: 1.5rem;
  text-align: center;
  transition: var(--transition);
}
.comp-card:hover { transform: translateY(-3px); }
.comp-card.raft {
  background: linear-gradient(135deg, rgba(255,71,87,0.06), rgba(255,71,87,0.02));
  border: 1px solid rgba(255,71,87,0.2);
}
.comp-card.tm {
  background: linear-gradient(135deg, rgba(0,212,170,0.06), rgba(0,212,170,0.02));
  border: 1px solid rgba(0,212,170,0.2);
}
.comp-card .cc-sys {
  font-size: 0.65rem;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-muted);
  margin-bottom: 0.2rem;
}
.comp-card .cc-dec { font-size: 2.5rem; font-weight: 900; }
.comp-card.raft .cc-dec { color: var(--danger); }
.comp-card.tm .cc-dec { color: var(--success); }
.comp-card .cc-result { font-size: 0.95rem; font-weight: 700; margin-top: 0.2rem; }
.comp-card .cc-result.correct { color: var(--success); }
.comp-card .cc-result.wrong { color: var(--danger); }

/* Winner banner */
.winner-banner {
  border-radius: var(--radius-md);
  padding: 1.2rem 1.5rem;
  margin-top: 1rem;
  font-size: 0.9rem;
  line-height: 1.7;
}
.winner-banner.tm-wins {
  background: linear-gradient(135deg, var(--success-dim), rgba(72,219,251,0.05));
  border: 1px solid rgba(0,212,170,0.3);
  color: var(--success);
}
.winner-banner.both-correct {
  background: var(--info-dim);
  border: 1px solid rgba(72,219,251,0.3);
  color: var(--info);
}
.winner-banner.both-wrong {
  background: var(--danger-dim);
  border: 1px solid rgba(255,71,87,0.3);
  color: var(--danger);
}

/* BFT Card */
.bft-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media(max-width:600px) { .bft-grid { grid-template-columns: 1fr; } }
.bft-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  text-align: center;
}
.bft-card .bft-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.bft-card .bft-value { font-size: 1.8rem; font-weight: 800; font-family: var(--mono); }
.bft-formula {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--text-secondary);
  margin-top: 1rem;
  text-align: center;
}

/* Metrics table */
.metrics-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.85rem;
}
.metrics-table thead th {
  text-align: left;
  padding: 12px 16px;
  background: var(--bg-surface);
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 1px;
  font-weight: 700;
  border-bottom: 1px solid var(--border);
}
.metrics-table thead th:first-child { border-radius: var(--radius-sm) 0 0 0; }
.metrics-table thead th:last-child { border-radius: 0 var(--radius-sm) 0 0; }
.metrics-table tbody td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.metrics-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.metrics-table .metric-name { font-weight: 700; color: var(--text-primary); }
.metrics-table .metric-desc { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
.metrics-table .metric-val { font-family: var(--mono); font-weight: 600; }
.metrics-table .metric-val.raft { color: var(--danger); }
.metrics-table .metric-val.tm { color: var(--success); }

/* Chart containers */
.chart-wrap {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  margin-top: 1rem;
}
.chart-wrap h4 {
  font-size: 0.78rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 0.5rem;
}

/* Network topology */
.topology-canvas-wrap {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  margin-top: 1rem;
  text-align: center;
}
#topologyCanvas {
  border-radius: var(--radius-sm);
  max-width: 100%;
}

/* Info boxes */
.info-box {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1.2rem;
  margin-top: 1rem;
}
.info-box h4 {
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.info-box p, .info-box li {
  font-size: 0.82rem;
  color: var(--text-secondary);
  line-height: 1.7;
}
.info-box ul {
  list-style: none;
  padding: 0;
}
.info-box ul li { padding: 3px 0; }
.info-box ul li::before { margin-right: 0.5rem; font-size: 0.7rem; }

/* Why grid */
.why-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
@media(max-width:700px) { .why-grid { grid-template-columns: 1fr; } }
.why-card { border-radius: var(--radius-md); padding: 1.2rem; }
.why-card.good-card { background: var(--success-dim); border: 1px solid rgba(0,212,170,0.2); }
.why-card.bad-card { background: var(--danger-dim); border: 1px solid rgba(255,71,87,0.2); }
.why-card h4 { font-size: 0.85rem; margin-bottom: 0.6rem; }
.why-card.good-card h4 { color: var(--success); }
.why-card.bad-card h4 { color: var(--danger); }
.why-card ul { list-style: none; font-size: 0.82rem; color: var(--text-secondary); }
.why-card ul li { padding: 3px 0; }

.hidden { display: none !important; }

/* Loading spinner */
.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid #06060e;
  border-top: 2px solid transparent;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
  margin-right: 6px;
  vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Footer */
.footer {
  text-align: center;
  padding: 2rem;
  color: var(--text-muted);
  font-size: 0.75rem;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════════════════ -->
<!--  HEADER                                                    -->
<!-- ══════════════════════════════════════════════════════════ -->
<header class="header">
  <div class="header-content">
    <h1>TrustMesh</h1>
    <p class="header-subtitle">Trust-Weighted Consensus for Fault Tolerant Distributed Systems</p>
    <div class="header-tags">
      <span class="header-tag">Byzantine Fault Tolerance</span>
      <span class="header-tag">Consensus Protocol</span>
      <span class="header-tag">Trust Systems</span>
      <span class="header-tag">Leader Election</span>
      <span class="header-tag">Distributed Computing</span>
    </div>
  </div>
</header>

<div class="container">

  <!-- ══════════════════════════════════════════════════════════ -->
  <!--  INPUT SECTION                                             -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <div class="glass-card input-section" id="inputSection">
    <h2>Node Configuration</h2>
    <div class="section-desc">
      Configure the nodes in your distributed network. Each node has a <strong>type</strong> (Good, Malicious, or Faulty)
      and an <strong>initial trust score</strong> (0–10). The system will simulate multiple consensus rounds to build
      trust profiles, then compare Raft (equal voting) vs TrustMesh (trust-weighted voting).
    </div>

    <div class="config-row">
      <label>Number of Nodes:</label>
      <input type="number" id="nodeCount" value="7" min="2" max="26" onchange="buildNodeCards()">
      <button class="btn-ghost" onclick="buildNodeCards()">Update</button>
    </div>

    <div class="config-row">
      <label>Simulation Rounds:</label>
      <input type="number" id="roundCount" value="20" min="5" max="100" step="1">
      <span style="color:var(--text-muted);font-size:0.78rem;">Number of trust-building rounds (5–100)</span>
    </div>

    <div class="presets-section">
      <div class="ps-label">Quick Scenarios</div>
      <div class="presets-row">
        <button class="btn-ghost" onclick="loadPreset('safe')">3G + 2M (Safe)</button>
        <button class="btn-ghost" onclick="loadPreset('boundary')">3G + 4M (Raft fails)</button>
        <button class="btn-ghost" onclick="loadPreset('mixed')">3G + 2M + 2F (Mixed)</button>
        <button class="btn-ghost" onclick="loadPreset('extreme')">4G + 6M (Extreme)</button>
        <button class="btn-ghost" onclick="loadPreset('all_good')">5G (Baseline)</button>
      </div>
    </div>

    <div class="node-grid" id="nodeGrid"></div>

    <div class="analyze-row">
      <button class="btn-primary" onclick="runAnalysis()" id="analyzeBtn">
        Run Consensus Analysis
      </button>
      <span style="color:var(--text-muted);font-size:0.78rem;">
        Ground truth: <strong style="color:var(--success)">YES</strong> is the correct consensus value
      </span>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!--  RESULTS                                                   -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <div id="resultsContainer" class="hidden results-section"></div>

</div>

<footer class="footer">
  TrustMesh — Distributed Systems Project · B.Tech CSE · 6th Semester
</footer>

<script>
// ══════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════
let nodeConfigs = [];

function buildNodeCards() {
  const n = Math.min(26, Math.max(2, parseInt(document.getElementById('nodeCount').value) || 7));
  document.getElementById('nodeCount').value = n;
  const grid = document.getElementById('nodeGrid');
  nodeConfigs = [];

  let html = '';
  for (let i = 0; i < n; i++) {
    const name = String.fromCharCode(65 + i);
    nodeConfigs.push({ name, type: 'good', trust: 5 });
    html += `
      <div class="node-card" id="ncard-${i}">
        <div class="nc-header">
          <span class="nc-name">Node ${name}</span>
          <span class="nc-id">ID: ${i}</span>
        </div>
        <div class="nc-field">
          <div class="nc-label">Node Type</div>
          <div class="type-selector">
            <button class="type-btn good active" id="tb-${i}-good" onclick="setType(${i},'good')">Good</button>
            <button class="type-btn malicious" id="tb-${i}-malicious" onclick="setType(${i},'malicious')">Malicious</button>
            <button class="type-btn faulty" id="tb-${i}-faulty" onclick="setType(${i},'faulty')">Faulty</button>
          </div>
        </div>
        <div class="nc-field">
          <div class="nc-label">Initial Trust Score</div>
          <div class="trust-slider-wrap">
            <input type="range" class="trust-slider" id="ts-${i}" min="0" max="10" step="1" value="5"
                   oninput="setTrust(${i}, this.value)">
            <span class="trust-val" id="tv-${i}">5</span>
          </div>
        </div>
      </div>`;
  }
  grid.innerHTML = html;
  document.getElementById('resultsContainer').classList.add('hidden');
}

function setType(idx, type) {
  nodeConfigs[idx].type = type;
  ['good', 'malicious', 'faulty'].forEach(t => {
    document.getElementById(`tb-${idx}-${t}`).classList.toggle('active', t === type);
  });
}

function setTrust(idx, val) {
  const v = parseInt(val);
  nodeConfigs[idx].trust = v;
  document.getElementById(`tv-${idx}`).textContent = v;
}

function loadPreset(preset) {
  const presets = {
    safe: [
      {type:'good',trust:8},{type:'good',trust:7},{type:'good',trust:8},
      {type:'malicious',trust:4},{type:'malicious',trust:3}
    ],
    boundary: [
      {type:'good',trust:9},{type:'good',trust:8},{type:'good',trust:9},
      {type:'malicious',trust:4},{type:'malicious',trust:3},
      {type:'malicious',trust:4},{type:'malicious',trust:3}
    ],
    mixed: [
      {type:'good',trust:8},{type:'good',trust:8},{type:'good',trust:7},
      {type:'malicious',trust:4},{type:'malicious',trust:3},
      {type:'faulty',trust:5},{type:'faulty',trust:6}
    ],
    extreme: [
      {type:'good',trust:10},{type:'good',trust:9},{type:'good',trust:9},{type:'good',trust:8},
      {type:'malicious',trust:4},{type:'malicious',trust:4},{type:'malicious',trust:3},
      {type:'malicious',trust:3},{type:'malicious',trust:2},{type:'malicious',trust:2}
    ],
    all_good: [
      {type:'good',trust:7},{type:'good',trust:7},{type:'good',trust:8},
      {type:'good',trust:8},{type:'good',trust:6}
    ],
  };

  const p = presets[preset];
  if (!p) return;

  document.getElementById('nodeCount').value = p.length;
  buildNodeCards();

  p.forEach((cfg, i) => {
    setType(i, cfg.type);
    document.getElementById(`ts-${i}`).value = cfg.trust;
    setTrust(i, cfg.trust);
  });
}

// ══════════════════════════════════════════════════════════════
//  ANALYSIS
// ══════════════════════════════════════════════════════════════
async function runAnalysis() {
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analyzing...';

  try {
    const numRounds = Math.min(100, Math.max(5, parseInt(document.getElementById('roundCount').value) || 20));
    document.getElementById('roundCount').value = numRounds;
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nodes: nodeConfigs, correct_answer: 'YES', num_rounds: numRounds })
    });
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert('Analysis failed: ' + err.message);
  }

  btn.disabled = false;
  btn.textContent = 'Run Consensus Analysis';
  document.getElementById('resultsContainer').scrollIntoView({ behavior: 'smooth' });
}

// ══════════════════════════════════════════════════════════════
//  RENDER RESULTS
// ══════════════════════════════════════════════════════════════
function renderResults(data) {
  const container = document.getElementById('resultsContainer');
  container.classList.remove('hidden');

  const raft = data.raft;
  const tm = data.trustmesh;
  const nodes = data.nodes;
  const bft = data.bft;
  const metrics = data.eval_metrics;

  const raftDecClass = raft.is_correct ? 'correct' : 'wrong';
  const tmDecClass = tm.is_correct ? 'correct' : 'wrong';
  const raftIcon = raft.is_correct ? 'CORRECT' : 'WRONG';
  const tmIcon = tm.is_correct ? 'CORRECT' : 'WRONG';

  // ── Raft votes ──
  let raftVotesHtml = nodes.map(n => {
    const v = data.raft.votes[n.name];
    const vClass = v === 'YES' ? 'yes' : (v === 'NO' ? 'no' : 'timeout');
    const vText = v || 'TIMEOUT';
    return `
      <div class="vote-row">
        <span class="vr-name">${n.name}</span>
        <span class="vr-arrow">→</span>
        <span class="vr-vote ${vClass}">${vText}</span>
        <span class="vr-meta">(1 vote — equal weight)</span>
      </div>`;
  }).join('');

  // ── Node classification ──
  let classCards = nodes.map(n => {
    const pct = (n.trust / 10 * 100).toFixed(0);
    const colors = {good: 'var(--success)', malicious: 'var(--danger)', faulty: 'var(--warning)'};
    const barColor = colors[n.classification] || 'var(--text-muted)';
    const icons = {good: 'GOOD', malicious: 'MALICIOUS', faulty: 'FAULTY'};
    return `
      <div class="classify-card ${n.classification}">
        <div class="cc-header">
          <span class="cc-name">${n.name}</span>
          <span class="cc-badge ${n.classification}">${icons[n.classification]}</span>
        </div>
        <div class="cc-trust-val" style="color:${barColor}">${n.trust}</div>
        <div class="cc-trust-bar"><div class="cc-trust-fill" style="width:${pct}%;background:${barColor}"></div></div>
        <div class="cc-stats">
          <div class="cc-stat">
            <div class="cc-stat-val" style="color:var(--success)">${n.correct_count}</div>
            <div class="cc-stat-label">Correct</div>
          </div>
          <div class="cc-stat">
            <div class="cc-stat-val" style="color:var(--danger)">${n.wrong_count}</div>
            <div class="cc-stat-label">Wrong</div>
          </div>
          <div class="cc-stat">
            <div class="cc-stat-val" style="color:var(--warning)">${n.timeout_count}</div>
            <div class="cc-stat-label">Timeout</div>
          </div>
        </div>
        <div class="cc-reason">${n.class_reason}</div>
      </div>`;
  }).join('');

  // ── TrustMesh weighted votes ──
  let tmVotesHtml = nodes.map(n => {
    const v = n.final_vote;
    const vClass = v === 'YES' ? 'yes' : (v === 'NO' ? 'no' : 'timeout');
    const vText = v || 'TIMEOUT';
    return `
      <div class="vote-row">
        <span class="vr-name">${n.name}</span>
        <span style="color:var(--text-muted);font-size:0.72rem;">[trust=${n.trust}]</span>
        <span class="vr-arrow">→</span>
        <span class="vr-vote ${vClass}">${vText}</span>
        <span class="vr-meta">weight = ${n.final_weight}</span>
      </div>`;
  }).join('');

  const yesCalc = tm.yes_parts.map(p => `${p.weight}`).join(' + ') || '0';
  const noCalc = tm.no_parts.map(p => `${p.weight}`).join(' + ') || '0';

  // ── Winner banner ──
  let winnerClass, winnerHtml;
  if (!raft.is_correct && tm.is_correct) {
    winnerClass = 'tm-wins';
    winnerHtml = `<strong>TrustMesh Wins</strong><br>
      Raft failed because malicious nodes had <strong>equal voting power</strong>.<br>
      TrustMesh succeeded by <strong>weighting votes using learned trust scores</strong> — trusted nodes dominate the consensus.`;
  } else if (raft.is_correct && tm.is_correct) {
    winnerClass = 'both-correct';
    winnerHtml = `<strong>Both systems reached correct consensus.</strong><br>
      The honest majority is sufficient for equal-weight voting. TrustMesh still provides <strong>node reputation tracking</strong> for future rounds.`;
  } else if (!raft.is_correct && !tm.is_correct) {
    winnerClass = 'both-wrong';
    winnerHtml = `<strong>Both systems failed.</strong><br>
      The number of faulty nodes exceeds tolerance limits. No consensus protocol guarantees correctness in this case.`;
  } else {
    winnerClass = 'both-correct';
    winnerHtml = '<strong>Raft succeeded in this configuration.</strong> This can happen when trust evolution timing creates a temporary disadvantage.';
  }

  // ── BFT analysis ──
  const bftStatusColor = bft.is_safe ? 'var(--success)' : 'var(--danger)';
  const bftStatusText = bft.is_safe ? 'WITHIN TOLERANCE' : 'EXCEEDS TOLERANCE';

  // ── Metrics table ──
  const metricsRows = [
    ['Consensus Accuracy', 'consensus_accuracy'],
    ['Fault Tolerance', 'fault_tolerance'],
    ['Average Latency', 'avg_latency'],
    ['Leader Stability', 'leader_stability'],
    ['Message Overhead', 'message_overhead'],
    ['Throughput', 'throughput'],
  ].map(([label, key]) => {
    const m = metrics[key];
    return `<tr>
      <td>
        <div class="metric-name">${label}</div>
        <div class="metric-desc">${m.description}</div>
      </td>
      <td class="metric-val raft">${m.raft}</td>
      <td class="metric-val tm">${m.trustmesh}</td>
    </tr>`;
  }).join('');

  // Chart IDs
  const trustChartId = 'trustChart_' + Date.now();
  const sigmoidChartId = 'sigmoidChart_' + Date.now();
  const roundScoreChartId = 'roundScoreChart_' + Date.now();
  const roundCompChartId = 'roundCompChart_' + Date.now();
  const topologyId = 'topologyCanvas';
  // Analysis Charts tab IDs
  const acAccuracyId = 'acAccuracy_' + Date.now();
  const acTrustEvoId = 'acTrustEvo_' + Date.now();
  const acLatencyId = 'acLatency_' + Date.now();
  const acThroughputId = 'acThroughput_' + Date.now();
  const acMsgOverheadId = 'acMsgOverhead_' + Date.now();
  const acOverallId = 'acOverall_' + Date.now();

  // ══════════════════════════════════════════════════════════
  //  FULL HTML
  // ══════════════════════════════════════════════════════════
  container.innerHTML = `
    <!-- TABS NAV -->
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="switchTab('consensus')" id="tabbtn-consensus">Consensus Results</button>
      <button class="tab-btn" onclick="switchTab('comparison')" id="tabbtn-comparison">Round Comparison</button>
      <button class="tab-btn" onclick="switchTab('charts')" id="tabbtn-charts">Analysis Charts</button>
      <button class="tab-btn" onclick="switchTab('static')" id="tabbtn-static">Static Charts</button>
      <button class="tab-btn" onclick="switchTab('trust')" id="tabbtn-trust">Trust Analysis</button>
      <button class="tab-btn" onclick="switchTab('concepts')" id="tabbtn-concepts">DS Concepts</button>
      <button class="tab-btn" onclick="switchTab('metrics')" id="tabbtn-metrics">Evaluation Metrics</button>
    </div>

    <!-- ═══ TAB 1: CONSENSUS RESULTS ═══ -->
    <div class="tab-content active" id="tab-consensus">

      <!-- Raft -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge danger">1</span>
          Raft — Traditional Equal-Weight Consensus
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem;">
            In Raft, every node gets <strong style="color:var(--text-primary)">exactly 1 vote</strong> regardless of
            its past behavior. A simple majority determines the consensus decision. This is vulnerable to Byzantine attacks.
          </p>
          <div style="font-weight:600;font-size:0.82rem;margin-bottom:0.5rem;color:var(--text-muted);">VOTING:</div>
          ${raftVotesHtml}
          <div class="tally-box">
            <div class="tally-line">Vote Count: <span class="val">${raft.yes_count} YES</span> vs <span class="val">${raft.no_count} NO</span>${raft.timeout_count > 0 ? ` (<span class="val">${raft.timeout_count} timeout</span>)` : ''}</div>
            <div class="tally-line">Majority threshold: <span class="val">${Math.ceil(data.num_nodes / 2)} of ${data.num_nodes}</span> nodes</div>
            <div class="tally-decision ${raftDecClass}">Decision: ${raft.decision} ${raftIcon}</div>
            ${!raft.is_correct ? '<div class="tally-note bad">Malicious nodes corrupted the consensus. Equal voting gave adversaries the same power as honest nodes.</div>' : '<div class="tally-note good">Honest majority is sufficient — equal voting works when most nodes are good.</div>'}
          </div>
        </div>
      </div>

      <!-- TrustMesh Weighted Voting -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge accent">2</span>
          TrustMesh — Trust-Weighted Consensus
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem;">
            TrustMesh uses <strong style="color:var(--text-primary)">learned trust scores</strong> converted to
            <strong style="color:var(--accent-primary)">voting weights via sigmoid function</strong>.
            Trusted nodes carry more influence. Adversarial nodes are suppressed by their low trust.
          </p>
          <div style="font-weight:600;font-size:0.82rem;margin-bottom:0.5rem;color:var(--text-muted);">WEIGHTED VOTING:</div>
          ${tmVotesHtml}
          <div class="tally-box">
            <div class="tally-line">YES weighted total = ${yesCalc} = <span class="val" style="color:var(--success)">${tm.yes_weighted}</span></div>
            <div class="tally-line">NO weighted total = ${noCalc} = <span class="val" style="color:var(--danger)">${tm.no_weighted}</span></div>
            <div class="tally-decision ${tmDecClass}">Decision: ${tm.decision} ${tmIcon}</div>
            ${tm.is_correct ? '<div class="tally-note good">Trust-weighted consensus succeeds. High-trust nodes override the adversarial minority.</div>' : '<div class="tally-note bad">Too many malicious nodes — even trust weighting cannot overcome this level of adversarial control.</div>'}
          </div>
        </div>
      </div>

      <!-- Comparison -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge purple">3</span>
          Final Comparison — Raft vs TrustMesh
        </div>
        <div class="result-card-body">
          <div class="comp-grid">
            <div class="comp-card raft">
              <div class="cc-sys">Raft (Equal Weight)</div>
              <div class="cc-dec">${raft.decision}</div>
              <div class="cc-result ${raftDecClass}">${raftIcon}</div>
            </div>
            <div class="comp-card tm">
              <div class="cc-sys">TrustMesh (Trust-Weighted)</div>
              <div class="cc-dec">${tm.decision}</div>
              <div class="cc-result ${tmDecClass}">${tmIcon}</div>
            </div>
          </div>
          <div class="winner-banner ${winnerClass}">${winnerHtml}</div>

          <div class="why-grid">
            <div class="why-card good-card">
              <h4>Why TrustMesh is Better</h4>
              <ul>
                <li>Votes weighted by earned trust — bad nodes can't override good ones</li>
                <li>EMA-based trust learning — builds profiles from behavioral history</li>
                <li>Sigmoid weight function — smooth, non-linear reputation mapping</li>
                <li>Self-healing — automatically reduces influence of detected adversaries</li>
              </ul>
            </div>
            <div class="why-card bad-card">
              <h4>Why Raft Fails</h4>
              <ul>
                <li>Equal voting — no differentiation between honest and malicious nodes</li>
                <li>No behavioral memory — cannot learn from past rounds</li>
                <li>Majority attack — more than 50% malicious nodes can take full control</li>
                <li>No trust model — every node is treated identically regardless of history</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══ TAB: ROUND COMPARISON ═══ -->
    <div class="tab-content" id="tab-comparison">
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge info">R</span>
          Per-Round Comparison — Raft vs TrustMesh
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem;">
            This table and chart compare the decisions made by <strong style="color:var(--danger)">Raft</strong> and
            <strong style="color:var(--success)">TrustMesh</strong> in each of the ${data.num_sim_rounds} simulation rounds.
            Cumulative accuracy shows how each system's correctness evolves over time.
            Ground truth: <strong style="color:var(--success)">${data.correct_answer}</strong>.
          </p>

          <!-- Accuracy Evolution Chart -->
          <div class="chart-wrap">
            <h4>Cumulative Accuracy — Raft vs TrustMesh per Round</h4>
            <canvas id="${roundCompChartId}"></canvas>
          </div>

          <!-- Per-round table -->
          <div style="overflow-x:auto;margin-top:1.5rem;">
            <table class="metrics-table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th style="color:var(--danger)">Raft Decision</th>
                  <th style="color:var(--danger)">Raft Votes (Y/N)</th>
                  <th style="color:var(--danger)">Raft Accuracy</th>
                  <th style="color:var(--success)">TrustMesh Decision</th>
                  <th style="color:var(--success)">TM Weighted (Y/N)</th>
                  <th style="color:var(--success)">TM Accuracy</th>
                </tr>
              </thead>
              <tbody>
                ${data.round_comparison.map(rc => `<tr>
                  <td style="font-weight:700;font-family:var(--mono);">R${rc.round}</td>
                  <td class="metric-val" style="color:${rc.raft_correct ? 'var(--success)' : 'var(--danger)'}">${rc.raft_decision} ${rc.raft_correct ? 'CORRECT' : 'WRONG'}</td>
                  <td class="metric-val raft">${rc.raft_yes} / ${rc.raft_no}</td>
                  <td class="metric-val raft">${rc.raft_accuracy}%</td>
                  <td class="metric-val" style="color:${rc.tm_correct ? 'var(--success)' : 'var(--danger)'}">${rc.tm_decision} ${rc.tm_correct ? 'CORRECT' : 'WRONG'}</td>
                  <td class="metric-val tm">${rc.tm_yes_w} / ${rc.tm_no_w}</td>
                  <td class="metric-val tm">${rc.tm_accuracy}%</td>
                </tr>`).join('')}
              </tbody>
            </table>
          </div>

          <!-- Summary -->
          <div class="bft-grid" style="margin-top:1.5rem;">
            <div class="bft-card" style="border-color:rgba(255,71,87,0.2);">
              <div class="bft-label" style="color:var(--danger)">Raft Final Accuracy</div>
              <div class="bft-value" style="color:var(--danger)">${data.round_comparison[data.round_comparison.length-1].raft_accuracy}%</div>
            </div>
            <div class="bft-card" style="border-color:rgba(0,212,170,0.2);">
              <div class="bft-label" style="color:var(--success)">TrustMesh Final Accuracy</div>
              <div class="bft-value" style="color:var(--success)">${data.round_comparison[data.round_comparison.length-1].tm_accuracy}%</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Detailed Per-Round Analysis -->
      <div class="result-card" style="margin-top:1.5rem;">
        <div class="result-card-header">
          <span class="step-badge accent">D</span>
          Detailed Round-by-Round Analysis — What Changed and Why
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1.5rem;">
            Each round below shows the key differences between Raft and TrustMesh, explaining
            how trust evolution affects consensus quality over time.
          </p>
          ${data.round_comparison.map((rc, idx) => {
            const prevRc = idx > 0 ? data.round_comparison[idx-1] : null;
            const raftChanged = prevRc ? (rc.raft_correct !== prevRc.raft_correct) : false;
            const tmChanged = prevRc ? (rc.tm_correct !== prevRc.tm_correct) : false;
            const raftAccDelta = prevRc ? rc.raft_accuracy - prevRc.raft_accuracy : rc.raft_accuracy;
            const tmAccDelta = prevRc ? rc.tm_accuracy - prevRc.tm_accuracy : rc.tm_accuracy;
            const roundNodes = data.history_log[idx] ? data.history_log[idx].node_data : {};
            const leader = data.history_log[idx] ? data.history_log[idx].leader : '—';

            let changeNote = '';
            if (!rc.raft_correct && rc.tm_correct) {
              changeNote = '<span style="color:var(--success);font-weight:700;">TrustMesh correct, Raft wrong</span> — Trust weighting neutralizes adversarial votes.';
            } else if (rc.raft_correct && rc.tm_correct) {
              changeNote = '<span style="color:var(--info);font-weight:700;">Both correct</span> — Honest majority is sufficient for both systems.';
            } else if (!rc.raft_correct && !rc.tm_correct) {
              changeNote = '<span style="color:var(--danger);font-weight:700;">Both wrong</span> — Adversarial majority overwhelms both systems.';
            } else {
              changeNote = '<span style="color:var(--warning);font-weight:700;">Raft correct, TrustMesh wrong</span> — Trust scores still evolving.';
            }

            return `
            <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-md);padding:1.2rem;margin-bottom:1rem;${(!rc.raft_correct && rc.tm_correct) ? 'border-left:3px solid var(--success);' : ((!rc.raft_correct && !rc.tm_correct) ? 'border-left:3px solid var(--danger);' : 'border-left:3px solid var(--info);')}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                <span style="font-weight:800;font-size:1.1rem;font-family:var(--mono);color:var(--accent-secondary);">Round ${rc.round}</span>
                <span style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;">Leader: Node ${leader}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:0.8rem;">
                <div style="background:rgba(255,71,87,0.05);border:1px solid rgba(255,71,87,0.15);border-radius:8px;padding:0.8rem;">
                  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;color:var(--danger);margin-bottom:0.3rem;font-weight:700;">Raft</div>
                  <div style="font-weight:800;font-size:1.1rem;color:${rc.raft_correct ? 'var(--success)' : 'var(--danger)'}">${rc.raft_decision} — ${rc.raft_correct ? 'CORRECT' : 'WRONG'}</div>
                  <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.3rem;">Votes: ${rc.raft_yes} YES / ${rc.raft_no} NO (equal weight)</div>
                  <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem;">Accuracy: ${rc.raft_accuracy}% ${prevRc ? '(' + (raftAccDelta >= 0 ? '+' : '') + raftAccDelta + '%)' : ''}</div>
                </div>
                <div style="background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.15);border-radius:8px;padding:0.8rem;">
                  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;color:var(--success);margin-bottom:0.3rem;font-weight:700;">TrustMesh</div>
                  <div style="font-weight:800;font-size:1.1rem;color:${rc.tm_correct ? 'var(--success)' : 'var(--danger)'}">${rc.tm_decision} — ${rc.tm_correct ? 'CORRECT' : 'WRONG'}</div>
                  <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.3rem;">Weighted: ${rc.tm_yes_w} YES / ${rc.tm_no_w} NO (trust-weighted)</div>
                  <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.2rem;">Accuracy: ${rc.tm_accuracy}% ${prevRc ? '(' + (tmAccDelta >= 0 ? '+' : '') + tmAccDelta + '%)' : ''}</div>
                </div>
              </div>
              <div style="font-size:0.82rem;color:var(--text-secondary);padding:0.5rem 0.8rem;background:var(--bg-card);border-radius:6px;">
                ${changeNote}
                ${raftChanged ? ' <span style="color:var(--warning);font-size:0.75rem;">[Raft outcome changed from previous round]</span>' : ''}
                ${tmChanged ? ' <span style="color:var(--accent-primary);font-size:0.75rem;">[TrustMesh outcome changed from previous round]</span>' : ''}
              </div>
              ${Object.keys(roundNodes).length > 0 ? `
              <div style="margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;">
                ${Object.entries(roundNodes).map(([name, nd]) => `
                  <span style="font-size:0.68rem;padding:3px 8px;border-radius:10px;font-family:var(--mono);font-weight:600;
                    background:${nd.vote === data.correct_answer ? 'var(--success-dim)' : (nd.vote === null ? 'var(--warning-dim)' : 'var(--danger-dim)')};
                    color:${nd.vote === data.correct_answer ? 'var(--success)' : (nd.vote === null ? 'var(--warning)' : 'var(--danger)')};
                  ">${name}: ${nd.vote || 'TIMEOUT'} [T=${nd.trust}, W=${Math.round(nd.weight)}]</span>
                `).join('')}
              </div>` : ''}
            </div>`;
          }).join('')}
        </div>
      </div>
    </div>

    <!-- ═══ TAB 2: TRUST ANALYSIS ═══ -->
    <div class="tab-content" id="tab-trust">

      <!-- Node Classification -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge info">A</span>
          Node Classification & Trust Profiles
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.5rem;">
            TrustMesh monitors each node over <strong style="color:var(--text-primary)">${data.num_sim_rounds} simulation rounds</strong>.
            Trust is updated using <strong style="color:var(--accent-primary)">Exponential Moving Average (EMA)</strong>:
          </p>
          <div class="bft-formula" style="margin-bottom:1rem;">
            T<sub>new</sub> = (alpha / 10) × T<sub>old</sub> + (1 − alpha/10) × S<sub>round</sub>
            &nbsp;&nbsp;where alpha = ${data.trust_params.alpha} (integer), S<sub>round</sub> ∈ [0, 10]
          </div>
          <p style="color:var(--text-secondary);font-size:0.82rem;margin-bottom:1rem;">
            <strong>Round scores are NOT binary.</strong> They depend on: correctness (primary), response latency, and whether the node responded at all.
            Examples: Correct + fast → 8.5–10.0 | Correct + slow → 5.5–7.5 | Timeout → 1.5–3.0 | Wrong → 0.5–2.8
          </p>
          <div class="classify-grid">${classCards}</div>

          <!-- Trust evolution chart -->
          <div class="chart-wrap">
            <h4>Trust Score Evolution Over ${data.num_sim_rounds} Rounds</h4>
            <canvas id="${trustChartId}"></canvas>
          </div>

          <!-- Round scores chart -->
          <div class="chart-wrap">
            <h4>Per-Round Scores (nuanced scoring, not binary)</h4>
            <canvas id="${roundScoreChartId}"></canvas>
          </div>
        </div>
      </div>

      <!-- Sigmoid explanation -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge purple">σ</span>
          Trust → Weight Conversion (Sigmoid Function)
        </div>
        <div class="result-card-body">
          <div class="info-box">
            <h4>Why convert trust scores to weights?</h4>
            <p>
              Raw trust scores (0–10) cannot be used directly as voting weights because:
            </p>
            <ul>
              <li>• A node with trust 9.0 should not have exactly 9× the power of a node with trust 1.0</li>
              <li>• We need a <strong>soft threshold</strong> — nodes above a midpoint get high influence, below get low</li>
              <li>• The <strong>sigmoid function</strong> provides a smooth S-curve that naturally models reputation-to-influence</li>
              <li>• It prevents abrupt cutoffs and is differentiable (useful for optimization)</li>
            </ul>
          </div>
          <div class="bft-formula">
            weight = int(100 / (1 + e<sup>−k(trust − threshold)</sup>))
            &nbsp;&nbsp;where k = ${data.sigmoid_params.k}, threshold = ${data.sigmoid_params.threshold} — outputs integer 0-100
          </div>
          <div class="chart-wrap">
            <h4>Sigmoid Weight Curve — Trust Score → Voting Weight</h4>
            <canvas id="${sigmoidChartId}"></canvas>
          </div>
          <div class="info-box" style="margin-top:1rem;">
            <h4>Per-Node Weight Values</h4>
            <table class="metrics-table" style="margin-top:0.5rem;">
              <thead><tr><th>Node</th><th>Trust Score</th><th>→ Sigmoid Weight</th><th>Classification</th></tr></thead>
              <tbody>
                ${nodes.map(n => `<tr>
                  <td style="font-weight:700">${n.name}</td>
                  <td class="metric-val" style="color:${n.trust >= 7 ? 'var(--success)' : n.trust < 4 ? 'var(--danger)' : 'var(--warning)'}">${n.trust}</td>
                  <td class="metric-val" style="color:var(--accent-secondary)">${n.final_weight}</td>
                  <td><span class="cc-badge ${n.classification}">${n.classification.toUpperCase()}</span></td>
                </tr>`).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══ TAB 3: DS CONCEPTS (Syllabus-Aligned) ═══ -->
    <div class="tab-content" id="tab-concepts">

      <!-- ── UNIT 1 ── -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge info">1</span>
          Unit 1 — Communication & System Models
        </div>
        <div class="result-card-body">

          <!-- Message Passing -->
          <div class="info-box">
            <h4 style="color:var(--accent-secondary)">Message Passing System (vs Shared Memory)</h4>
            <p>TrustMesh uses a <strong>message passing model</strong> — nodes do NOT share memory.
            Each node sends its vote as a message to all other nodes. This is the standard model
            for geographically distributed systems where shared memory is infeasible.</p>
            <ul style="margin-top:0.5rem;">
              <li>• <strong>In TrustMesh</strong>: Each node independently computes its vote and sends it as a message to the coordinator</li>
              <li>• <strong>Messages per round</strong>: Each node sends 1 vote request + 1 vote response = <strong>${data.num_nodes * 2} messages/round</strong></li>
              <li>• <strong>No shared state</strong>: Nodes do not access each other's trust scores — only the coordinator aggregates</li>
            </ul>
          </div>

          <!-- Sync vs Async -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--accent-secondary)">Synchronous vs Asynchronous Communication</h4>
            <p>TrustMesh operates in an <strong>asynchronous model</strong> — there is no global clock. Nodes respond at different times, may be slow, or may not respond at all.</p>
            <div class="bft-grid" style="margin-top:0.8rem;">
              <div class="bft-card">
                <div class="bft-label">Synchronous</div>
                <div class="bft-value" style="font-size:0.85rem;color:var(--text-secondary)">Fixed timeout, all nodes respond in bounded time. Simpler but unrealistic.</div>
              </div>
              <div class="bft-card">
                <div class="bft-label">Asynchronous (TrustMesh)</div>
                <div class="bft-value" style="font-size:0.85rem;color:var(--success)">No time bound. Nodes may be slow (latency varies) or not respond (crash fault). Realistic model.</div>
              </div>
            </div>
            <p style="margin-top:0.6rem;font-size:0.8rem;color:var(--text-muted)">
              In our simulation: Good nodes respond in 1-8ms, Faulty nodes in 3-50ms (or timeout), Malicious nodes in 2-15ms.
              The varying response times model real-world asynchronous behavior.
            </p>
          </div>

          <!-- Message Oriented Communication -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--accent-secondary)">Message Oriented Communication</h4>
            <p>TrustMesh uses <strong>request-response messaging</strong>, a form of message-oriented communication:</p>
            <ul>
              <li>• <strong>Step 1</strong>: Coordinator broadcasts a vote request to all ${data.num_nodes} nodes</li>
              <li>• <strong>Step 2</strong>: Each node processes independently and sends back a vote response</li>
              <li>• <strong>Step 3</strong>: Coordinator aggregates responses (weighted or equal) to reach consensus</li>
              <li>• <strong>Timeout handling</strong>: If a node doesn't respond, it's treated as a crash fault (penalty applied)</li>
            </ul>
            <p style="margin-top:0.5rem;font-size:0.8rem;color:var(--text-muted)">
              This is similar to how RPC (Remote Procedure Call) works — the coordinator "calls" each node's vote function remotely and waits for results.
            </p>
          </div>
        </div>
      </div>

      <!-- ── UNIT 2 ── -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge accent">2</span>
          Unit 2 — Consensus, Leader Election & Mutual Exclusion
        </div>
        <div class="result-card-body">

          <!-- Logical Time -->
          <div class="info-box">
            <h4 style="color:var(--success)">Logical Time and Event Ordering</h4>
            <p>Since TrustMesh is asynchronous (no global clock), we use <strong>logical time</strong>
            via <strong>simulation rounds</strong> to order events.</p>
            <ul>
              <li>• Each consensus round acts as a <strong>logical clock tick</strong> — Round 1 → Round 2 → ... → Round ${data.num_sim_rounds}</li>
              <li>• Events within a round are concurrent (all nodes vote independently)</li>
              <li>• Trust updates happen in causal order — Round R's trust depends on Round R-1's behavior</li>
              <li>• This follows <strong>Lamport's "happened-before"</strong> relation: vote(R) → trust_update(R) → vote(R+1)</li>
            </ul>
          </div>

          <!-- Leader Election -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--success)">Leader Election</h4>
            <p>Both Raft and TrustMesh elect leaders, but the mechanism is fundamentally different:</p>
            <div class="bft-grid" style="margin-top:0.8rem;">
              <div class="bft-card">
                <div class="bft-label">Raft Leader Election</div>
                <div class="bft-value" style="font-size:0.8rem;color:var(--danger)">Random timeout → candidate → majority vote. Any node can become leader, even a malicious one.</div>
              </div>
              <div class="bft-card">
                <div class="bft-label">TrustMesh Leader Election</div>
                <div class="bft-value" style="font-size:0.8rem;color:var(--success)">Highest trust score wins. Leader = node "${tm.final_leader}" (trust-based). ${tm.leader_changes} leader changes in ${data.num_sim_rounds} rounds.</div>
              </div>
            </div>
          </div>

          <!-- Consensus Algorithms: Raft vs TrustMesh -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--success)">Consensus Algorithms: Raft and Its Variant (TrustMesh)</h4>
            <p>TrustMesh is a <strong>trust-weighted variant of Raft</strong> that extends the Raft consensus model:</p>
            <div class="bft-grid" style="margin-top:0.8rem;">
              <div class="info-box" style="border-color:rgba(255,71,87,0.2);">
                <h4 style="color:var(--danger)">Raft (Standard)</h4>
                <ul>
                  <li>• <strong>Leader Election</strong>: Random timeout + majority vote</li>
                  <li>• <strong>Log Replication</strong>: Leader → followers</li>
                  <li>• <strong>Voting</strong>: 1 node = 1 vote</li>
                  <li>• <strong>Fault Tolerance</strong>: Crash faults only (f < n/2)</li>
                  <li>• <strong>Consistency</strong>: Strong consistency via log matching</li>
                </ul>
              </div>
              <div class="info-box" style="border-color:rgba(0,212,170,0.2);">
                <h4 style="color:var(--success)">TrustMesh (Raft Variant)</h4>
                <ul>
                  <li>• <strong>Leader Election</strong>: Trust-score based</li>
                  <li>• <strong>Trust Model</strong>: EMA learning over rounds</li>
                  <li>• <strong>Voting</strong>: weight = sigmoid(trust)</li>
                  <li>• <strong>Fault Tolerance</strong>: Byzantine faults via trust isolation</li>
                  <li>• <strong>Consistency</strong>: Weighted consensus + checkpointing</li>
                </ul>
              </div>
            </div>
          </div>

          <!-- Mutual Exclusion (Agreement) -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--success)">Distributed Mutual Exclusion and Agreement</h4>
            <p>Consensus is a form of <strong>distributed agreement problem</strong>. In TrustMesh:</p>
            <ul>
              <li>• <strong>Agreement</strong>: All honest nodes must agree on the same decision (YES or NO)</li>
              <li>• <strong>Validity</strong>: The decision must be a value proposed by some node</li>
              <li>• <strong>Termination</strong>: Every correct node must eventually decide — guaranteed by round-based execution</li>
              <li>• <strong>Integrity</strong>: A node decides at most once per round (mutual exclusion on decision-making)</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- ── UNIT 3 ── -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge danger">3</span>
          Unit 3 — Fault Tolerance, Consistency & Recovery
        </div>
        <div class="result-card-body">

          <!-- Fault Models -->
          <div class="info-box">
            <h4 style="color:var(--danger)">Fault Models (Byzantine and Crash)</h4>
            <p>TrustMesh handles two types of faults from the distributed systems fault taxonomy:</p>
            <div class="bft-grid" style="margin-top:0.8rem;">
              <div class="bft-card" style="border-color:rgba(255,71,87,0.2);">
                <div class="bft-label" style="color:var(--danger)">Byzantine Faults</div>
                <div class="bft-value" style="font-size:0.85rem;color:var(--text-secondary)">${bft.byzantine_count} nodes — send incorrect/conflicting votes intentionally. Worst-case fault model.</div>
              </div>
              <div class="bft-card" style="border-color:rgba(255,165,2,0.2);">
                <div class="bft-label" style="color:var(--warning)">Crash Faults</div>
                <div class="bft-value" style="font-size:0.85rem;color:var(--text-secondary)">${bft.crash_fault_count} nodes — stop responding (timeout). Simpler fault model.</div>
              </div>
            </div>
          </div>



          <!-- Voting Protocols -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--danger)">Voting Protocols</h4>
            <p>TrustMesh implements a <strong>weighted voting protocol</strong>, an extension of majority voting:</p>
            <ul>
              <li>• <strong>Raft (Majority Voting)</strong>: Each node gets exactly 1 vote. Decision = simple majority. Threshold: n/2 + 1</li>
              <li>• <strong>TrustMesh (Weighted Voting)</strong>: Each node's vote is multiplied by its sigmoid weight. Decision = side with higher weighted total</li>
              <li>• <strong>Quorum</strong>: Raft requires a fixed quorum. TrustMesh achieves effective quorum through trust weighting — low-trust nodes contribute almost nothing</li>
            </ul>
          </div>

          <!-- Consistency -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--danger)">Consistency and Replica Management</h4>
            <p>Trust scores across rounds must be consistent. TrustMesh ensures:</p>
            <ul>
              <li>• <strong>Data-Centric Consistency</strong>: Trust scores are updated in strict round order (sequential consistency)</li>
              <li>• <strong>Replica Management</strong>: Each node maintains its own trust history — the coordinator holds the "primary replica" of aggregated trust</li>
              <li>• <strong>Consistency Protocol</strong>: EMA update acts as a consistency protocol — T<sub>new</sub> = α × T<sub>old</sub> + (1-α) × S<sub>round</sub> ensures monotonic convergence</li>
            </ul>
          </div>

          <!-- Checkpointing & Recovery -->
          <div class="info-box" style="margin-top:1rem;">
            <h4 style="color:var(--danger)">Checkpointing and Recovery</h4>
            <p>TrustMesh implements a form of <strong>checkpointing</strong> for fault recovery:</p>
            <ul>
              <li>• <strong>Trust History</strong> = checkpoint log — each round's trust score is saved as a checkpoint</li>
              <li>• <strong>Recovery</strong>: If a faulty node starts behaving correctly, its trust gradually recovers via EMA (exponential moving average)</li>
              <li>• <strong>Self-healing</strong>: The system doesn't permanently ban nodes — trust can be rebuilt over time</li>
              <li>• <strong>No single point of failure</strong>: Even if the leader fails, the next highest-trust node takes over (leader re-election)</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Network Topology -->
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge accent">4</span>
          Network Topology — Message Passing Visualization
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.5rem;">
            Each node communicates via <strong>message passing</strong> (no shared memory). Node size = trust score.
            Lines = communication channels between nodes.
            Color: <span style="color:var(--success)">■ Good</span>
            <span style="color:var(--danger)">■ Malicious</span>
            <span style="color:var(--warning)">■ Faulty</span>.
            Leader (highest trust) has a dashed ring.
          </p>
          <div class="topology-canvas-wrap">
            <canvas id="${topologyId}" width="700" height="420"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══ TAB 4: EVALUATION METRICS ═══ -->
    <div class="tab-content" id="tab-metrics">
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge purple">E</span>
          Evaluation Parameters — Raft vs TrustMesh
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1rem;">
            Six key parameters are used to evaluate and compare the two consensus mechanisms.
            All values are computed from the ${data.num_sim_rounds}-round simulation.
          </p>
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th style="color:var(--danger)">Raft</th>
                <th style="color:var(--success)">TrustMesh</th>
              </tr>
            </thead>
            <tbody>${metricsRows}</tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ═══ TAB: ANALYSIS CHARTS (Dynamic from current analysis) ═══ -->
    <div class="tab-content" id="tab-charts">
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge purple">C</span>
          Analysis Charts — Current Configuration
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.5rem;">
            Dynamic charts generated from your current node configuration (${data.num_nodes} nodes, ${data.num_sim_rounds} rounds).
            Node composition: <strong style="color:var(--success)">${bft.good_count} Good</strong>,
            <strong style="color:var(--danger)">${bft.byzantine_count} Malicious</strong>,
            <strong style="color:var(--warning)">${bft.crash_fault_count} Faulty</strong>.
            Leader: <strong style="color:var(--accent-secondary)">Node ${tm.final_leader}</strong>.
          </p>

          <div style="display:grid;grid-template-columns:1fr;gap:2rem;">

            <div class="chart-wrap">
              <h4>TrustMesh vs Raft — Overall Metric Comparison</h4>
              <canvas id="${acOverallId}"></canvas>
            </div>

            <div class="chart-wrap">
              <h4>Consensus Accuracy — Raft vs TrustMesh</h4>
              <canvas id="${acAccuracyId}"></canvas>
            </div>

            <div class="chart-wrap">
              <h4>Trust Score Evolution Over ${data.num_sim_rounds} Rounds</h4>
              <canvas id="${acTrustEvoId}"></canvas>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
              <div class="chart-wrap">
                <h4>Latency Comparison</h4>
                <canvas id="${acLatencyId}"></canvas>
              </div>
              <div class="chart-wrap">
                <h4>Throughput Comparison</h4>
                <canvas id="${acThroughputId}"></canvas>
              </div>
            </div>

            <div class="chart-wrap">
              <h4>Message Overhead</h4>
              <canvas id="${acMsgOverheadId}"></canvas>
            </div>

          </div>
        </div>
      </div>
    </div>

    <!-- ═══ TAB: STATIC CHARTS (Pre-generated PNGs) ═══ -->
    <div class="tab-content" id="tab-static">
      <div class="result-card">
        <div class="result-card-header">
          <span class="step-badge warning">S</span>
          Pre-Generated Analysis Charts (graphs.py)
        </div>
        <div class="result-card-body">
          <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:1.5rem;">
            Publication-quality charts generated by <strong>graphs.py</strong> across multiple malicious-node scenarios
            (0%, 10%, 20%, 30%, 40%, 50%, 60%, 80%). These demonstrate TrustMesh's advantage as adversarial nodes increase.
          </p>

          <div style="display:grid;grid-template-columns:1fr;gap:2rem;">

            <div class="chart-wrap">
              <h4>TrustMesh vs Raft — Overall Comparison</h4>
              <img src="/results/trustmesh_vs_raft_comparison.png" alt="TrustMesh vs Raft Comparison" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first to generate charts.</p>'">
            </div>

            <div class="chart-wrap">
              <h4>Consensus Accuracy Comparison</h4>
              <img src="/results/accuracy_comparison.png" alt="Accuracy Comparison" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
            </div>

            <div class="chart-wrap">
              <h4>Trust Score Evolution</h4>
              <img src="/results/trust_evolution.png" alt="Trust Evolution" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
            </div>

            <div class="chart-wrap">
              <h4>Fault Tolerance Curve</h4>
              <img src="/results/fault_tolerance.png" alt="Fault Tolerance" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
              <div class="chart-wrap">
                <h4>Latency Comparison</h4>
                <img src="/results/latency_comparison.png" alt="Latency Comparison" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
              </div>
              <div class="chart-wrap">
                <h4>Throughput Comparison</h4>
                <img src="/results/throughput.png" alt="Throughput" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
              </div>
            </div>

            <div class="chart-wrap">
              <h4>Message Overhead</h4>
              <img src="/results/message_overhead.png" alt="Message Overhead" style="width:100%;border-radius:8px;margin-top:0.5rem;" onerror="this.parentElement.innerHTML='<p style=color:var(--danger)>Image not found. Run graphs.py first.</p>'">
            </div>

          </div>
        </div>
      </div>
    </div>
  `;

  // ── Render all charts ──
  renderTrustChart(trustChartId, nodes, data.num_sim_rounds);
  renderRoundScoreChart(roundScoreChartId, nodes, data.num_sim_rounds);
  renderSigmoidChart(sigmoidChartId, data.sigmoid_data, nodes);
  renderRoundCompChart(roundCompChartId, data.round_comparison);
  renderTopology(topologyId, nodes, tm.final_leader);
  // Analysis Charts tab — dynamic charts
  renderTrustChart(acTrustEvoId, nodes, data.num_sim_rounds);
  renderBarComparison(acAccuracyId, 'Consensus Accuracy (%)', metrics.consensus_accuracy.raft, metrics.consensus_accuracy.trustmesh);
  renderBarComparison(acLatencyId, 'Avg Latency (ms)', parseFloat(raft.avg_latency), parseFloat(tm.avg_latency));
  renderBarComparison(acThroughputId, 'Decisions/sec', parseFloat(metrics.throughput.raft), parseFloat(metrics.throughput.trustmesh));
  renderBarComparison(acMsgOverheadId, 'Total Messages', parseInt(metrics.message_overhead.raft), parseInt(metrics.message_overhead.trustmesh));
  renderOverallComparison(acOverallId, data);
}

// ══════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════
function switchTab(name) {
  const tabNames = ['consensus','comparison','charts','static','trust','concepts','metrics'];
  tabNames.forEach(t => {
    const btn = document.getElementById('tabbtn-' + t);
    if (btn) btn.classList.toggle('active', t === name);
  });
  document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
  const el = document.getElementById('tab-' + name);
  if (el) el.classList.add('active');
}

// ══════════════════════════════════════════════════════════════
//  CHART RENDERING
// ══════════════════════════════════════════════════════════════
const chartColors = {
  good: '#00d4aa',
  malicious: '#ff4757',
  faulty: '#ffa502',
};
const chartOptions = {
  responsive: true,
  animation: { duration: 800 },
  plugins: {
    legend: { labels: { color: '#9898b8', font: { size: 11, family: 'Inter' } }, position: 'bottom' }
  },
  scales: {
    x: { ticks: { color: '#5a5a7a' }, grid: { color: 'rgba(255,255,255,0.04)' } },
    y: { ticks: { color: '#5a5a7a' }, grid: { color: 'rgba(255,255,255,0.04)' } },
  }
};

function renderTrustChart(canvasId, nodes, numRounds) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const labels = Array.from({length: numRounds + 1}, (_, i) => `R${i}`);
  const datasets = nodes.map(n => ({
    label: `Node ${n.name} — ${n.type} (trust: ${n.trust})`,
    data: n.trust_history,
    borderColor: chartColors[n.classification] || '#888',
    backgroundColor: 'transparent',
    borderWidth: 2.5,
    pointRadius: 3,
    pointBackgroundColor: chartColors[n.classification] || '#888',
    tension: 0.3,
    borderDash: n.classification === 'malicious' ? [6,3] : [],
  }));
  new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        x: { ...chartOptions.scales.x, title: { display: true, text: 'Simulation Round', color: '#5a5a7a' } },
        y: { ...chartOptions.scales.y, title: { display: true, text: 'Trust Score', color: '#5a5a7a' }, min: 0, max: 10 }
      }
    }
  });
}

function renderRoundScoreChart(canvasId, nodes, numRounds) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const labels = Array.from({length: numRounds}, (_, i) => `R${i+1}`);
  const datasets = nodes.map(n => ({
    label: `Node ${n.name} — ${n.type} (trust: ${n.trust})`,
    data: n.round_scores,
    borderColor: chartColors[n.classification] || '#888',
    backgroundColor: (chartColors[n.classification] || '#888') + '30',
    borderWidth: 2,
    pointRadius: 4,
    pointBackgroundColor: chartColors[n.classification] || '#888',
    tension: 0.2,
    borderDash: n.classification === 'malicious' ? [6,3] : [],
  }));
  new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      ...chartOptions,
      scales: {
        ...chartOptions.scales,
        x: { ...chartOptions.scales.x, title: { display: true, text: 'Simulation Round', color: '#5a5a7a' } },
        y: { ...chartOptions.scales.y, title: { display: true, text: 'Round Score (0-10)', color: '#5a5a7a' }, min: 0, max: 10 }
      },
      plugins: {
        ...chartOptions.plugins,
        subtitle: {
          display: true,
          text: 'Each dot shows the nuanced score for that round — NOT binary 0/10',
          color: '#5a5a7a', font: { size: 11 }, padding: { bottom: 10 }
        }
      }
    }
  });
}

function renderSigmoidChart(canvasId, sigmoidData, nodes) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const labels = sigmoidData.map(d => d.trust);
  const weights = sigmoidData.map(d => d.weight);

  // Mark node positions
  const nodePoints = nodes.map(n => ({
    x: n.trust,
    y: n.final_weight,
    name: n.name,
    classification: n.classification,
  }));

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Sigmoid Weight Curve',
          data: weights,
          borderColor: '#a29bfe',
          backgroundColor: 'rgba(162,155,254,0.1)',
          fill: true,
          borderWidth: 3,
          pointRadius: 0,
          tension: 0.4,
        },
        {
          label: 'Node Positions',
          data: nodePoints.map(p => ({ x: p.x, y: p.y })),
          type: 'scatter',
          backgroundColor: nodePoints.map(p => chartColors[p.classification]),
          borderColor: '#fff',
          borderWidth: 1.5,
          pointRadius: 8,
          pointHoverRadius: 10,
        }
      ]
    },
    options: {
      ...chartOptions,
      scales: {
        x: { ...chartOptions.scales.x, title: { display: true, text: 'Trust Score (0-10)', color: '#5a5a7a' }, type: 'linear', min: 0, max: 10 },
        y: { ...chartOptions.scales.y, title: { display: true, text: 'Voting Weight (0-100)', color: '#5a5a7a' }, min: 0, max: 105 }
      }
    }
  });
}

function renderRoundCompChart(canvasId, roundComparison) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const labels = roundComparison.map(rc => `R${rc.round}`);
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Raft Cumulative Accuracy (%)',
          data: roundComparison.map(rc => rc.raft_accuracy),
          borderColor: '#ff4757',
          backgroundColor: 'rgba(255,71,87,0.1)',
          borderWidth: 3,
          pointRadius: 5,
          pointBackgroundColor: '#ff4757',
          tension: 0.3,
          fill: false,
        },
        {
          label: 'TrustMesh Cumulative Accuracy (%)',
          data: roundComparison.map(rc => rc.tm_accuracy),
          borderColor: '#00d4aa',
          backgroundColor: 'rgba(0,212,170,0.1)',
          borderWidth: 3,
          pointRadius: 5,
          pointBackgroundColor: '#00d4aa',
          tension: 0.3,
          fill: false,
        }
      ]
    },
    options: {
      ...chartOptions,
      scales: {
        x: { ...chartOptions.scales.x, title: { display: true, text: 'Simulation Round', color: '#5a5a7a' } },
        y: { ...chartOptions.scales.y, title: { display: true, text: 'Cumulative Accuracy (%)', color: '#5a5a7a' }, min: 0, max: 105 }
      }
    }
  });
}

function renderBarComparison(canvasId, label, raftVal, tmVal) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Raft (Equal Weight)', 'TrustMesh (Trust-Weighted)'],
      datasets: [{
        label: label,
        data: [raftVal, tmVal],
        backgroundColor: ['rgba(255,71,87,0.7)', 'rgba(0,212,170,0.7)'],
        borderColor: ['#ff4757', '#00d4aa'],
        borderWidth: 2,
        borderRadius: 8,
        barPercentage: 0.5,
      }]
    },
    options: {
      ...chartOptions,
      indexAxis: 'x',
      plugins: {
        ...chartOptions.plugins,
        legend: { display: false },
      },
      scales: {
        x: { ...chartOptions.scales.x },
        y: { ...chartOptions.scales.y, title: { display: true, text: label, color: '#5a5a7a' }, beginAtZero: true }
      }
    }
  });
}

function renderOverallComparison(canvasId, data) {
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return;
  const metrics = data.eval_metrics;
  const raft = data.raft;
  const tm = data.trustmesh;

  const raftAccuracy = parseFloat(metrics.consensus_accuracy.raft);
  const tmAccuracy = parseFloat(metrics.consensus_accuracy.trustmesh);
  const raftLatency = parseFloat(raft.avg_latency);
  const tmLatency = parseFloat(tm.avg_latency);
  const raftMsg = parseInt(metrics.message_overhead.raft);
  const tmMsg = parseInt(metrics.message_overhead.trustmesh);
  const raftThroughput = parseFloat(metrics.throughput.raft);
  const tmThroughput = parseFloat(metrics.throughput.trustmesh);

  // Normalize: for 'higher=better' use value/max, for 'lower=better' INVERT so lower value = higher bar
  const maxLatency = Math.max(raftLatency, tmLatency) || 1;
  const minLatency = Math.min(raftLatency, tmLatency) || 1;
  const maxMsg = Math.max(raftMsg, tmMsg) || 1;
  const minMsg = Math.min(raftMsg, tmMsg) || 1;
  const maxThroughput = Math.max(raftThroughput, tmThroughput) || 1;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Accuracy', 'Latency (lower=better)', 'Messages (lower=better)', 'Throughput'],
      datasets: [
        {
          label: 'Raft',
          data: [
            raftAccuracy,
            (minLatency/raftLatency)*100,
            (minMsg/raftMsg)*100,
            (raftThroughput/maxThroughput)*100
          ],
          backgroundColor: 'rgba(255,71,87,0.65)',
          borderColor: '#ff4757',
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: 'TrustMesh',
          data: [
            tmAccuracy,
            (minLatency/tmLatency)*100,
            (minMsg/tmMsg)*100,
            (tmThroughput/maxThroughput)*100
          ],
          backgroundColor: 'rgba(0,212,170,0.65)',
          borderColor: '#00d4aa',
          borderWidth: 2,
          borderRadius: 6,
        }
      ]
    },
    options: {
      ...chartOptions,
      scales: {
        x: { ...chartOptions.scales.x },
        y: { ...chartOptions.scales.y, title: { display: true, text: 'Performance Score (higher=better)', color: '#5a5a7a' }, min: 0, max: 105 }
      }
    }
  });
}

function renderTopology(canvasId, nodes, leader) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(W, H) * 0.35;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0a0a16';
  ctx.fillRect(0, 0, W, H);

  // Compute positions
  const positions = nodes.map((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length - Math.PI / 2;
    return {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      node: n,
    };
  });

  // Draw edges (connections)
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      ctx.beginPath();
      ctx.moveTo(positions[i].x, positions[i].y);
      ctx.lineTo(positions[j].x, positions[j].y);
      ctx.stroke();
    }
  }

  // Draw nodes
  const classColors = { good: '#00d4aa', malicious: '#ff4757', faulty: '#ffa502' };
  positions.forEach(pos => {
    const n = pos.node;
    const nodeRadius = 16 + (n.trust / 10) * 14;
    const color = classColors[n.classification] || '#888';

    // Glow
    const grad = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, nodeRadius * 2);
    grad.addColorStop(0, color + '30');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, nodeRadius * 2, 0, Math.PI * 2);
    ctx.fill();

    // Leader ring
    if (n.name === leader) {
      ctx.strokeStyle = '#48dbfb';
      ctx.lineWidth = 3;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, nodeRadius + 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Node circle
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
    ctx.fill();

    // Border
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(n.name, pos.x, pos.y);

    // Trust label
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText(n.trust, pos.x, pos.y + nodeRadius + 14);
  });

  // Legend
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  let ly = 20;
  [['Good', '#00d4aa'], ['Malicious', '#ff4757'], ['Faulty', '#ffa502'], ['Leader', '#48dbfb']].forEach(([label, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(15, ly - 5, 10, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText(label, 30, ly + 2);
    ly += 18;
  });
}

// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════
buildNodeCards();
loadPreset('boundary');
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("\n🔷 TrustMesh Dashboard")
    print("   Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
