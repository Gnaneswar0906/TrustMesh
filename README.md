<div align="center">

# 🔗 TrustMesh

### A Trust-Aware Distributed Consensus Framework

Enhancing the **Raft Consensus Algorithm** with **Dynamic Trust-Weighted Voting** for Reliable Consensus in Adversarial Environments.

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Distributed Systems](https://img.shields.io/badge/Distributed-Systems-success?style=for-the-badge)
![Consensus](https://img.shields.io/badge/Consensus-Raft-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</div>

---

# 📖 Overview

**TrustMesh** is a Python-based distributed consensus simulation framework that extends the traditional **Raft Consensus Algorithm** by introducing **dynamic trust-weighted voting**. Instead of treating every node equally, TrustMesh continuously evaluates node behavior and assigns trust scores that influence leader election and consensus decisions.

The framework simulates **honest**, **faulty**, and **malicious** nodes to evaluate the effectiveness of trust-aware consensus under adversarial network conditions.

---

# ✨ Features

- 🔒 Dynamic Trust Score Management
- ⚖️ Trust-Weighted Consensus Voting
- 👑 Trust-Based Leader Election
- 🌐 Honest, Faulty & Malicious Node Simulation
- 📊 Consensus Accuracy Evaluation
- ⚡ Latency & Throughput Analysis
- 📡 Message Overhead Measurement
- 📈 Trust Evolution Visualization
- 🖥️ Interactive Command Line Interface
- 📉 Comparative Performance Analysis with Raft

---

# 🏆 Experimental Results

| Metric | TrustMesh | Traditional Raft |
|:-------|:---------:|:----------------:|
| Consensus Accuracy (60% Malicious Nodes) | **98%** | 22% |
| Consensus Accuracy (80% Malicious Nodes) | **96%** | 0% |
| Fault Tolerance | ✅ High | ❌ Low |
| Latency | Comparable | Comparable |
| Message Overhead | Comparable | Comparable |

### 📌 Highlights

- ✅ Achieved **98% consensus accuracy** with **60% malicious nodes**.
- ✅ Achieved **96% consensus accuracy** with **80% malicious nodes**.
- ✅ Significantly outperformed the traditional Raft algorithm while maintaining similar latency and communication overhead.
- ✅ Improved reliability in highly adversarial distributed environments.

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Libraries | NumPy, Matplotlib |
| Concepts | Distributed Systems, Consensus Algorithms, Trust Management, Network Simulation |

---

## 📂 Project Structure

```text
TrustMesh/
│
├── main.py                 # Main execution file
├── dashboard.py            # Interactive dashboard
├── simulation.py           # Network simulation engine
├── node.py                 # Node behavior implementation
├── trust.py                # Trust score management
├── voting.py               # Trust-weighted voting logic
├── metrics.py              # Performance metrics calculation
├── graphs.py               # Visualization generation
├── requirements.txt        # Project dependencies
│
├── results/
│   ├── accuracy_comparison.png
│   ├── fault_tolerance.png
│   ├── latency_comparison.png
│   ├── message_overhead.png
│   ├── throughput.png
│   ├── trust_evolution.png
│   ├── trustmesh_vs_raft_comparison.png
│   ├── metrics_summary.csv
│   └── report.md
│
└── README.md
```

---

# ⚙️ Installation

```bash
git clone https://github.com/your-username/TrustMesh.git

cd TrustMesh

pip install -r requirements.txt
```

---

# ▶️ Running the Project

Execute the simulation

```bash
python main.py
```

Launch the dashboard

```bash
python dashboard.py
```

---

# 📊 Performance Metrics

The framework evaluates the consensus algorithm using:

- 📌 Consensus Accuracy
- 📌 Latency
- 📌 Throughput
- 📌 Message Overhead
- 📌 Fault Tolerance
- 📌 Trust Evolution

---

# 🧠 How TrustMesh Works

1. Initialize a distributed network.
2. Create honest, faulty, and malicious nodes.
3. Assign initial trust values.
4. Perform trust-aware leader election.
5. Execute weighted consensus voting.
6. Update trust scores after each consensus round.
7. Compare results with the standard Raft algorithm.
8. Generate performance reports and visualizations.

---

# 🚀 Future Enhancements

- 🔹 Byzantine Fault Tolerance (BFT)
- 🔹 Blockchain Network Integration
- 🔹 Dynamic Network Topology Support
- 🔹 AI-Based Trust Prediction
- 🔹 Web-Based Monitoring Dashboard
- 🔹 Real-Time Distributed Deployment

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

Made with ❤️ for Distributed Systems Research

</div>
