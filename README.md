# UTDTB v5.0 — Universal Turbofan Digital Twin Benchmark
> **BEAST Mode Architecture & ThermoPINN Baseline**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

UTDTB v5.0 is a massive, physics-grounded dataset designed to bridge the "scale gap" in turbofan prognostics. It simulates 1.1M+ flight cycles across a global fleet, providing high-fidelity signals for RUL regression, causal inference, and Physics-Informed Neural Networks (PINNs).

---

## 🧪 Ablation Study Summary
We conducted **25+ controlled experiments** across 7 categories to isolate the contribution of architectural components and robustness mechanisms in our baseline model, **ThermoPINN**.

#### 🧩 1. Physics Constraint Contribution
Removing physics-informed loss terms degraded performance (RMSE: 42.9 → 45.2). Physics priors improve accuracy but do not strictly guarantee physical validity.
![Physics Ablation](results/ablation/P_physics_ablation.png)

#### 📉 2. Dimensionality Robustness (Sensor Pruning)
Performance remained nearly constant during aggressive pruning (55D → 18D). The model relies heavily on a core subset of dominant sensors, maintaining an RMSE of ~124.7 even when stripped down past the N-CMAPSS equivalent baseline (22D). 
![Dimensionality Stress](results/ablation/S_dimensionality_stress.png)

#### 🔁 3. Meta-Learning Depth (Few-Shot Adaptation)
Optimal performance is observed early in the adaptation phase. The model reaches a **Pareto optimum at $k = 2$ shots** (minimizing the NASA asymmetric score) and hits its lowest RMSE (~95 cycles) at $k = 3$. Adapting beyond $k \ge 5$ destabilizes the representations, leading to **catastrophic forgetting** by $k = 7$ where RMSE violently spikes to ~287.
![K-Shot Adaptation](results/ablation/K_kshot_adaptation.png)

#### 🎲 4. Uncertainty Calibration & Efficiency
MC Dropout improves in-distribution calibration (ECE: 0.18) but suffers from **epistemic uncertainty deflation** under Out-of-Distribution (OOD) shift.
![Uncertainty Calibration](results/ablation/U_uncertainty_calibration.png)


#### 🏗️ 5. Core Architecture Comparison (Accuracy vs. Calibration)
Comparing architectures across adaptation steps reveals a critical trade-off. While standard data-driven models (Transformer, LSTM) improve their RMSE during extended adaptation, their calibration (ECE) severely degrades, making them overconfident. **ThermoPINN** maintains stable, well-calibrated uncertainty bounds (ECE ~0.20), but suffers from significant RMSE instability if fine-tuned too aggressively.
![Architecture Ablation](results/ablation/A_architecture_ablation.png)


#### ⚡ 6. Computational Efficiency & Deployment Viability
An analysis of model size, inference latency, and memory footprint reveals that ThermoPINN is highly optimized for edge deployment. 
* **Result:** ThermoPINN requires **~48% less Peak VRAM** (82.7 MB) and fewer parameters (2.98M) compared to a standard sequence-based LSTM (160.4 MB / 3.93M). 
* **Insight:** Integrating explicit physics constraints allows the network to remain lightweight without bloating the parameter count. While there is a marginal latency trade-off during inference (8.58 ms vs. 5.88 ms for LSTM), the architecture remains highly viable for real-time digital twin monitoring on constrained hardware.
![Efficiency Plot](results/ablation/C_efficiency.png)

#### 🧩 7. Feature Group Contribution
Evaluating discrete feature sets reveals that adding auxiliary data (environment variables, physics states, cross-engine signals) on top of the base sensor suite yields no measurable improvement in predictive accuracy. This indicates the base telemetry already captures the maximum utilizable variance, or the architecture is suffering from feature collapse and ignoring the auxiliary inputs.
![Feature Ablation](results/ablation/D_feature_ablation.png)

---

## 📊 Dataset Reference: BEAST Mode
The benchmark provides over **1.1 million rows** generated via transient Brayton-cycle ODEs.

### Split Specifications (Reference Run)
| Split | Engines | Rows | Domain Characteristics |
| :--- | :--- | :--- | :--- |
| **Train** | **1,300** | 898,225 | Baseline noise and faults |
| **Val** | **150** | 107,921 | +50% Sensor Dropout, +30% Drift Faults |
| **Test** | **150** | 103,245 | +200% Dropout, +150% Drift, +80% Bird Strike |
| **Total** | **1,600** | **1,109,391** | **Global Fleet Simulation** |

### Scale Configuration Table
| Mode | $n_{train}$ | $n_{test}$ | Max Cycles | ~Rows | ~HDF5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QUICK** | 400 | 50 | 800 | ~500K | 200MB |
| **MEDIUM**| 2,000 | 300 | 1,000 | ~1.5M | 600MB |
| **BEAST** | **16,000** | **2,000** | **1,200** | **~16M** | **6GB** |

---

## ⚔️ UTDTB v5.0 vs. NASA N-CMAPSS
| Property | N-CMAPSS (DS002/006) | UTDTB v5.0 BEAST |
| :--- | :--- | :--- |
| **Physics Model** | Steady-state | **Transient Brayton + Thermal ODE** |
| **Degradation** | 1 mode (implicit) | **4 explicit (Fatigue, Creep, Corros., Thermal)**|
| **Causal Graph** | None | **19-node DAG (38 edges)** |
| **RUL Labels** | Point estimate | **Distributional (Mean, Std, CI, Failure Prob)** |
| **Events** | None | **10+ (Bird strike, Stall, Fuel contam, etc.)** |
| **Timestamps** | None | **Irregular (~60 min) with ACARS dropout** |

---

## 🛠️ Benchmark Tasks & Methodology
UTDTB v5.0 is designed to evaluate multiple AI/ML disciplines.

| Task | Methodology | Key Inputs |
| :--- | :--- | :--- |
| **RUL Regression** | Transformers / TCN | `sensors (W, 20)` |
| **Probabilistic RUL** | BNN / Conformal Prediction | `RUL_mean`, `RUL_std` |
| **PINN Research** | Physics-Regularized Loss | `pinn_res_*` signals |
| **Causal Inference** | DAG-GNN / NOTEARS | `causal_state`, `causal_adjacency` |
| **Domain Adaptation** | DANN / CORAL | `train` → `test` shift |
| **Anomaly Detection** | Autoencoders / OCSVM | `event_flag`, `sensor_faults` |

---

## 📂 Project Structure
- `thermopinn/`: Core model architecture and physics loss functions.
- `generator/`: The UTDTB v5 engine generation pipeline.
- `docs/`: Technical derivations, causal graph specs, and validation protocols.
- `results/ablation/`: High-resolution plots for all 25+ experiments.

## 📜 Citation
```bibtex
@dataset{utdtb_v5,
  title={UTDTB v5: Universal Turbofan Digital Twin Benchmark},
  author={Guru Prasaath S.},
  year={2026}
}
