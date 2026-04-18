# Validation Protocols & Quality Assurance
**UTDTB v5.0 — Verification of Physical and Statistical Integrity**

This document details the checks performed to ensure that the synthetic data adheres to the governing laws of thermodynamics and structural mechanics, while maintaining the statistical properties required for robust machine learning.

## 1. Physics Validation Checks
These checks verify that the simulation logic correctly implements the Brayton cycle and degradation kinetics.

| Check | Method | Expected Outcome |
| :--- | :--- | :--- |
| **Thermodynamic Consistency** | Assert $T_3 > T_0$ for all cycles | Monotone with OPR; always passes at $\eta_c \ge 0.60$. |
| **Redline Protection** | $\max(T_4) \le T_{4,max}$ | Enforced at 1750 K by FADEC combustor logic. |
| **Energy Conservation** | $EGT < T_4$ | Turbine loses work; passes by construction. |
| **Health Index Trend** | Avg. $HI$ across lifecycle | Decreasing on average; non-monotone due to maintenance. |
| **Surge Margin Distribution** | $SM > 0$ frequency | ~98% positive in Train; higher stall frequency in Test. |
| **Fatigue Acceleration** | $dD_{fat}$ at Takeoff | $\approx 3.5\times$ Cruise (Confirmed by `takeoff_fat_mult`). |
| **Environmental Impact** | Mean $D_{cor}$ (Coastal vs Desert) | Coastal rates significantly higher due to salt factor. |
| **Fracture Mechanics** | $a < a_{crit}$ until failure | Enforced; fracture at 12mm triggers `is_failed()`. |

---

## 2. Statistical Validation Checks
These checks ensure the dataset is well-distributed and suitable for training predictive models.

| Check | Method | Expected Outcome |
| :--- | :--- | :--- |
| **RUL Distribution** | $hist(RUL)$ | Roughly uniform; partial trajectories bias toward longer lives. |
| **Sensor Data Quality** | `nanmean` per split | Train $\approx$ 0.3%, Val $\approx$ 0.45%, Test $\approx$ 0.9%. |
| **EoL Convergence** | $HI$ at final cycle | Should be $< 0.02$ for non-truncated (full) trajectories. |
| **Rare Event Frequency** | `event_flag.mean()` | ~0.5–2% of rows contain rare physical events. |
| **Cross-Engine Baseline** | Mean `cross_delta_EGT` | Healthy fleet baseline centers at zero. |
| **Uncertainty Decay** | $\sigma_{total} / RUL$ | Relative uncertainty decreases as engine approaches failure. |

---

## 3. Visualization Samples
To manually verify these protocols, researchers are encouraged to use the plotting utilities in `examples/visualization_demo.py` to generate:
- **T-s Diagrams:** Verifying the thermodynamic state changes.
- **Degradation Slopes:** Visualizing the interaction between Creep, Fatigue, and Corrosion.
- **RUL Histograms:** Checking the balance of the dataset splits.
