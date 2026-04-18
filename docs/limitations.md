# Known Limitations and Research Trade-offs
**UTDTB v5.0 — Technical Constraints & Simplifications**

The UTDTB v5.0 benchmark is designed to balance physical fidelity with computational scalability. Consequently, the following eight simplifications represent deliberate trade-offs made during the development of the generation pipeline.

---

## 1. Steady-State Thermodynamic Assumption
Each flight cycle solves the Brayton equations at a single representative operating point. The model does not integrate within-flight transient dynamics (e.g., warm-up, acceleration/deceleration, or surge-recovery oscillations) via a high-frequency (per-second) ODE solver. The thermal mass ODE operates at cycle-level timesteps (~60 minutes).

## 2. Simplified Compressor Map
The compressor performance is modeled using a 3-speed-line parabolic approximation. While this effectively ranks operating conditions, a real-world compressor map utilizes 7–12 speed lines and accounts for complex stall cells and stage-by-stage interactions. As a result, non-monotone efficiency contours typical of test-cell data are not perfectly reproduced.

## 3. Absence of High-Fidelity Mechanical Dynamics
Vibration is modeled as a proxy function of cumulative fatigue damage and mechanical imbalance. The dataset lacks true high-frequency bearing fault signatures (e.g., ball pass frequency, cage frequency, or race defect patterns) and does not model catastrophic mechanical events like Fan-Blade-Off (FBO).

## 4. Uniform Corrosion Modeling
The Arrhenius model treats the hot section as a homogeneous surface. In reality, hot corrosion is highly localized, nucleating at grain boundaries and cooling holes. This non-uniform propagation creates stress concentrations that accelerate fatigue asymmetrically—a phenomenon currently approximated by a global corrosion scalar.

## 5. Single-Spool Degradation Coupling
Degradation is tracked using scalar efficiency terms ($\eta_c$ and $\eta_t$). A multi-spool architecture (two-spool or three-spool) would allow for independent degradation of the LPC, HPC, HPT, and LPT, which would yield richer cross-sensor correlation patterns for diagnostic tasks.

## 6. Synthetic Cross-Engine Deltas
$\Delta EGT$ and $\Delta RPM$ are computed as the difference between a specific engine’s state and normally distributed reference values. The dataset does not simulate a true "physical pair" in parallel; therefore, the health trajectory of the paired engine is not explicitly tracked.

## 7. Combustion Instability Modes
The simulation excludes fuel nozzle coking and combustor liner cracking. Advanced combustion instability modes, such as thermoacoustic oscillations and lean blowout precursors, are absent, despite being significant drivers for MRO (Maintenance, Repair, and Overhaul) actions in high-temperature operations.

## 8. Independent Manufacturing Variability
Manufacturing tolerances for initial efficiency, mass flow, and spool speed are modeled as independent log-normal distributions. In reality, these parameters are often correlated (e.g., compressor tip clearance directly correlates with mass flow and efficiency scatter).

---

> **Note to Researchers:** These limitations provide fertile ground for future iterations of the UTDTB benchmark and should be considered when interpreting the generalizability of models trained on this data.
