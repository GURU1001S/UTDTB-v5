# UTDTB v5.0 — Dataset Specification
**Universal Turbofan Digital Twin Benchmark (BEAST Mode)**

## 1. Overview
The UTDTB v5.0 is a synthetic, physics-grounded dataset simulating the complete degradation lifecycle of a fleet of turbofan engines. Unlike statistical datasets, every row in UTDTB is generated via a closed-form physical model including Brayton thermodynamics and structural mechanics.

### Key Metrics
- **Total Rows:** 1,109,391 (Reference Run)
- **Engine Count:** ~1,600 across splits
- **Sampling:** 1 row = 1 flight cycle
- **Precision:** `float32` for sensors, `float64` for timestamps

---

## 2. Feature Channels

### 2.1 Observable Sensors (20 Channels)
These represent the "noisy" telemetry data available to the FADEC and maintenance teams.

| Index | Name | Symbol | Unit | Description |
|:---|:---|:---|:---|:---|
| 0 | T2 | $T_2$ | K | Fan inlet temperature |
| 1 | T24 | $T_{24}$ | K | LPC outlet temperature |
| 2 | T30 | $T_{30}$ | K | HPC outlet temperature |
| 3 | T50 | $T_{50}$ | K | Turbine outlet temperature (EGT proxy) |
| 4 | P2 | $P_2$ | Pa | Fan inlet pressure |
| 5 | P15 | $P_{15}$ | Pa | Fan exit / bypass duct pressure |
| 6 | P30 | $P_{30}$ | Pa | HPC exit pressure |
| 7 | Nf | $N_f$ | RPM | Fan speed |
| 8 | Nc | $N_c$ | RPM | Core (HPC) speed |
| 9 | EPR | $EPR$ | — | Engine pressure ratio ($P_5/P_2$) |
| 10 | Ps30 | $Ps_{30}$ | Pa | Static pressure at HPC exit |
| 11 | phi | $\phi$ | kg/s | Fuel flow rate |
| 12 | NRc | $NR_c$ | RPM | Corrected core speed |
| 13 | BPR | $BPR$ | — | Bypass ratio |
| 14 | vib_rms | $\bar{a}_g$ | V | Vibration RMS (Mechanical imbalance) |
| 15 | oil_temp| $T_{oil}$ | K | Lubrication oil temperature |
| 16 | EGT | $EGT$ | K | Direct exhaust gas temperature |
| 17 | SM | $SM$ | — | Compressor surge margin |
| 18 | dEGT | $\Delta EGT$ | K | Paired-engine EGT differential |
| 19 | dRPM | $\Delta RPM$ | RPM | Paired-engine RPM differential |

### 2.2 Environment & Context (16 Channels)
Stored in the `env` array. These define the operational envelope.

- **Indices 0-4:** Altitude, Throttle, Humidity, $\Delta T_{ISA}$, Salt Factor.
- **Index 8:** `flight_phase` (0=Taxi, 1=Takeoff, 2=Climb, 3=Cruise, 4=Descent, 5=Landing).
- **Index 14:** `time_since_maint` (Cycles since last service).

---

## 3. Ground Truth Labels

### 3.1 RUL Distribution
UTDTB v5 does not provide a single RUL point. It provides the full distribution:
- `RUL`: The true cycles remaining.
- `RUL_std`: Combined aleatoric and epistemic uncertainty.
- `failure_prob`: Logistic probability of failure based on Health Index (HI).
- `RUL_ci05` to `RUL_ci95`: Percentile-based confidence intervals.

### 3.2 PINN Supervision (Physics Residuals)
For Physics-Informed Neural Networks, we provide the derivative of degradation states ($dX/dN$):
- `pinn_res_D_fat`: Fatigue damage increment.
- `pinn_res_D_crp`: Creep damage increment.
- `pinn_res_eff_c`: Efficiency loss increment.

---

## 4. Domain Shift Configuration
To test model robustness, the `test` split contains more aggressive noise and failure modes than the `train` split.

| Parameter | Train | Test |
|:---|:---|:---|
| **Sensor Dropout** | 0.3% | 0.9% |
| **Drift Fault Prob** | $2\times10^{-4}$ | $5\times10^{-4}$ |
| **Bird Strike Prob** | $1\times10^{-4}$ | $2\times10^{-4}$ |
| **Corrosion Rate** | Baseline | $1.8\times$ Baseline |

---

## 5. Causal Structure
The dataset is governed by a 19-node Directed Acyclic Graph (DAG). 
- **Exogenous nodes:** Altitude, Humidity, Throttle.
- **Latent states:** $D_{fatigue}$, $D_{creep}$, $D_{corrosion}$.
- **Observable nodes:** Sensor suite.

The adjacency matrix is available in `metadata/causal_adjacency`.
