### HDF5 Folder Hierarchy
The benchmark is structured in a single HDF5 file (`utdtb_v5.h5`) for high-performance data retrieval.

| Path | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `metadata/sensor_names` | (20,) | string | Names of the 20 observable sensors |
| `metadata/causal_nodes` | (19,) | string | Names of latent causal state nodes |
| `metadata/causal_adjacency`| (19,19) | float32 | Adjacency matrix for the causal DAG |
| `train/sensors` | (898225, 20) | float32 | Noisy sensor telemetry (Input) |
| `train/sensors_clean` | (898225, 20) | float32 | Ground truth sensors (No noise/bias) |
| `train/degrad` | (898225, 12) | float32 | Physical degradation parameters |
| `train/env` | (898225, 16) | float32 | Operational & environmental context |
| `train/causal_state` | (898225, 19) | float32 | Latent causal engine states |
| `train/RUL` | (898225,) | float32 | Ground truth Remaining Useful Life |
| `train/RUL_std` | (898225,) | float32 | Total uncertainty (Alea + Epi) |
| `train/failure_prob` | (898225,) | float32 | Logistic probability of failure |
| `train/pinn_res_*` | (≈897K,) | float32 | Physics residuals ($dX/dN$) for PINNs |
| `train/RUL_ci*` | (898225,) | float32 | Confidence intervals (05, 10, 25, 75, 90, 95) |


### Split Domain Shift Parameters
These parameters define the distribution shift between the training set and the test set to evaluate model robustness.

| Parameter | Train | Val | Test (Shift Severity) |
| :--- | :--- | :--- | :--- |
| **Missing Probability** | 0.003 | 0.0045 | 0.009 (**×3.0**) |
| **Stuck Sensor Prob** | 8e-4 | 8e-4 | 1.6e-3 (**×2.0**) |
| **Drift Fault Prob** | 2e-4 | 2.6e-4 | 5e-4 (**×2.5**) |
| **Bird Strike Prob** | 1e-4 | 1e-4 | 2e-4 (**×2.0**) |
| **Stall Probability** | 8e-5 | 8e-5 | 1.2e-4 (**×1.5**) |
| **Fuel Contamination** | 3e-5 | 3e-5 | 6e-5 (**×2.0**) |
| **Corrosion Rate ($k_{cor}$)**| Baseline | Baseline | Baseline **×1.8** |
| **ACARS Dropout Prob** | 0.008 | 0.008 | 0.016 (**×2.0**) |


### Observable Sensor Channels (20)
Standard telemetry measured by the Digital Twin's sensor layer.

| # | Feature Name | Symbol | Unit | Physical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| 0 | **T2** | $T_2$ | K | Fan inlet temperature |
| 1 | **T24** | $T_{24}$ | K | LPC outlet temperature |
| 2 | **T30** | $T_{30}$ | K | HPC outlet temperature |
| 3 | **T50** | $T_{50}$ | K | Turbine outlet temperature (EGT proxy) |
| 4 | **P2** | $P_2$ | Pa | Fan inlet pressure |
| 5 | **P15** | $P_{15}$ | Pa | Fan exit / bypass duct pressure |
| 6 | **P30** | $P_{30}$ | Pa | HPC exit pressure (Fouling indicator) |
| 7 | **Nf** | $N_f$ | RPM | Fan speed |
| 8 | **Nc** | $N_c$ | RPM | Core (HPC) speed |
| 9 | **EPR** | EPR | — | Engine pressure ratio ($P_5/P_2$) |
| 10 | **Ps30** | $Ps_{30}$ | Pa | Static pressure at HPC exit |
| 11 | **phi** | $\phi$ | kg/s | Fuel flow rate |
| 12 | **NRc** | $NR_c$ | RPM | Corrected core speed |
| 13 | **BPR** | BPR | — | Bypass ratio |
| 14 | **vib_rms** | $\bar{a}_g$ | V | Vibration RMS (Mechanical imbalance) |
| 15 | **oil_temp** | $T_{oil}$ | K | Lubrication oil temperature |
| 16 | **EGT_direct** | EGT | K | Direct exhaust gas temperature |
| 17 | **surge_margin** | SM | — | Distance to surge line (Negative = Stall) |
| 18 | **cross_delta_EGT** | $\Delta EGT$ | K | Paired-engine EGT differential |
| 19 | **cross_delta_RPM** | $\Delta RPM$ | RPM | Paired-engine RPM differential |


### Environment & Operational Variables (16)
Contextual variables defining the flight envelope and ambient conditions.

| # | Feature Name | Unit | Physical Meaning |
| :--- | :--- | :--- | :--- |
| 0 | **altitude** | m | Cruise altitude (Sets $P_0, T_0$) |
| 1 | **throttle** | 0-1 | Demanded thrust lever angle |
| 2 | **humidity** | 0-1 | Relative humidity (Corrosion driver) |
| 3 | **dT_ISA** | K | ISA temperature deviation |
| 4 | **salt_factor** | 0-1 | Coastal route salt exposure index |
| 5 | **throttle_eff** | 0-1 | FADEC-effective throttle post-lag |
| 6 | **bleed_flag** | binary | Compressor bleed valve state |
| 7 | **event_present** | binary | 1 if any rare event occurred this cycle |
| 8 | **flight_phase** | 0-5 | 0:Taxi, 1:Takeoff, 2:Climb, 3:Cruise, 4:Descent, 5:Landing |
| 9 | **route_dist** | nm | Total flight distance |
| 10 | **flight_dur** | min | Estimated flight time |
| 11 | **airport_alt** | ft | Departure/arrival airport elevation |
| 12 | **sand_index** | 0-1 | Sand/dust concentration |
| 13 | **salt_index** | 0-1 | Route-specific salt concentration |
| 14 | **maint_age** | cycles | Cycles since last maintenance action |
| 15 | **thermal_delta** | K | Phase-transition thermal shock ($\Delta T$) |


### Latent Causal Engine States (19)
Hidden ground-truth states governing the engine's internal physics.

| # | Feature Name | Unit | Physical Meaning |
| :--- | :--- | :--- | :--- |
| 0 | **T4** | K | Turbine entry temperature (Primary load driver) |
| 1 | **RPM** | RPM | Normalized shaft speed |
| 5 | **D_fat** | 0-1 | Cumulative fatigue damage (Paris-law) |
| 6 | **D_crp** | 0-1 | Cumulative creep damage (Norton-Bailey) |
| 7 | **D_cor** | 0-1 | Cumulative corrosion damage (Arrhenius) |
| 8 | **D_th** | 0-1 | Cumulative thermal fatigue (Coffin-Manson) |
| 9 | **eff_c** | 0-1 | Compressor isentropic efficiency |
| 10 | **eff_t** | 0-1 | Turbine isentropic efficiency |
| 11 | **crack_len** | m | Blade crack length (Critical at 12mm) |
| 12 | **EGT_true** | K | Pre-sensor-noise EGT |
| 13 | **vib_true** | g | Mechanical imbalance + bearing fault proxy |
| 15 | **RUL** | cycles | Remaining Useful Life (Ground Truth) |
| 17 | **fuel_contam** | binary | Active fuel contamination state |
| 18 | **dEGT_true** | K | Pre-sensor-noise paired-engine delta |
