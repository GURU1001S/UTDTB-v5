# UTDTB v5.0 Research Tasks & Benchmarking

UTDTB is designed to support a wide range of prognostic and diagnostic research tasks. Below is the mapping of dataset columns to specific research methodologies.

## Research Task Table

| Task | UTDTB Columns Used | Methodology Examples |
| :--- | :--- | :--- |
| **RUL Regression** | `sensors`, `RUL`, `RUL_lower`, `RUL_upper` | LSTM, Transformer, TCN |
| **Probabilistic RUL** | `RUL_mean`, `RUL_std`, `RUL_alea`, `RUL_epi`, `failure_prob` | NLL loss, Conformal prediction, BNN |
| **Physics-informed NNs** | `pinn_res_*` (fat, crp, cor, th, eff_c, crack_len) | Soft/hard physics constraints in loss |
| **Causal Inference** | `causal_state`, `causal_adjacency` | GNN, DAG-GNN, NOTEARS |
| **Domain Adaptation** | Train → Test shift in `missing_prob`, `drift_fault_prob` | DANN, MMD, CORAL |
| **Anomaly Detection** | `sensor_faults`, `event_flag`, `maint_flag` | Autoencoder, Isolation Forest, OCSVM |
| **Fault Diagnosis** | `maint_type`, `event_log` (bird strike, stall, FOD, etc.) | Multi-class classifier, Sequence labeling |
| **Digital Twin Calibration**| `sensors_clean` vs `sensors` (noisy) | Kalman Filter, Particle Filter, UKF |
| **Flight-Phase Modeling** | `env[:,8]` (phase), `env[:,14]` (time_since_maint) | Phase-conditioned model, Multi-task learning |
| **Cross-Engine Analysis** | `sensors[:,18:19]` (delta EGT/RPM), `pair_id` | Paired engine models, Twin anomaly detection |
