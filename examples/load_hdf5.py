import h5py, numpy as np

with h5py.File('utdtb_v5.h5', 'r') as f:
    # Core sensor and label arrays
    X       = f['train/sensors'][:]          # (898225, 20)  noisy sensors
    X_clean = f['train/sensors_clean'][:]    # (898225, 20)  ground truth
    y       = f['train/RUL'][:]              # (898225,)     mean RUL
    y_lo    = f['train/RUL_lower'][:]        # (898225,)     95% lower
    y_hi    = f['train/RUL_upper'][:]        # (898225,)     95% upper
    pf      = f['train/failure_prob'][:]     # (898225,)     P(fail)
    hi      = f['train/health_index'][:]     # (898225,)
    
    # Environment and causal structure
    env     = f['train/env'][:]              # (898225, 16)  incl. flight_phase
    causal  = f['train/causal_state'][:]     # (898225, 19)  causal node vector
    A       = f['metadata/causal_adjacency'][:]  # (19, 19)  DAG weights
    
    # PINN supervision targets
    pinn_fat = f['train/pinn_res_D_fat'][:]  # (≈897K,)  dD_fat/dcycle
    pinn_crp = f['train/pinn_res_D_crp'][:]
    pinn_cor = f['train/pinn_res_D_cor'][:]
    pinn_th  = f['train/pinn_res_D_th'][:]
    pinn_eff = f['train/pinn_res_eff_c'][:]
    pinn_cr  = f['train/pinn_res_crack_len'][:]
    
    # Fault flags and irregular timestamps
    sfaults  = f['train/sensor_faults'][:]   # (898225, 6)
    ts       = f['train/timestamps'][:]      # (898225,)  unix float64, NaN = dropout
    si       = f['train/sampling_intervals'][:]  # (898225,)  minutes
    
    # Metadata
    sensor_names = f['metadata/sensor_names'][:].astype(str)
    phase_names  = f['metadata/flight_phase_names'][:].astype(str)
    env_cols     = f['metadata/env_col_names'][:].astype(str)
