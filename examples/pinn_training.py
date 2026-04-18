import h5py, torch, numpy as np

with h5py.File('utdtb_v5.h5', 'r') as f:
    g         = f['train']
    sensors   = torch.tensor(np.nan_to_num(g['sensors'][:]),    torch.float32)
    degrad    = torch.tensor(g['degrad'][:],                     torch.float32)
    res_D_fat = torch.tensor(g['pinn_res_D_fat'][:],            torch.float32)
    res_D_crp = torch.tensor(g['pinn_res_D_crp'][:],            torch.float32)
    res_eff_c = torch.tensor(g['pinn_res_eff_c'][:],            torch.float32)

def pinn_loss(pred_states, res_targets, lambda_phys=0.1):
    """
    pred_states: (N, 6) model predictions of [D_fat, D_crp, D_cor, D_th, eff_c, crack_len]
    res_targets: (N-1, 6) physics residuals dX/dN from the simulator
    """
    pred_res  = pred_states[1:] - pred_states[:-1]   # predicted increments
    phys_loss = torch.mean((pred_res - res_targets)**2)
    return lambda_phys * phys_loss

# In training loop:
pred_degrad = physics_model(sensors_window)
loss_data   = mse(pred_degrad[:, 4], degrad[:, 4])    # eff_c prediction
loss_phys   = pinn_loss(pred_degrad, torch.stack([res_D_fat, res_D_crp,
                                                   ..., res_eff_c], dim=1))
total_loss  = loss_data + loss_phys
