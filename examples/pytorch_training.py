from utdtb_v5_torch import UTDTBv5Dataset
from torch.utils.data import DataLoader
import torch, torch.nn as nn

ds_train = UTDTBv5Dataset('utdtb_v5.h5', split='train', window=30, stride=5)
ds_val   = UTDTBv5Dataset('utdtb_v5.h5', split='val',   window=30, stride=5)

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=4)

model = MyTransformer(in_channels=20, env_channels=16, out_dim=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def gaussian_nll_loss(pred_mean, pred_log_var, target):
    """Train against distributional labels with NLL loss."""
    var = torch.exp(pred_log_var).clamp(1e-6)
    return (0.5 * ((pred_mean - target)**2 / var + pred_log_var)).mean()

for epoch in range(50):
    for batch in dl_train:
        sensors  = batch["sensors"]        # (B, 30, 20)
        env      = batch["env"]            # (B, 30, 16)
        fadec    = batch["fadec"]          # (B, 30, 4)   — FADEC signals as features
        sfaults  = batch["sensor_faults"]  # (B, 30, 6)   — fault awareness
        rul_mean = batch["RUL"]            # (B,)
        rul_lo   = batch["RUL_lower"]      # (B,)         — 95% CI supervision
        rul_hi   = batch["RUL_upper"]      # (B,)
        pf       = batch["failure_prob"]   # (B,)         — binary failure head
        
        # Mask NaN sensors (from dropout/stuck faults)
        sensors = torch.nan_to_num(sensors, nan=0.0)
        
        pred_rul, pred_logvar, pred_pf = model(sensors, env)
        
        loss_rul  = gaussian_nll_loss(pred_rul, pred_logvar, rul_mean)
        loss_pf   = nn.BCELoss()(pred_pf, (pf > 0.5).float())
        loss      = loss_rul + 0.3 * loss_pf
        
        optimizer.zero_grad(); loss.backward(); optimizer.step()
