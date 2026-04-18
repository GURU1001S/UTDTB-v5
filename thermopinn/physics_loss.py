"""
physics_loss.py  ·  AeroMRO Digital Twin  ·  v19.19 (Symmetric Physics)
═══════════════════════════════════════════════════════════════════════
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional

class CompositePINNLoss(nn.Module):
    def __init__(self, base_lambdas: Optional[Dict[str, float]] = None):
        super().__init__()
        self.lambdas = {
            "huber": 1.0, "monotonic": 0.10, "health": 0.20,
            "phys_sup": 0.50, "damage_mono": 1.0, "health_mono": 1.0
        }
        if base_lambdas:
            for k in base_lambdas: self.lambdas[k] = base_lambdas[k]

    def set_epoch(self, epoch: int) -> None: pass
    def _safe_mean(self, t: Tensor) -> Tensor:
        valid = t[torch.isfinite(t)]
        return valid.mean() if valid.numel() > 0 else torch.tensor(0.0, device=t.device)

    def compute(self, model_output: Dict[str, Tensor], target_rul_log: Tensor, sensor_batch: Tensor,
                true_health: Optional[Tensor] = None, true_rul_alea: Optional[Tensor] = None,
                true_rul_epis: Optional[Tensor] = None, use_pde: bool = True, max_rul_log: float = 6.2):
        pred_log, true_log = model_output["rul_log"].squeeze(-1).float(), target_rul_log.squeeze(-1).float()
        mask = torch.isfinite(true_log)
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=pred_log.device, requires_grad=True)
            return {"total": zero, "data": zero}

        p, t = pred_log[mask], true_log[mask]
        
        # 🚨 FIX: Removed the 2.0x asymmetric penalty. The outer loop is now fearless.
        L_huber = self._safe_mean(F.smooth_l1_loss(p, t, reduction="none", beta=1.0))

        sidx = torch.argsort(t, descending=True)
        L_mono = self._safe_mean(torch.relu(p[sidx][1:] - p[sidx][:-1]))

        L_health = torch.tensor(0.0, device=p.device)
        if true_health is not None:
            L_health = self._safe_mean(F.binary_cross_entropy_with_logits(
                model_output["health_logit"].squeeze(-1).float()[mask],
                torch.where(torch.isfinite(true_health.squeeze(-1).float()[mask]), true_health.squeeze(-1).float()[mask], torch.full_like(p, 0.5))
            ))

        L_delta_mono = torch.tensor(0.0, device=p.device)
        if "delta" in model_output:
            delta = model_output["delta"].squeeze(-1).float()[mask][sidx]
            L_delta_mono = self._safe_mean(torch.relu(delta[:-1] - delta[1:]))

        L_health_mono = torch.tensor(0.0, device=p.device)
        if "health" in model_output:
            health = model_output["health"].squeeze(-1).float()[mask][sidx]
            L_health_mono = self._safe_mean(torch.relu(health[1:] - health[:-1]))

        L_nll = torch.tensor(0.0, device=p.device)
        L_consistency = torch.tensor(0.0, device=p.device)
        if "rul_log_var" in model_output:
            log_var = model_output["rul_log_var"].squeeze(-1).float()[mask]
            var = torch.exp(log_var.clamp(-5, 2.5)) + 1e-4
            
            L_nll = self._safe_mean(0.5 * (((p - t) ** 2) / var + log_var))
            
            with torch.no_grad(): target_var = ((p - t) ** 2).detach().clamp(min=0.01, max=25.0)
            log_ratio = torch.log(var) - torch.log(target_var + 1e-4)
            L_consistency = torch.where(log_ratio < 0, 2.0 * log_ratio ** 2, 0.5 * log_ratio ** 2).mean()

        L_phys = torch.tensor(0.0, device=p.device)
        if "physics_preds" in model_output and sensor_batch.shape[-1] >= 55:
            L_phys = self._safe_mean(F.mse_loss(model_output["physics_preds"][mask], sensor_batch[:, -1, 36:55][mask], reduction='none'))

        L_tot = (
            self.lambdas["huber"] * L_huber + self.lambdas["monotonic"] * L_mono +
            self.lambdas["health"] * L_health + 
            self.lambdas["phys_sup"] * L_phys + 0.5 * L_nll + 0.10 * L_consistency +
            self.lambdas["damage_mono"] * L_delta_mono + self.lambdas["health_mono"] * L_health_mono
        )
        return {"total": L_tot, "data": L_huber, "mono": L_mono}

class NASAAsymmetricScore(nn.Module):
    def __init__(self, clamp_range: float = 50.0):
        super().__init__()
        self.clamp_range = clamp_range
    @torch.no_grad()
    def forward(self, rul_log_pred: Tensor, rul_log_true: Tensor) -> Tensor:
        d = torch.clamp(torch.expm1(rul_log_pred.float().squeeze(-1)) - torch.expm1(rul_log_true.float().squeeze(-1)), -self.clamp_range, self.clamp_range)
        return torch.where(d < 0, torch.exp(-d / 13.0) - 1.0, torch.exp( d / 10.0) - 1.0).mean()

def log_nasa(rul_log_pred: Tensor, rul_log_true: Tensor) -> float:
    return float(torch.log1p(torch.tensor(NASAAsymmetricScore(clamp_range=50)(rul_log_pred, rul_log_true).item())).item())
