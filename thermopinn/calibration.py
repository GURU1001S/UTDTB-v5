"""
calibration.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Utility module for Uncertainty Quantification (UQ) and Expected Calibration 
Error (ECE) metrics for the ThermoPINN ablation suite.
"""

import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.amp import autocast

class CalibrationEvaluator:
    """
    Computes the Expected Calibration Error (ECE) for regression tasks.
    Evaluates how closely the predicted confidence intervals match 
    the actual empirical coverage of the model.
    """
    def __init__(self, n_bins=20):
        self.n_bins = n_bins
        self.preds = []
        self.stds = []
        self.trues = []
        self.normal = Normal(0, 1)

    def reset(self):
        self.preds = []
        self.stds = []
        self.trues = []

    def update(self, mean_log, std, true_log):
        self.preds.append(mean_log.detach().cpu())
        self.stds.append(std.detach().cpu())
        self.trues.append(true_log.detach().cpu())

    def summary(self):
        if not self.preds:
            return {"ece": 1.0}
            
        preds = torch.cat(self.preds).flatten()
        stds = torch.cat(self.stds).flatten()
        trues = torch.cat(self.trues).flatten()

        ece = 0.0
        # Sweep through target confidence levels (e.g., 5% to 95%)
        confidences = np.linspace(0.05, 0.95, self.n_bins)
        emp_coverages = []

        for conf in confidences:
            # Calculate the z-score for two-tailed confidence intervals
            z = self.normal.icdf(torch.tensor(0.5 + conf / 2.0)).item()
            lower = preds - z * stds
            upper = preds + z * stds
            
            # Calculate actual empirical coverage
            cov = ((trues >= lower) & (trues <= upper)).float().mean().item()
            emp_coverages.append(cov)
            
            # Accumulate the absolute error between expected and actual
            ece += abs(cov - conf)

        ece /= self.n_bins

        return {
            "ece": float(ece),
            "reliability_diagram": {
                "expected": confidences.tolist(),
                "actual_coverage": emp_coverages
            }
        }

class MCDropoutPredictor:
    """
    Wraps a base model to perform Monte Carlo Dropout inference.
    Separates total uncertainty into Epistemic (model doubt) 
    and Aleatoric (data noise) variance.
    """
    def __init__(self, model, n_passes=10, device='cuda'):
        self.model = model
        self.n_passes = n_passes
        self.device = device

    def predict(self, x, op_setting=None, event_flag=None):
        # Enable dropout layers while keeping BatchNorm frozen
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()

        preds = []
        log_vars = []
        
        with torch.no_grad():
            for _ in range(self.n_passes):
                with autocast('cuda'):
                    out = self.model(x, op_setting=op_setting, event_flag=event_flag)
                preds.append(out["rul_log"])
                log_vars.append(out["rul_log_var"])

        preds = torch.stack(preds)
        log_vars = torch.stack(log_vars)

        mean_log = preds.mean(dim=0)
        epist = preds.std(dim=0)
        aleat = torch.exp(0.5 * log_vars.mean(dim=0))

        # Return model to standard evaluation mode
        self.model.eval()
        
        return mean_log, epist, aleat
