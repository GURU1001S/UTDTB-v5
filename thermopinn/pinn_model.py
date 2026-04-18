"""
pinn_model.py  ·  AeroMRO Digital Twin  ·  v19.18 (Certification Grade)
═══════════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing import Dict, Optional

class DilatedResBlock(nn.Module):
    def __init__(self, d: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        # 🚨 FIX: GroupNorm is immune to small-batch shifting during MAML adaptation
        self.norm1 = nn.GroupNorm(8, d) 
        self.conv2 = nn.Conv1d(d, d, 1)
        self.norm2 = nn.GroupNorm(8, d)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(x + self.drop(self.norm2(self.conv2(h))))

class TriStreamTCN(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        d2, d4 = d // 2, d // 4
        self.s_proj  = nn.Conv1d(20, d, 1)
        self.s_blks  = nn.ModuleList([DilatedResBlock(d, 2**i, dropout) for i in range(4)])
        self.p_proj  = nn.Conv1d(19, d2, 1)
        self.p_blks  = nn.ModuleList([DilatedResBlock(d2, 2**i, dropout) for i in range(2)])
        self.p_up    = nn.Conv1d(d2, d, 1)
        self.e_enc   = nn.Sequential(nn.Conv1d(16, d4, 1), nn.GELU(), nn.Conv1d(d4, d2, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(d + d + d2, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:    
        s, e, p = x[:, :, :20].transpose(1, 2), x[:, :, 20:36].transpose(1, 2), x[:, :, 36:55].transpose(1, 2)
        hs = self.s_proj(s)
        for blk in self.s_blks: hs = blk(hs)
        hp = self.p_proj(p)
        for blk in self.p_blks: hp = blk(hp)
        hp = self.p_up(hp)
        return self.fuse(torch.cat([hs, hp, self.e_enc(e)], dim=1).transpose(1, 2))

class PhysicsGate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(19, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)
    def forward(self, feat: Tensor, phys_raw: Tensor) -> Tensor:
        return self.norm(feat * self.gate(phys_raw[:, -1, :]).unsqueeze(1))

class DualPathTemporal(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attn_w = nn.Sequential(nn.Linear(d, d // 4), nn.Tanh(), nn.Linear(d // 4, 1))
        self.fuse   = nn.Sequential(nn.Linear(d * 2, d), nn.LayerNorm(d), nn.GELU())
    def forward(self, enc: Tensor) -> Tensor:   
        path_a, recent = enc[:, -1, :], enc[:, -5:, :]              
        return self.fuse(torch.cat([path_a, (recent * torch.softmax(self.attn_w(recent), dim=1)).sum(1)], dim=-1))  

class TransformerEncoder(nn.Module):
    def __init__(self, d: int, nhead: int, ff: int, layers: int, dropout: float):
        super().__init__()
        self.enc  = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff, dropout=dropout, batch_first=True, norm_first=True) 
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
    def forward(self, x: Tensor) -> Tensor:
        for lyr in self.enc:
            with sdpa_kernel(SDPBackend.MATH): x = lyr(x)
        return self.norm(x)

class PINNModel(nn.Module):
    def __init__(
        self, max_rul: float = 500.0, n_sensors: int = 55, conv_channels: int = 256, 
        gru_hidden: int = 512, head_hidden: int = 128, dropout: float = 0.30,
        n_op_settings: int = 32, n_events: int = 10, mean_rul_log: float = 5.50,   
    ):
        super().__init__()
        d = conv_channels
        self.max_rul_log, self.mean_rul_log = math.log1p(max_rul), mean_rul_log
        self.encoder, self.physics_gate, self.dual_path = TriStreamTCN(d, dropout), PhysicsGate(d), DualPathTemporal(d)
        
        self.task_transformer = TransformerEncoder(d, 8, gru_hidden, 2, dropout)
        self.global_task_prior = nn.Parameter(torch.zeros(1, d))
        
        for lyr in self.task_transformer.enc:
            if hasattr(lyr, 'linear2'):
                nn.init.zeros_(lyr.linear2.weight); nn.init.zeros_(lyr.linear2.bias)
        
        self.gamma_head, self.beta_head = nn.Linear(d, d), nn.Linear(d, d)
        nn.init.xavier_uniform_(self.gamma_head.weight, gain=0.05); nn.init.zeros_(self.gamma_head.bias)
        nn.init.xavier_uniform_(self.beta_head.weight, gain=0.05); nn.init.zeros_(self.beta_head.bias)
        self.film_alpha = nn.Parameter(torch.tensor(-2.0))  

        self.degradation_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(head_hidden, 1))
        nn.init.constant_(self.degradation_head[-1].bias, -2.0) 

        # 🚨 The Severed Baseline: Only looks at Engine ID (z_exp)
        self.baseline_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        nn.init.constant_(self.baseline_head[-1].bias, mean_rul_log) 
        
        self.task_to_rul = nn.Linear(d, 1)
        nn.init.zeros_(self.task_to_rul.weight); nn.init.zeros_(self.task_to_rul.bias)

        self.health_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        self.aleatoric_head = nn.Sequential(nn.Linear(d + 20, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        self.epistemic_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        
        nn.init.constant_(self.aleatoric_head[-1].bias, 0.5)
        nn.init.constant_(self.epistemic_head[-1].bias, 0.0)

        self.physics_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 19))
        self.burst_head = nn.Sequential(nn.Linear(d, head_hidden), nn.SiLU(), nn.Linear(head_hidden, 1))
        self.task_dropout_rate = 0.50 

    def count_params(self) -> int: return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(self, x: Tensor) -> Tensor:
        return self.dual_path(self.physics_gate(self.encoder(x), x[:, :, 36:55]))

    def forward_from_features(
        self, shared: Tensor, x_last_obs: Tensor, z_task: Optional[Tensor] = None,
        op_setting: Optional[Tensor] = None, event_flag: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        z_task_raw = z_task if z_task is not None else (self.task_transformer(shared.unsqueeze(0)).mean(1) + self.global_task_prior)

        # 🚨 Categorical Loophole Closed: Model MUST read sensors, no categorical crutch provided here.

        z_norm = F.normalize(z_task_raw, p=2, dim=-1, eps=1e-8)
        if self.training and torch.rand(1).item() < self.task_dropout_rate: z_norm = torch.zeros_like(z_norm)
        z_exp = z_norm.expand(shared.size(0), -1)

        scale = torch.sigmoid(self.film_alpha)
        cond = (shared * (1.0 + scale * self.gamma_head(z_exp)) + scale * self.beta_head(z_exp))

        # 🚨 Delta drives the RUL decay
        baseline = F.softplus(self.baseline_head(z_exp) + self.task_to_rul(z_exp))     
        delta    = F.softplus(self.degradation_head(cond))   
        
        rul_log  = torch.clamp(baseline - delta, min=math.log1p(1.0), max=self.max_rul_log)          

        health_logit = torch.clamp(self.health_head(shared), -15.0, 15.0)
        log_alea = torch.clamp(self.aleatoric_head(torch.cat([cond, x_last_obs], dim=-1)), -5.0, 2.5)
        log_epis = torch.clamp(self.epistemic_head(z_exp), -5.0, 2.5)

        return {
            "rul_log": rul_log, "rul_log_var": torch.logaddexp(log_alea, log_epis),
            "log_aleatoric": log_alea, "log_epistemic": log_epis,
            "health": torch.sigmoid(health_logit), "health_logit": health_logit,
            "baseline": baseline, "delta": delta,
            "physics_preds": self.physics_head(shared), "burst_logit": self.burst_head(shared),
        }

    def forward(self, x: Tensor, z_task: Optional[Tensor] = None, op_setting: Optional[Tensor] = None, event_flag: Optional[Tensor] = None) -> Dict[str, Tensor]:
        return self.forward_from_features(self.extract_features(x), x[:, -1, :20], z_task, op_setting, event_flag)
