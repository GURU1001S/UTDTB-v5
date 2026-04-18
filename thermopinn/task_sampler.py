"""
task_sampler.py  ·  AeroMRO Digital Twin  ·  v19.1 (NaN-Safe Beast Mode)
════════════════════════════════════════════════════════════════════════
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import h5py
import torch

class DigitalTwinTaskSampler:
    def __init__(
        self, h5_path: str, window_size: int = 30, stride: int = 5,
        support_ratio: float = 0.6, seed: int = 42,
        device: torch.device = torch.device("cpu"), hard_neg_ratio: float = 0.20,
    ):
        self.window_size = window_size
        self.stride = stride
        self.support_ratio = support_ratio
        self.device = device
        self.hard_neg_ratio = hard_neg_ratio
        
        raw_path = h5_path.replace("\\", "/")
        if raw_path.startswith("D:/"):
            raw_path = "/mnt/d/" + raw_path[3:]
        self.h5_path = Path(raw_path).expanduser()
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"CRITICAL: HDF5 file not found at {self.h5_path}")

        random.seed(seed); np.random.seed(seed)
        
        print("[Data] Mapping HDF5 Pointers and Computing Statistics...")
        self._compute_global_stats()
        self._build_index_registry()
        self._hard_neg_scores: Dict[Tuple, float] = {}

    def _compute_global_stats(self) -> None:
        with h5py.File(self.h5_path, 'r') as f:
            s_tr = f['train']['sensors'][:]
            e_tr = f['train']['env'][:]
            p_tr = f['train']['causal_state'][:]
            rul_tr = f['train']['RUL'][:]

            X_tr = np.concatenate([s_tr, e_tr, p_tr], axis=1).astype(np.float32)
            
            # 🚨 FIX: Scrub NaNs from the dataset before computing stats
            X_tr = np.nan_to_num(X_tr, nan=0.0)
            
            self.X_mean = torch.tensor(X_tr.mean(axis=0), device=self.device)
            self.X_std  = torch.tensor(X_tr.std(axis=0) + 1e-8, device=self.device)
            
            self.max_rul = float(np.max(np.nan_to_num(rul_tr, nan=0.0)))
            self.mean_rul_log = float(np.log1p(np.mean(np.nan_to_num(rul_tr, nan=0.0))))
            
            del s_tr, e_tr, p_tr, X_tr, rul_tr

    def _build_index_registry(self) -> None:
        self._registry = {}
        self.train_tasks = []
        self.test_tasks = []
        chunk_size, chunk_stride = 80, 40

        with h5py.File(self.h5_path, 'r') as f:
            for split in ['train', 'val', 'test']:
                if split not in f: continue
                
                eng_ids = f[split]['engine_id'][:]
                ruls = f[split]['RUL'][:]
                unique_engines = np.unique(eng_ids)
                
                for eng in unique_engines:
                    eng_mask = (eng_ids == eng)
                    idx = np.where(eng_mask)[0]
                    if len(idx) < self.window_size: continue
                    
                    max_rul_eng = np.max(np.nan_to_num(ruls[idx], nan=0.0))
                    
                    for start in range(0, len(idx) - chunk_size + 1, chunk_stride):
                        chunk_idx = idx[start : start + chunk_size]
                        if len(chunk_idx) <= self.window_size: continue
                        
                        health_frac = np.mean(np.nan_to_num(ruls[chunk_idx], nan=0.0)) / (max_rul_eng + 1e-8)
                        fault = 0 if health_frac > 0.70 else 1 if health_frac > 0.40 else 2 if health_frac > 0.20 else 3
                        
                        key = (split, int(eng), fault, start // chunk_stride)
                        self._registry[key] = (chunk_idx[0], chunk_idx[-1] + 1)
                        
                        if split == 'train': self.train_tasks.append(key)
                        elif split == 'test': self.test_tasks.append(key)

    def get_curriculum_tasks(self, epoch: int, n_epochs: int) -> List[Tuple]:
        pool_size = len(self.train_tasks)
        early = [t for t in self.train_tasks if t[2] in (0, 1)]
        mid   = [t for t in self.train_tasks if t[2] == 2]
        eol   = [t for t in self.train_tasks if t[2] == 3]

        if epoch < 50:
            n_early, n_mid = int(pool_size * 0.50), int(pool_size * 0.30)
            n_eol = pool_size - n_early - n_mid
        else:
            n_eol, n_mid = int(pool_size * 0.40), int(pool_size * 0.30)
            n_early = pool_size - n_eol - n_mid

        return random.choices(early, k=n_early) + random.choices(mid, k=n_mid) + random.choices(eol, k=n_eol)

    def update_hard_negatives(self, task_id: Tuple, rmse: float) -> None:
        self._hard_neg_scores[task_id] = rmse

    def sample_with_hard_negatives(self, base_tasks: List[Tuple], n_total: int) -> List[Tuple]:
        n_hard = max(1, int(n_total * self.hard_neg_ratio))
        hard = [tid for tid, _ in sorted(self._hard_neg_scores.items(), key=lambda x: -x[1])[:n_hard] if tid in set(self.train_tasks)]
        base = random.sample(base_tasks, min(n_total - len(hard), len(base_tasks)))
        if not hard: return base
        while len(hard) < n_hard: hard = hard + hard
        return base + hard[:n_hard]

    def get_fast_task_tensors(self, task_id: Tuple) -> Tuple[Optional[Dict], Optional[Dict]]:
        if task_id not in self._registry: return None, None
        
        split, eng, fault, chunk_id = task_id
        start_idx, end_idx = self._registry[task_id]

        with h5py.File(self.h5_path, 'r') as f:
            grp = f[split]
            s_raw = grp['sensors'][start_idx:end_idx]
            e_raw = grp['env'][start_idx:end_idx]
            p_raw = grp['causal_state'][start_idx:end_idx]
            
            rul = grp['RUL'][start_idx:end_idx]
            rul_alea = grp['RUL_alea'][start_idx:end_idx]
            rul_epis = grp['RUL_epi'][start_idx:end_idx]
            health = grp['health_index'][start_idx:end_idx]
            
            op_set = grp['op_setting'][start_idx:end_idx] 
            ev_flag = grp['event_flag'][start_idx:end_idx] 

        X_raw = np.concatenate([s_raw, e_raw, p_raw], axis=1).astype(np.float32)
        
        # 🚨 FIX: Scrub NaNs from the dynamic chunk
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        
        X = torch.tensor(X_raw, device=self.device)
        X = torch.clamp((X - self.X_mean) / self.X_std, -5.0, 5.0)

        n_win = max(0, (len(X) - self.window_size) // self.stride + 1)
        if n_win < 2: return None, None

        X_win = X.unfold(0, self.window_size, self.stride).transpose(1, 2)
        t_idx = (np.arange(n_win) * self.stride + (self.window_size - 1))
        
        Y_rul, Y_alea, Y_epis = torch.tensor(rul[t_idx], device=self.device), torch.tensor(rul_alea[t_idx], device=self.device), torch.tensor(rul_epis[t_idx], device=self.device)
        h_t = torch.tensor(health[t_idx], device=self.device)
        
        Y_rul = torch.nan_to_num(Y_rul, nan=0.0)
        rul_log = torch.log1p(Y_rul.clamp(min=0.0))
        
        op_t = torch.tensor(np.nan_to_num(op_set[t_idx], nan=0), dtype=torch.long, device=self.device)
        ev_t = torch.tensor(np.nan_to_num(ev_flag[t_idx], nan=0), dtype=torch.long, device=self.device)

        query_mask = torch.zeros(n_win, dtype=torch.bool, device=self.device)
        query_mask[::5] = True 
        sup_idx, qry_idx = (~query_mask).nonzero(as_tuple=True)[0], query_mask.nonzero(as_tuple=True)[0]

        if len(sup_idx) == 0 or len(qry_idx) == 0: return None, None

        def make(idx):
            return {
                "x": X_win[idx].contiguous(), "rul_log": rul_log[idx], 
                "rul_alea": Y_alea[idx], "rul_epis": Y_epis[idx],
                "health": h_t[idx], "op_setting": op_t[idx], "event_flag": ev_t[idx]
            }
            
        return make(sup_idx), make(qry_idx)

    def held_out_split(self) -> Tuple[List[Tuple], List[Tuple]]:
        return self.train_tasks, self.test_tasks
