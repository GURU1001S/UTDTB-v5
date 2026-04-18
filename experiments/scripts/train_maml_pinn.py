"""
train_maml_pinn.py  ·  AeroMRO Digital Twin  ·  v19.21 (Goldilocks Tuning)
══════════════════════════════════════════════════════════════════════════
"""

import math, copy, random, os, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import higher
import mlflow
from tqdm import tqdm

from task_sampler      import DigitalTwinTaskSampler
from pinn_model        import PINNModel
from physics_loss      import CompositePINNLoss, NASAAsymmetricScore, log_nasa

# ─── Config (40-EPOCH MAML-SAFE) ──────────────────────────────────────────────
CONFIG = {
    "h5_path":         "~/nasa_research/data/utdtb_v5.h5",
    "mlflow_uri":      "sqlite:///mlflow.db",
    "checkpoint_dir":  "~/nasa_research/checkpoints",
    "seed":            42,
    "window_size":     30,
    "stride":          5,
    "support_ratio":   0.6, 

    "reptile_epochs":  20,
    "reptile_lr":      0.5,

    # 🚀 THE FULL BURN
    "n_meta_epochs":   300,      # 🚨 Unleashed to 300
    "batch_size":      128,      
    "tasks_per_batch": 12,       
    "accum_steps":     4,       
    "inner_steps":     3,       
    
    "inner_lr":             0.01,  
    "eval_inner_lr_factor": 1.0,  
    
    "encoder_lr":           1e-3,
    "head_lr":              1e-3, 
    "grad_clip_norm":       1.0,     
    "max_task_loss":        300.0,

    "n_sensors":       55,
    "n_op_settings":   32,
    "n_events":        10,
    "conv_channels":   256,     
    "gru_hidden":      512,
    "head_hidden":     128,
    "dropout":         0.30,    

    "base_lambdas": {
        "huber":       1.0,
        "monotonic":   0.10,
        "health":      0.20,
        "unc":         0.10,
        "phys_sup":    0.50,
        "damage_mono": 1.0,    
        "health_mono": 1.0     
    },

    "fewshot_batches": [0, 5, 10],   
    "eval_tasks":      100,       
    "mc_passes":       20,        
}

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

def get_anil_params(model: PINNModel) -> List[nn.Parameter]:
    return (
        list(model.dual_path.parameters()) + list(model.baseline_head.parameters()) + 
        list(model.degradation_head.parameters()) + list(model.health_head.parameters()) + 
        list(model.gamma_head.parameters()) + list(model.beta_head.parameters()) + 
        list(model.aleatoric_head.parameters()) + list(model.epistemic_head.parameters()) +
        list(model.physics_head.parameters()) + list(model.burst_head.parameters()) +
        list(model.task_to_rul.parameters()) + 
        [model.film_alpha]
    )

def augment_gpu(x: torch.Tensor, rul: torch.Tensor, device) -> Tuple:
    B, T, C = x.shape
    sigma   = torch.rand(1, device=device).item() * 0.012
    noise   = torch.cat([torch.randn(B, T, 20, device=device) * sigma, torch.zeros(B, T, C - 20, device=device)], dim=-1)
    x_aug   = (x + noise) * (0.93 + torch.rand(1, 1, C, device=device) * 0.14)
    warp    = 0.85 + torch.rand(1, device=device).item() * 0.30
    wT      = max(10, int(T * warp))
    idx     = torch.linspace(0, T - 1, wT).long().to(device)
    xw      = x_aug[:, idx, :]
    if wT < T: x_aug = torch.cat([xw, xw[:, -1:, :].expand(-1, T - wT, -1)], dim=1)
    else: x_aug = xw[:, :T, :]
    return x_aug.clamp(-5.0, 5.0), rul - math.log(warp)

def nasa_shaped_loss(pred, true, delta=1.0):
    error = pred - true
    huber = F.smooth_l1_loss(pred, true, reduction="none", beta=delta)
    late_amp = torch.where(error > 0, (1.0 + torch.log1p(torch.relu(error) / 5.0)).clamp(max=2.0), torch.ones_like(error))
    return (huber * late_amp).mean()

def maml_inner_loop(fmodel, diffopt, sup_data: Dict, cfg: dict, device: torch.device) -> bool:
    if sup_data is None: return False
    fmodel.train()
    num = sup_data["x"].shape[0]
    idx = torch.randperm(num, device=device)[:min(cfg["batch_size"], num)]

    xb, yb = sup_data["x"][idx], sup_data["rul_log"][idx]
    op_b, ev_b = sup_data["op_setting"][idx], sup_data["event_flag"][idx]
    x_a, y_a = augment_gpu(xb, yb, device)
    xa, ya = torch.cat([xb, x_a], dim=0), torch.cat([yb, y_a], dim=0)
    op_a, ev_a = torch.cat([op_b, op_b], dim=0), torch.cat([ev_b, ev_b], dim=0)

    with autocast("cuda"): shared = fmodel.extract_features(xa)
        
    for _ in range(cfg["inner_steps"]):
        with autocast("cuda"):
            out  = fmodel.forward_from_features(shared, xa[:, -1, :20], op_setting=op_a, event_flag=ev_a)
            p, t = out["rul_log"].squeeze(-1), ya.squeeze(-1)
            
            loss = F.smooth_l1_loss(p, t, reduction="none").mean()
            
            sidx = torch.argsort(t, descending=True)
            loss += 0.10 * torch.relu(p[sidx][1:] - p[sidx][:-1]).mean()
            
            delta_inner = out["delta"].squeeze(-1)[sidx]
            loss += cfg.get("base_lambdas", {}).get("damage_mono", 0.5) * torch.relu(delta_inner[:-1] - delta_inner[1:]).mean()

        if not math.isfinite(loss.item()): return False
        diffopt.step(loss)
    return True

def query_eval(fmodel, qry_data: Dict, loss_fn: CompositePINNLoss, device: torch.device) -> Tuple[torch.Tensor, float, float]:
    if qry_data is None: return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0
    fmodel.train()
    x, rul = qry_data["x"], qry_data["rul_log"]
    with torch.enable_grad():
        with autocast("cuda"):
            out = fmodel(x, op_setting=qry_data["op_setting"], event_flag=qry_data["event_flag"])
            losses = loss_fn.compute(
                out, rul.float().unsqueeze(-1), x,
                true_health=qry_data["health"].float().unsqueeze(-1),
                use_pde=True,
            )
    pred_cy, true_cy = torch.expm1(out["rul_log"].detach().squeeze(-1)), torch.expm1(rul)
    rmse = float(torch.sqrt(F.mse_loss(pred_cy, true_cy)).item())
    return losses["total"], rmse, log_nasa(out["rul_log"].detach(), rul.unsqueeze(-1))

def train(cfg: dict = CONFIG, return_best: bool = False, return_metrics: bool = False):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device} | 🚀 FINAL CERTIFICATION RUN (40-EPOCH MAML)")
    
    ckpt_dir = Path(cfg["checkpoint_dir"]).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sampler = DigitalTwinTaskSampler(
        h5_path=cfg["h5_path"], window_size=cfg["window_size"], stride=cfg["stride"],
        support_ratio=cfg["support_ratio"], seed=cfg["seed"], device=device,
    )
    train_tasks, test_tasks = sampler.held_out_split()

    model = PINNModel(
        max_rul=sampler.max_rul, n_sensors=cfg["n_sensors"], conv_channels=cfg["conv_channels"],
        gru_hidden=cfg["gru_hidden"], head_hidden=cfg["head_hidden"], dropout=cfg["dropout"],
        n_op_settings=cfg["n_op_settings"], n_events=cfg["n_events"], mean_rul_log=sampler.mean_rul_log,
    ).to(device)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    print("\n[Reptile] Warming up feature encoder...")
    for r_epoch in range(cfg["reptile_epochs"]):
        model.train()
        r_tasks = random.sample(train_tasks, min(4, len(train_tasks)))
        for task_id in r_tasks:
            sup, _ = sampler.get_fast_task_tensors(task_id)
            if sup is None: continue
            
            clone = copy.deepcopy(model)
            opt_r = torch.optim.SGD(clone.parameters(), lr=3e-3)
            
            for _ in range(3):  
                num = sup["x"].shape[0]
                idx = torch.randperm(num, device=device)[:min(32, num)]
                with autocast("cuda"):
                    out = clone(sup["x"][idx], op_setting=sup["op_setting"][idx], event_flag=sup["event_flag"][idx])
                    loss = F.smooth_l1_loss(out["rul_log"].squeeze(-1), sup["rul_log"][idx].squeeze(-1), beta=1.0)
                
                opt_r.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt_r)
                torch.nn.utils.clip_grad_norm_(clone.parameters(), 1.0)
                scaler.step(opt_r)
                scaler.update()
            
            with torch.no_grad():
                for p, q in zip(model.parameters(), clone.parameters()):
                    p.data.add_(cfg["reptile_lr"] * (q.data - p.data))
    
    print("[Reptile] Complete. Starting MAML phase.\n")

    anil_params = get_anil_params(model)
    anil_param_ids = {id(p) for p in anil_params}
    encoder_params = [p for p in model.parameters() if id(p) not in anil_param_ids]
    
    outer_optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["encoder_lr"], "weight_decay": 1e-4},
        {"params": anil_params,    "lr": cfg["head_lr"],    "weight_decay": 1e-5},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(outer_optimizer, T_max=cfg["n_meta_epochs"], eta_min=1e-6)
    loss_fn  = CompositePINNLoss(base_lambdas=cfg["base_lambdas"])
    nasa_sc  = NASAAsymmetricScore(clamp_range=50.0)
    
    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment("aeromro-pinn-v19")

    best_nasa_log, best_rmse_val = float("inf"), float("inf")
    
    with mlflow.start_run():
        pbar = tqdm(range(cfg["n_meta_epochs"]), desc="Epochs")
        for epoch in pbar:
            model.train()
            loss_fn.set_epoch(epoch)
            available = sampler.get_curriculum_tasks(epoch, cfg["n_meta_epochs"]) if hasattr(sampler, "get_curriculum_tasks") else train_tasks
                
            model.task_dropout_rate = max(0.0, min(0.50, (epoch - 10) / max(1, cfg["n_meta_epochs"]) * 0.50))
            epoch_rmse, epoch_lnasa, epoch_loss = [], [], 0.0
            outer_optimizer.zero_grad(set_to_none=True)
            total_valid = 0

            for p in model.task_to_rul.parameters(): p.requires_grad_(True)

            for accum_step in range(cfg["accum_steps"]):
                base = random.sample(available, min(cfg["tasks_per_batch"], len(available)))
                task_ids = sampler.sample_with_hard_negatives(base, cfg["tasks_per_batch"]) if hasattr(sampler, "sample_with_hard_negatives") else base
                accum_loss, accum_valid = torch.tensor(0.0, device=device), 0

                for task_id in task_ids:
                    sup, qry = sampler.get_fast_task_tensors(task_id)
                    if sup is None or qry is None: continue
                    with torch.no_grad():
                        try:
                            with autocast("cuda"):
                                pred_0_cy = torch.expm1(model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])["rul_log"].squeeze(-1))
                                true_cy = torch.expm1(qry["rul_log"])
                                L_anchor = 0.05 * torch.relu((true_cy.mean() - pred_0_cy.mean()) - 80.0) / sampler.max_rul
                        except Exception: L_anchor = 0.0

                    inner_opt = torch.optim.Adam(get_anil_params(model), lr=cfg["inner_lr"])
                    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True, track_higher_grads=True, device=device) as (fmodel, diffopt):
                        if not maml_inner_loop(fmodel, diffopt, sup, cfg, device): continue
                        q_loss, rmse, lnasa = query_eval(fmodel, qry, loss_fn, device)

                    if not math.isfinite(q_loss.item()) or q_loss.item() > cfg["max_task_loss"]: continue

                    accum_loss += (q_loss + L_anchor) / (q_loss.detach() + 1e-6)
                    accum_valid += 1
                    epoch_rmse.append(rmse); epoch_lnasa.append(lnasa); epoch_loss += q_loss.item()

                if accum_valid > 0:
                    scaled = accum_loss / (accum_valid * cfg["accum_steps"])
                    if scaled.grad_fn is not None and torch.isfinite(scaled):
                        scaler.scale(scaled).backward()
                        total_valid += accum_valid

            if total_valid > 0 and any(p.grad is not None for p in model.parameters()):
                scaler.unscale_(outer_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                scaler.step(outer_optimizer)
                scaler.update()
                scheduler.step()

            if epoch_rmse:
                avg_rmse, avg_lnasa = float(np.mean(epoch_rmse)), float(np.mean(epoch_lnasa))
                pbar.set_postfix_str(f"[MAML] Loss:{epoch_loss / max(1, len(epoch_rmse)):.3f} | RMSE:{avg_rmse:.1f} | log-NASA:{avg_lnasa:.2f} | LR:{scheduler.get_last_lr()[0]:.1e}")

                if avg_lnasa < best_nasa_log:
                    best_nasa_log, best_rmse_val = avg_lnasa, avg_rmse
                    if not return_metrics and not return_best:
                        # Saving intermediate best checkpoints
                        torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_dir / "best_model_v19.pt")

        print("\n[Eval] Few-shot evaluation on held-out engines...")
        best_ece = run_fewshot_eval(model, sampler, test_tasks, loss_fn, nasa_sc, cfg, device)

        # ─── THE NEW FINAL SAVE LOGIC ──────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        final_model_name = f"thermoPINN_metal_ready_v20_{timestamp}.pt"
        save_path = os.path.join(os.path.expanduser("~/nasa_research/"), final_model_name)
        
        # Save the pure state_dict for clean deployment to physical inference scripts
        torch.save(model.state_dict(), save_path)
        print(f"\n✅ FULL TRAINING COMPLETE.")
        print(f"✅ Final physical-transfer weights safely locked to: {final_model_name}")

    if return_best: return best_nasa_log
    if return_metrics: return {"rmse": best_rmse_val, "nasa": best_nasa_log, "ece": best_ece}

def run_fewshot_eval(model, sampler, test_tasks, loss_fn, nasa_sc, cfg, device) -> float:
    from eval_runner import ConformalECECalibrator 
    
    adapted_model = PINNModel(
        max_rul=sampler.max_rul, n_sensors=cfg["n_sensors"], conv_channels=cfg["conv_channels"],
        gru_hidden=cfg["gru_hidden"], head_hidden=cfg["head_hidden"], dropout=cfg["dropout"],
        n_op_settings=cfg["n_op_settings"], n_events=cfg["n_events"], mean_rul_log=sampler.mean_rul_log,
    ).to(device)

    print("\n[Calibration] Building conformal calibrators from validation split...")
    val_tasks = [t for t in sampler._registry.keys() if t[0] == 'val']
    
    conf_calibrators = {}
    eval_tasks = random.sample(test_tasks, min(cfg["eval_tasks"], len(test_tasks)))
    
    print("\n── Few-shot evaluation (lower = better) ──")
    print(f"   {'k':>5} | {'RMSE':>8} | {'NASA':>10} | {'log-N':>7} | {'ECE':>7} | {'Pred':>7} | {'True':>7}")
    final_ece = 1.0

    for k_batches in cfg["fewshot_batches"]:
        k_conf = ConformalECECalibrator(alpha=0.10)
        val_k_preds, val_k_trues = [], []
        
        for val_tid in random.sample(val_tasks, min(20, max(1, len(val_tasks)))):
            sup_v, qry_v = sampler.get_fast_task_tensors(val_tid)
            if sup_v is None or qry_v is None: continue
            
            val_model = copy.deepcopy(model)
            if k_batches > 0:
                val_model.train()
                adapt_opt = torch.optim.Adam(get_anil_params(val_model), lr=cfg["inner_lr"] * cfg.get("eval_inner_lr_factor", 0.25))
                for _ in range(k_batches):
                    num = sup_v["x"].shape[0]
                    idx = torch.randperm(num, device=device)[:min(cfg["batch_size"], num)]
                    with autocast("cuda"):
                        p = val_model(sup_v["x"][idx], op_setting=sup_v["op_setting"][idx], event_flag=sup_v["event_flag"][idx])["rul_log"].squeeze(-1)
                        loss = F.smooth_l1_loss(p, sup_v["rul_log"][idx].squeeze(-1))
                    adapt_opt.zero_grad()
                    loss.backward()
                    adapt_opt.step()
            
            val_model.eval()
            with torch.no_grad():
                out_v = val_model(qry_v["x"], op_setting=qry_v["op_setting"], event_flag=qry_v["event_flag"])
            val_k_preds.append(out_v["rul_log"].detach().cpu().flatten())
            val_k_trues.append(qry_v["rul_log"].cpu().flatten())
            
        if val_k_preds:
            k_conf.calibrate(torch.cat(val_k_preds), torch.cat(val_k_trues))
            conf_calibrators[k_batches] = k_conf

        all_preds, all_trues = [], []
        for task_id in tqdm(eval_tasks, desc=f"{k_batches}-shot", leave=False):
            sup, qry = sampler.get_fast_task_tensors(task_id)
            if sup is None or qry is None: continue

            adapted_model.load_state_dict(copy.deepcopy(model.state_dict()))

            if k_batches > 0:
                adapted_model.train()
                eval_lr_base = cfg["inner_lr"] * cfg.get("eval_inner_lr_factor", 0.25)
                adapt_opt = torch.optim.Adam(get_anil_params(adapted_model), lr=eval_lr_base)
                scaler_e = GradScaler("cuda", enabled=(device.type == "cuda"))
                
                for step_i in range(k_batches):
                    cos_factor = 0.5 * (1 + math.cos(math.pi * step_i / max(1, k_batches - 1)))
                    for pg in adapt_opt.param_groups: pg['lr'] = eval_lr_base * (0.05 + 0.95 * cos_factor)

                    num = sup["x"].shape[0]
                    idx = torch.randperm(num, device=device)[:min(cfg["batch_size"], num)]
                    xb, yb = sup["x"][idx], sup["rul_log"][idx]
                    op_b, ev_b = sup["op_setting"][idx], sup["event_flag"][idx]
                    x_a, y_a = augment_gpu(xb, yb, device)
                    xc, yc = torch.cat([xb, x_a], dim=0), torch.cat([yb, y_a], dim=0)
                    op_c, ev_c = torch.cat([op_b, op_b], dim=0), torch.cat([ev_b, ev_b], dim=0)

                    with autocast("cuda"):
                        p, t = adapted_model(xc, op_setting=op_c, event_flag=ev_c)["rul_log"].squeeze(-1), yc.squeeze(-1)
                        loss = F.smooth_l1_loss(p, t, reduction="none").mean()

                    adapt_opt.zero_grad()
                    scaler_e.scale(loss).backward()
                    scaler_e.unscale_(adapt_opt)
                    torch.nn.utils.clip_grad_norm_(get_anil_params(adapted_model), 0.3)
                    scaler_e.step(adapt_opt)
                    scaler_e.update()

            adapted_model.train() 
            with torch.no_grad():
                mc_preds, mc_aleats = [], []
                for _ in range(cfg["mc_passes"]):
                    with autocast("cuda"):
                        out_pass = adapted_model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])
                        mc_preds.append(out_pass["rul_log"].detach())
                        mc_aleats.append(torch.exp(0.5 * out_pass["rul_log_var"].detach()))
                
                mc_preds_stack = torch.stack(mc_preds, dim=0)
                mc_aleats_stack = torch.stack(mc_aleats, dim=0)
                
                mean_log = mc_preds_stack.mean(0)
                epist_var = mc_preds_stack.var(0)
                aleat_var = (mc_aleats_stack ** 2).mean(0)
                total_var = epist_var + aleat_var
                
                health = out_pass["health"].detach()
                health_ceiling_log = torch.log1p(sampler.max_rul * health.squeeze(-1) * 1.5)
                max_log_val = math.log1p(sampler.max_rul)
                max_bound = torch.minimum(torch.tensor(max_log_val, device=device), health_ceiling_log).expand_as(mean_log.squeeze(-1))
                mean_log = torch.minimum(mean_log.squeeze(-1).clamp(min=0.0), max_bound).unsqueeze(-1)
                
            adapted_model.eval()

            all_preds.append(mean_log.cpu().flatten())
            all_trues.append(qry["rul_log"].cpu().flatten())

        if not all_preds: continue
        
        gp, gt = torch.cat(all_preds), torch.cat(all_trues)
        gp_cy, gt_cy = torch.expm1(gp), torch.expm1(gt)
        nasa = float(nasa_sc(gp.unsqueeze(-1), gt.unsqueeze(-1)).item())
        
        current_ece = 1.0
        if k_batches in conf_calibrators:
            current_ece = conf_calibrators[k_batches].compute_conformal_ece(gp, gt)
            final_ece = current_ece

        print(f"   {k_batches:>5} | {float(torch.sqrt(F.mse_loss(gp_cy, gt_cy)).item()):>8.2f} | {nasa:>10.2f} | "
              f"{math.log1p(nasa):>7.3f} | {current_ece:>7.4f} | {gp_cy.mean().item():>7.1f} | {gt_cy.mean().item():>7.1f}")
              
    return final_ece

if __name__ == "__main__":
    train()
