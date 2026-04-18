"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  UTDTB v5.0  —  UNIVERSAL TURBOFAN DIGITAL TWIN BENCHMARK                      ║
║  BEAST MODE  ·  Highest Fidelity  ·  Google Colab Ready                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  NEW IN v5.0 (10 major upgrades over v4.1):                                    ║
║  1.  Flight-phase dynamics  (taxi/takeoff/climb/cruise/descent/landing)         ║
║  2.  Rich maintenance events  (wash, blade repair, module swap + columns)       ║
║  3.  Advanced sensor failure modes  (stuck/drift/dropout + fault flags)         ║
║  4.  Multi-engine fleet pairs  (cross-engine delta RPM/EGT + pair_id)           ║
║  5.  Mission-level context  (route_distance, flight_duration, elevation, sand)  ║
║  6.  Rare catastrophic events  (bird/stall/ice/fuel contamination)              ║
║  7.  Pilot / operational variation  (throttle ramp rate, climb profile)         ║
║  8.  Per-engine sensor calibration bias  (unique b_i per engine)                ║
║  9.  Timestamp + irregular sampling  (timestamp, sampling_interval cols)        ║
║  10. Full RUL uncertainty  (RUL_mean/std/lower/upper/failure_prob)              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  HOW TO RUN IN GOOGLE COLAB:                                                    ║
║    Cell 1:  !pip install h5py pyarrow scipy -q                                  ║
║    Cell 2:  %run utdtb_v5_beast.py                                              ║
║    Cell 3:  from google.colab import files                                      ║
║             files.download('/content/utdtb_v5_complete.zip')                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  MODES:  QUICK(~3min,~200MB)  MEDIUM(~15min,~600MB)                            ║
║          FULL(~50min,~2.5GB)  BEAST(~2.5hr,~6GB) ← DEFAULT                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-INSTALL
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

def _pip(pkg):
    try:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg],
                              stderr=subprocess.DEVNULL)
    except Exception:
        pass

for _p, _m in [("h5py","h5py"),("pyarrow","pyarrow"),
               ("scipy","scipy"),("fastparquet","fastparquet")]:
    try:
        __import__(_m)
    except ImportError:
        print(f"  Installing {_p}...")
        _pip(_p)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, time, json, warnings, logging, zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from copy import deepcopy
from enum import IntEnum

import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.stats import norm as scipy_norm
import h5py

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("UTDTB_v50")

# ══════════════════════════════════════════════════════════════════════════════
#  ▶▶  SET MODE HERE  ◀◀
#  "QUICK"  ~500K rows  ~200MB   ~3 min
#  "MEDIUM" ~1.5M rows  ~600MB   ~15 min
#  "FULL"   ~7M   rows  ~2.5GB   ~50 min
#  "BEAST"  ~16M  rows  ~6GB     ~2.5 hr  ← HIGHEST FIDELITY
# ══════════════════════════════════════════════════════════════════════════════
MODE = "BEAST"


# ══════════════════════════════════════════════════════════════════════════════
# §0  FLIGHT PHASE ENUM  [NEW — Feature 1]
# ══════════════════════════════════════════════════════════════════════════════

class FlightPhase(IntEnum):
    TAXI     = 0
    TAKEOFF  = 1
    CLIMB    = 2
    CRUISE   = 3
    DESCENT  = 4
    LANDING  = 5

PHASE_NAMES = ["taxi", "takeoff", "climb", "cruise", "descent", "landing"]

# Per-phase envelope: (throttle_base, throttle_sigma, alt_frac, EGT_bias, duration_frac)
# alt_frac = fraction of cruise altitude for this phase
PHASE_ENVELOPE = {
    FlightPhase.TAXI:    dict(thr=0.12, thr_s=0.02, alt_f=0.00, EGT_b=-120, dur=0.04),
    FlightPhase.TAKEOFF: dict(thr=0.98, thr_s=0.01, alt_f=0.02, EGT_b=+80,  dur=0.05),
    FlightPhase.CLIMB:   dict(thr=0.92, thr_s=0.03, alt_f=0.45, EGT_b=+40,  dur=0.15),
    FlightPhase.CRUISE:  dict(thr=0.82, thr_s=0.04, alt_f=1.00, EGT_b=0,    dur=0.55),
    FlightPhase.DESCENT: dict(thr=0.35, thr_s=0.05, alt_f=0.30, EGT_b=-60,  dur=0.14),
    FlightPhase.LANDING: dict(thr=0.55, thr_s=0.04, alt_f=0.01, EGT_b=-20,  dur=0.07),
}


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsConfig:
    gamma_c: float = 1.40;       gamma_t: float = 1.33
    cp_c: float    = 1005.0;     cp_t: float    = 1148.0
    R_gas: float   = 287.0;      LHV: float     = 43.2e6
    T_ref: float   = 288.15;     P_ref: float   = 101325.0
    OPR_design: float = 32.0;    BPR_design: float = 8.5
    eta_c0: float  = 0.880;      eta_t0: float  = 0.900
    eta_comb0: float = 0.998;    T4_max: float  = 1750.0
    mdot_core: float = 50.0;     N_design: float = 12500.0
    eta_c_alpha_fat: float  = 0.12
    eta_c_alpha_cor: float  = 0.18
    eta_c_alpha_foul: float = 0.25
    thermal_mass_core: float = 800.0
    heat_loss_coeff: float   = 12.0
    a0_mean: float  = 0.0010;   a0_sigma: float  = 0.0003
    a_crit: float   = 0.0120;   paris_C: float   = 1.0e-11
    paris_m: float  = 3.0;      K_Ic: float      = 50.0
    weibull_k: float = 3.0;     weibull_sigma0: float = 900.0
    fouling_eff_min: float = 0.70
    mfg_eta_c_sig: float = 0.012
    mfg_eta_t_sig: float = 0.010
    mfg_mdot_sig: float  = 0.015
    mfg_N_sig: float     = 0.006


@dataclass
class DegradationConfig:
    C_fat: float    = 2.5e-6;   m_fat: float    = 3.2
    walker_c: float = 0.5
    A_crp: float    = 1.2e-15;  n_crp: float    = 3.5
    Q_crp: float    = 250e3;    R_univ: float   = 8.314
    C_th: float     = 1.5e-5;   m_th: float     = 1.8
    DT_ref: float   = 50.0
    k_cor: float    = 3.0e-7;   Q_cor: float    = 60e3
    salt_coeff: float = 2.5
    k_foul: float   = 4.5e-5;   foul_exp: float = 1.4
    fat_crp_coupling: float = 0.15
    cor_fat_coupling: float = 0.10
    drift_vol: float = 1.5e-4
    # Takeoff stress multiplier on fatigue (highest damage per cycle)
    takeoff_fat_mult: float = 3.5
    # Thermal shock multiplier on taxi→takeoff transition
    thermal_shock_mult: float = 2.2


@dataclass
class FADECConfig:
    EGT_redline: float         = 960.0
    EGT_amber: float           = 920.0
    N_overspeed: float         = 1.05
    throttle_lag_tau: float    = 2.0
    throttle_rate_limit: float = 0.15
    fuel_trim_auth: float      = 0.05
    surge_margin_min: float    = 0.10
    bleed_eff_ratio: float     = 0.75
    overtemp_cycles_limit: int = 3


@dataclass
class MaintenanceConfig:
    """Feature 2: enriched maintenance with new event types."""
    wash_prob: float        = 5.0e-4
    wash_recovery: float    = 0.30
    wash_quality_sig: float = 0.20
    # Blade repair — reduces crack length and fatigue
    blade_repair_prob: float    = 1.5e-4
    blade_repair_crack_frac: float = 0.40   # fraction of crack length healed
    blade_repair_fat_frac: float   = 0.25
    # Module replacement — major event, resets multiple degradation channels
    module_replace_prob: float  = 3.0e-5
    module_replace_eff_c_reset: float = 0.92  # fraction of new efficiency restored
    module_replace_eff_t_reset: float = 0.90
    # Existing
    repair_prob: float      = 2.0e-4
    repair_frac_sig: float  = 0.20
    sensor_rep_prob: float  = 1.0e-4
    sensor_rep_n: int       = 3
    miscal_prob: float      = 5.0e-5
    miscal_sig: float       = 0.025


@dataclass
class RareEventConfig:
    bird_strike_prob: float = 1.0e-4
    bird_EGT_spike: float   = 80.0
    bird_vib_shock: float   = 0.15
    bird_fat_boost: float   = 0.08
    stall_prob: float       = 8.0e-5
    stall_cycles: int       = 5
    stall_P3_drop: float    = 0.18
    oil_fail_prob: float    = 5.0e-5
    oil_temp_rate: float    = 3.0
    oil_fail_max_cyc: int   = 20
    seal_prob: float        = 7.0e-5
    seal_decay: float       = 0.002
    seal_max_cyc: int       = 50
    ash_prob: float         = 2.0e-5
    ash_cor_mult: float     = 5.0
    sand_prob: float        = 6.0e-5
    sand_eta_drop: float    = 0.003
    ice_prob: float         = 4.0e-5
    ice_EGT: float          = 40.0
    rain_prob: float        = 3.0e-5
    rain_EGT_drop: float    = 15.0
    crosswind_prob: float   = 8.0e-5
    crosswind_vib: float    = 0.04
    # Feature 6: fuel contamination
    fuel_contam_prob: float = 3.0e-5
    fuel_contam_EGT: float  = 35.0
    fuel_contam_SFC: float  = 0.08    # SFC increase fraction
    fuel_contam_dur: int    = 15      # cycles of instability
    # Compressor blade FOD (foreign object damage)
    fod_prob: float         = 2.0e-5
    fod_crack_boost: float  = 0.0005
    fod_vib: float          = 0.06


@dataclass
class SensorFailConfig:
    """Feature 3: enhanced sensor failure modes with fault flag tracking."""
    stuck_prob: float  = 8.0e-4;  stuck_min: int   = 5;   stuck_max: int = 50
    delay_prob: float  = 5.0e-4;  delay_max: int   = 3
    quant_prob: float  = 3.0e-3;  quant_bits: int  = 8
    sat_prob: float    = 2.0e-4;  sat_frac: float  = 0.98
    spike_prob: float  = 1.0e-3;  spike_mult: float = 8.0
    missing_prob: float = 0.003
    # Gradual drift fault (feature 3)
    drift_fault_prob: float = 2.0e-4   # probability per engine of a drifting sensor
    drift_fault_rate: float = 0.0008   # units per cycle drift rate
    drift_fault_max_sensors: int = 3


@dataclass
class FleetConfig:
    """Feature 4 & 5: fleet pairs and mission context."""
    n_airlines: int = 6
    route_types: List[str] = field(default_factory=lambda: [
        "desert", "coastal", "arctic", "tropical", "highland", "mixed"])
    maint_agg: List[float] = field(default_factory=lambda:
                                   [0.3, 0.5, 0.7, 0.9, 0.4, 0.6])
    utilization: List[float] = field(default_factory=lambda:
                                     [0.70, 0.85, 1.0, 0.90, 0.75, 0.80])
    # Feature 4: engine pairs
    pair_egt_tolerance: float = 15.0   # [K] normal cross-engine delta
    pair_rpm_tolerance: float = 50.0   # [RPM]
    # Feature 5: mission profiles
    route_distances_nm: Dict[str, Tuple[float,float]] = field(
        default_factory=lambda: {
            "desert":   (800, 3000),
            "coastal":  (200, 1200),
            "arctic":   (1500, 5000),
            "tropical": (300, 2000),
            "highland": (400, 1800),
            "mixed":    (500, 4000),
        })
    airport_elevations_ft: Dict[str, Tuple[float,float]] = field(
        default_factory=lambda: {
            "desert":   (1000, 5000),
            "coastal":  (0, 200),
            "arctic":   (100, 2000),
            "tropical": (0, 500),
            "highland": (3000, 14000),
            "mixed":    (0, 3000),
        })
    sand_exposure: Dict[str, float] = field(
        default_factory=lambda: {
            "desert": 0.9, "coastal": 0.3, "arctic": 0.05,
            "tropical": 0.2, "highland": 0.15, "mixed": 0.3
        })


@dataclass
class PilotConfig:
    """Feature 7: pilot / operational variation."""
    # Throttle ramp aggressiveness (multiplier on rate limit)
    ramp_rate_min: float = 0.5
    ramp_rate_max: float = 2.0
    # Climb profile: fraction of max throttle during climb
    climb_throttle_min: float = 0.85
    climb_throttle_max: float = 0.99
    # Cruise altitude variation (fraction of nominal)
    cruise_alt_var: float = 0.06
    # Engine-out margin preference (conservative vs aggressive)
    surge_margin_pref_min: float = 0.08
    surge_margin_pref_max: float = 0.20


@dataclass
class UncertaintyConfig:
    """Feature 10: full RUL uncertainty with failure probability."""
    alea_frac: float = 0.08
    epi_frac: float  = 0.05
    ci_levels: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.25, 0.75, 0.90, 0.95])
    # Failure probability threshold (HI below this → compute Pf)
    failure_hi_thresh: float = 0.30


@dataclass
class SamplingConfig:
    """Feature 9: timestamp and irregular sampling."""
    # Base sampling interval in minutes
    sample_interval_min: float = 60.0
    sample_interval_sigma: float = 8.0
    # Probability of skipping a measurement entirely (ACARS dropout)
    dropout_prob: float = 0.008
    # Start date (Unix timestamp, default Jan 1 2015)
    epoch_start: float = 1420070400.0


@dataclass
class DatasetConfig:
    n_train: int        = 1000
    n_val: int          = 200
    n_test: int         = 200
    min_cycles: int     = 100
    max_cycles: int     = 1200
    partial_frac: float = 0.20
    n_op_cond: int      = 6
    output_dir: str     = "/content/"
    save_hdf5: bool     = True
    save_parquet: bool  = True
    save_csv: bool      = False
    save_lf_csv: bool   = False
    save_zip: bool      = True
    seed: int           = 42
    verbose: bool       = True
    physics: PhysicsConfig         = field(default_factory=PhysicsConfig)
    degrad: DegradationConfig      = field(default_factory=DegradationConfig)
    fadec: FADECConfig             = field(default_factory=FADECConfig)
    maint: MaintenanceConfig       = field(default_factory=MaintenanceConfig)
    events: RareEventConfig        = field(default_factory=RareEventConfig)
    sensor_fail: SensorFailConfig  = field(default_factory=SensorFailConfig)
    fleet: FleetConfig             = field(default_factory=FleetConfig)
    pilot: PilotConfig             = field(default_factory=PilotConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    sampling: SamplingConfig       = field(default_factory=SamplingConfig)


# ══════════════════════════════════════════════════════════════════════════════
# §2  CAUSAL GRAPH  (extended for new nodes)
# ══════════════════════════════════════════════════════════════════════════════

class CausalGraph:
    NODES = ["T4","RPM","humidity","altitude","throttle",
             "D_fat","D_crp","D_cor","D_th",
             "eff_c","eff_t","crack_len","EGT","vibration","oil_temp","RUL",
             "flight_phase","fuel_contam","cross_EGT_delta"]   # 3 new nodes
    N = len(NODES)
    IDX = {n: i for i, n in enumerate(NODES)}

    EDGES = [
        ("T4","D_crp",0.90), ("T4","D_th",0.85), ("T4","eff_t",0.80),
        ("T4","EGT",0.95),   ("T4","oil_temp",0.55),
        ("RPM","D_fat",0.85),("RPM","crack_len",0.75),("RPM","vibration",0.70),
        ("RPM","D_th",0.60),
        ("humidity","D_cor",0.80),("humidity","eff_c",0.35),
        ("altitude","T4",0.40),  ("altitude","D_cor",0.30),
        ("throttle","T4",0.85),  ("throttle","RPM",0.90),
        ("D_fat","crack_len",0.90),("D_fat","vibration",0.60),("D_fat","RUL",0.95),
        ("D_crp","eff_t",0.70),   ("D_crp","oil_temp",0.50),("D_crp","RUL",0.80),
        ("D_cor","D_fat",0.40),   ("D_cor","eff_c",0.60),   ("D_cor","RUL",0.70),
        ("D_th","D_fat",0.65),    ("D_th","crack_len",0.55),("D_th","RUL",0.60),
        ("eff_c","EGT",0.70),     ("eff_t","EGT",0.80),
        ("crack_len","RUL",0.90), ("crack_len","vibration",0.50),
        ("oil_temp","RUL",0.60),  ("EGT","RUL",0.50),
        ("flight_phase","throttle",0.95),("flight_phase","T4",0.85),
        ("flight_phase","D_fat",0.70),   ("flight_phase","EGT",0.75),
        ("fuel_contam","EGT",0.80),      ("fuel_contam","D_th",0.50),
        ("cross_EGT_delta","RUL",0.40),
    ]

    @classmethod
    def adjacency(cls) -> np.ndarray:
        A = np.zeros((cls.N, cls.N), dtype=np.float32)
        for c, e, w in cls.EDGES:
            A[cls.IDX[c], cls.IDX[e]] = w
        return A

    @classmethod
    def to_dict(cls) -> dict:
        return {
            "nodes": cls.NODES,
            "edges": [{"from": c, "to": e, "w": w} for c, e, w in cls.EDGES],
            "adjacency": cls.adjacency().tolist()
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3  FLIGHT PHASE MANAGER  [NEW — Feature 1]
# ══════════════════════════════════════════════════════════════════════════════

class FlightPhaseManager:
    """
    Generates the sequence of flight phases within a single flight cycle.
    Each 'cycle' in the dataset represents one complete flight.
    Sub-samples N_SUBCYCLES points across the flight to capture
    within-flight sensor variation.

    Physics:
      T4 = f(phase, m_dot_fuel, altitude)
      throttle = PHASE_ENVELOPE[phase].thr +/- sigma + pilot_style
    """

    N_SUBCYCLES = 6   # number of sub-samples per flight stored per cycle

    def __init__(self, pilot_cfg: PilotConfig, rng: np.random.Generator):
        self.pc  = pilot_cfg
        self.rng = rng

    def sample_pilot_style(self) -> Dict:
        """Per-engine pilot style: drawn once at engine init."""
        rng = self.rng
        pc  = self.pc
        return {
            "ramp_rate":     float(rng.uniform(pc.ramp_rate_min, pc.ramp_rate_max)),
            "climb_thr":     float(rng.uniform(pc.climb_throttle_min, pc.climb_throttle_max)),
            "cruise_alt_var":float(rng.uniform(-pc.cruise_alt_var, pc.cruise_alt_var)),
            "surge_margin":  float(rng.uniform(pc.surge_margin_pref_min,
                                                pc.surge_margin_pref_max)),
        }

    def get_phase_conditions(self, phase: FlightPhase, base_env: Dict,
                              pilot: Dict) -> Dict:
        """
        Compute throttle/altitude for a given flight phase.
        Overrides base_env values based on phase physics.
        """
        env  = dict(base_env)
        ep   = PHASE_ENVELOPE[phase]
        rng  = self.rng

        # Throttle by phase
        if phase == FlightPhase.CLIMB:
            thr = float(np.clip(pilot["climb_thr"] + rng.normal(0, ep["thr_s"]), 0.4, 1.0))
        elif phase == FlightPhase.TAXI:
            thr = float(np.clip(ep["thr"] + rng.normal(0, ep["thr_s"]), 0.05, 0.25))
        elif phase == FlightPhase.TAKEOFF:
            thr = float(np.clip(ep["thr"] * pilot["ramp_rate"] + rng.normal(0, 0.01), 0.90, 1.0))
        else:
            thr = float(np.clip(ep["thr"] + rng.normal(0, ep["thr_s"]), 0.3, 1.0))

        # Altitude by phase
        base_alt = base_env["altitude"]
        alt = float(np.clip(base_alt * ep["alt_f"] * (1 + pilot["cruise_alt_var"]), 0, 12500))

        env["throttle"]    = thr
        env["altitude"]    = alt
        env["EGT_phase_bias"] = ep["EGT_b"]
        env["flight_phase"]   = int(phase)
        return env

    def flight_sequence(self, base_env: Dict, pilot: Dict) -> List[Tuple[FlightPhase, Dict]]:
        """Return ordered list of (phase, env) for one complete flight."""
        seq = []
        for phase in list(FlightPhase):
            env = self.get_phase_conditions(phase, base_env, pilot)
            seq.append((phase, env))
        return seq

    def representative_phase(self, cyc: int, total_cyc: int,
                              base_env: Dict, pilot: Dict) -> Tuple[FlightPhase, Dict, float]:
        """
        For each cycle pick a representative phase weighted by flight duration.
        Also returns thermal_delta (phase transition thermal shock).
        """
        rng = self.rng
        # Weight phases by duration fraction
        dur_w = np.array([PHASE_ENVELOPE[p]["dur"] for p in list(FlightPhase)])
        dur_w /= dur_w.sum()
        phase = FlightPhase(int(rng.choice(len(FlightPhase), p=dur_w)))
        env   = self.get_phase_conditions(phase, base_env, pilot)

        # Thermal delta: large at takeoff (cold taxi → hot takeoff)
        if phase == FlightPhase.TAKEOFF:
            thermal_delta = float(rng.normal(180.0, 20.0))
        elif phase == FlightPhase.CLIMB:
            thermal_delta = float(rng.normal(80.0, 15.0))
        elif phase == FlightPhase.LANDING:
            thermal_delta = float(rng.normal(60.0, 12.0))
        else:
            thermal_delta = float(rng.normal(20.0, 8.0))
        return phase, env, thermal_delta


# ══════════════════════════════════════════════════════════════════════════════
# §4  MISSION CONTEXT GENERATOR  [NEW — Feature 5]
# ══════════════════════════════════════════════════════════════════════════════

class MissionContext:
    """
    Generates per-flight mission-level context features:
      route_distance_nm, flight_duration_min, airport_elevation_ft,
      sand_exposure_index, salt_exposure_index
    These feed into corrosion and fouling rates.
    """

    def __init__(self, fleet: FleetConfig, rng: np.random.Generator):
        self.f   = fleet
        self.rng = rng

    def sample(self, route_type: str) -> Dict:
        rng = self.rng
        f   = self.f
        d_range = f.route_distances_nm.get(route_type, (500, 3000))
        e_range = f.airport_elevations_ft.get(route_type, (0, 2000))
        dist    = float(rng.uniform(*d_range))
        dur     = dist / 480.0 * 60.0   # assume 480 kt cruise → minutes
        elev    = float(rng.uniform(*e_range))
        sand    = float(np.clip(rng.normal(f.sand_exposure.get(route_type, 0.2), 0.05), 0, 1))
        salt    = float(np.clip(rng.normal(
            1.0 if route_type == "coastal" else 0.1, 0.05), 0, 1))
        # High elevation reduces air density → higher N for same thrust
        density_ratio = float(np.exp(-elev * 0.304 / 1000.0 / 8.435))
        return {
            "route_distance_nm": dist,
            "flight_duration_min": dur,
            "airport_elevation_ft": elev,
            "sand_exposure_index": sand,
            "salt_exposure_index": salt,
            "air_density_ratio": density_ratio,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §5  THERMODYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

class TransientThermo:
    def __init__(self, cfg: PhysicsConfig):
        self.c = cfg
        self._map = self._build_map()

    def _build_map(self) -> np.ndarray:
        rows = []
        for Nc_f in [0.70, 0.85, 1.00]:
            for mf in np.linspace(0.65, 1.05, 12):
                PR  = max(0.3, Nc_f**2 * (1.0 - 0.8*(mf - Nc_f)**2))
                eta = float(np.clip(np.exp(-4.0*(mf - Nc_f*0.97)**2), 0.30, 1.0))
                rows.append([Nc_f, mf, PR, eta])
        return np.array(rows)

    def _ambient(self, alt: float, dT: float) -> Tuple[float, float]:
        T_sl = 288.15 + dT
        if alt <= 11000.0:
            T = T_sl - 0.0065 * alt
            P = 101325.0 * (T / T_sl) ** 5.2561
        else:
            T11 = T_sl - 71.5
            P11 = 101325.0 * (T11 / T_sl) ** 5.2561
            T   = T11
            P   = P11 * np.exp(-9.80665 * (alt - 11000.0) / (287.05 * T11))
        return float(T), float(P)

    def _map_lookup(self, Nc_f: float, mf: float) -> Tuple[float, float, float]:
        m = self._map
        speeds = np.unique(m[:, 0])
        spd   = speeds[np.argmin(np.abs(speeds - Nc_f))]
        row   = m[m[:, 0] == spd]
        PR_n  = float(np.interp(mf, row[:, 1], row[:, 2]))
        eta_n = float(np.interp(mf, row[:, 1], row[:, 3]))
        m_sg  = float(row[:, 1].min()) * 0.95
        surge = (mf - m_sg) / max(1e-6, m_sg)
        return PR_n, eta_n, float(surge)

    def compute_cycle(self, alt: float, dT_ISA: float, throttle: float,
                      D_fat: float, D_crp: float, D_cor: float,
                      eff_c: float, eff_t: float, eta_comb: float,
                      mdot_scale: float, T_core_prev: float,
                      fuel_trim: float = 0.0, bleed_active: bool = False,
                      phase_EGT_bias: float = 0.0,
                      fuel_contam_factor: float = 0.0,
                      density_ratio: float = 1.0) -> Dict[str, float]:
        c  = self.c
        T0, P0 = self._ambient(alt, dT_ISA)
        P0 *= density_ratio   # airport elevation density effect
        T2  = T0
        P2  = P0 * (0.995 if alt > 5000 else 1.0)

        theta = T0 / c.T_ref
        delta = P0 / c.P_ref
        Nc_f  = float(np.clip(np.sqrt(theta) * mdot_scale, 0.5, 1.1))
        mf    = float(np.clip(mdot_scale * delta / np.sqrt(theta), 0.5, 1.15))

        PR_n, _, surge_margin = self._map_lookup(
            Nc_f, mf * (0.92 if bleed_active else 1.0))
        OPR  = float(np.clip(
            c.OPR_design * PR_n * (0.6 + 0.4*throttle**0.8)
            * (0.92 if bleed_active else 1.0), 5.0, 50.0))

        tau  = OPR**((c.gamma_c - 1.0)/c.gamma_c)
        T3   = T2 * (1.0 + (tau - 1.0)/max(eff_c, 0.60))
        P3   = P2 * OPR
        W_c  = c.cp_c * (T3 - T2)

        T4_tgt = c.T4_max * (0.70 + 0.30*throttle)
        f_far  = float(np.clip(
            (c.cp_t*T4_tgt - c.cp_c*T3) / (c.LHV*eta_comb*(1-fuel_contam_factor)
                                              - c.cp_t*T4_tgt)
            * (1.0 + fuel_trim), 0.01, 0.045))
        mdot  = c.mdot_core * mdot_scale * (P0/c.P_ref) * np.sqrt(c.T_ref/T0)
        fm    = f_far * mdot
        T4    = float(min(c.T4_max,
                          (c.cp_c*T3 + f_far*c.LHV*eta_comb*(1-fuel_contam_factor))
                          / (c.cp_t*(1.0+f_far))))
        P4    = P3 * 0.96

        gt   = c.gamma_t
        P45  = P4 / (OPR**0.45); P5 = P2 * 1.05
        T45  = T4  - eff_t*T4  *(1.0-(P45/P4)**((gt-1)/gt))
        T5   = T45 - eff_t*0.97*T45*(1.0-(P5/P45)**((gt-1)/gt))
        WHPT = c.cp_t*(T4  - T45)
        WLPT = c.cp_t*(T45 - T5)

        Wnet = WHPT + WLPT - W_c
        Nrpm = c.N_design * float(np.clip(
            np.sqrt(max(0.0, Wnet/max(1.0, W_c)))*throttle*0.85+0.15, 0.5, 1.05))

        Qin   = fm * c.LHV * eta_comb
        Wout  = (WHPT + WLPT) * mdot
        Qloss = c.heat_loss_coeff * (T_core_prev - T0)
        T_core = float(np.clip(
            T_core_prev + (Qin - Wout - Qloss)/c.thermal_mass_core * 0.05, T2, c.T4_max))

        EGT  = T5 - 50.0*(alt/10000.0) + 0.03*(T_core - T4) + phase_EGT_bias
        SFC  = fm / max(1.0, mdot*(1.0+f_far)*400.0*throttle) * 3600.0
        Nf   = Nrpm / c.N_design
        foul = max(0.0, 1.0 - eff_c/c.eta_c0)
        vib  = float(np.clip(0.05+0.03*Nf**2+0.01*foul+0.02*D_fat**1.5, 0, 3))
        sigma_n = max(0.0, Nf**2*(1.0+0.3*(T4/c.T4_max-0.7)))
        oil  = float(np.clip(T0+85.0*(D_crp+0.5*D_fat)*sigma_n, 250, 600))

        return dict(T2=T2,P2=P2,T3=T3,P3=P3,T4=T4,P4=P4,T45=T45,T5=T5,P5=P5,
                    EGT=EGT,N_rpm=Nrpm,mdot=mdot,fuel_mdot=fm,FAR=f_far,SFC=SFC,
                    W_c=W_c,W_t=WHPT+WLPT,OPR=OPR,vibration=vib,oil_temp=oil,
                    T_core=T_core,surge_margin=float(surge_margin),
                    stall_active=float(surge_margin < 0.0),
                    bleed_active=float(bleed_active),fouling_degrad=foul,
                    eta_c_eff=eff_c,eta_t_eff=eff_t,altitude=alt,throttle=throttle)


# ══════════════════════════════════════════════════════════════════════════════
# §6  DEGRADATION ENGINE  (phase-aware, mission-aware)
# ══════════════════════════════════════════════════════════════════════════════

N_SENSORS = 20  # +2 new channels: cross_delta_EGT, cross_delta_RPM

class DegradationEngine:
    def __init__(self, cfg: PhysicsConfig, deg: DegradationConfig,
                 rng: np.random.Generator):
        self.p   = cfg
        self.d   = deg
        self.rng = rng
        self.hi_w = np.array([0.22, 0.16, 0.10, 0.08, 0.10, 0.08, 0.10, 0.08, 0.08])

    def init_state(self, mfg: Dict) -> Dict:
        a0    = float(np.clip(self.rng.normal(self.p.a0_mean, self.p.a0_sigma),
                               1e-5, self.p.a0_sigma*4))
        eff_c = float(np.clip(self.p.eta_c0*mfg["eta_c"],
                               self.p.fouling_eff_min+0.06, self.p.eta_c0*1.01))
        eff_t = float(np.clip(self.p.eta_t0*mfg["eta_t"], 0.73, self.p.eta_t0*1.01))
        mdot  = float(np.clip(mfg["mdot"], 0.87, 1.13))
        return {
            "D_fat": 0.0, "D_crp": 0.0, "D_cor": 0.0, "D_th": 0.0,
            "eff_c": eff_c, "eff_t": eff_t,
            "eta_comb": self.p.eta_comb0, "mdot_scale": mdot,
            "crack_len": a0, "disk_burst_risk": 0.0, "fracture": False,
            "drift": np.zeros(N_SENSORS, dtype=np.float64),
            "drift_cum": np.zeros(N_SENSORS, dtype=np.float64),
            "T_core": 600.0, "last_T4": 1400.0,
            "in_stall": False,    "stall_rem": 0,
            "in_oil_fail": False, "oil_excess": 0.0,
            "in_seal_leak": False,"seal_acc": 0.0,
            "in_fuel_contam": False, "fuel_contam_rem": 0,
            "time_since_maint": 0,
            "events": [], "cycle": 0, "last_EGT": 800.0,
        }

    def _fatigue(self, sigma_n: float, D_cor: float, DK: float, R: float,
                 phase: FlightPhase) -> float:
        d  = self.d
        R  = float(np.clip(R, 0.0, 0.9))
        DKe = DK / max(1e-9, (1.0-R)**(1.0-d.walker_c))
        dD  = d.C_fat * DKe**d.m_fat * (1.0 + d.cor_fat_coupling*D_cor)
        # Takeoff is highest stress per cycle
        if phase == FlightPhase.TAKEOFF:
            dD *= d.takeoff_fat_mult
        elif phase == FlightPhase.CLIMB:
            dD *= 1.8
        elif phase == FlightPhase.TAXI:
            dD *= 0.15
        return float(np.clip(dD * self.rng.lognormal(0.0, 0.08), 0.0, 0.06))

    def _creep(self, sigma_n: float, T4: float, D_fat: float) -> float:
        d = self.d
        r = (d.A_crp * max(0.0, sigma_n)**d.n_crp *
             np.exp(-d.Q_crp/(d.R_univ*max(T4,300.0))) *
             (1.0 + d.fat_crp_coupling*D_fat))
        return float(np.clip(r, 0.0, 0.008))

    def _thermal_fatigue(self, DT: float, phase: FlightPhase) -> float:
        d  = self.d
        mult = self.d.thermal_shock_mult if phase == FlightPhase.TAKEOFF else 1.0
        return 0.0 if DT <= 0 else float(d.C_th*(DT/d.DT_ref)**d.m_th * mult)

    def _corrosion(self, hum: float, T_hot: float, salt: float,
                   sand: float) -> float:
        d = self.d
        r = (d.k_cor * hum *
             np.exp(-d.Q_cor/(d.R_univ*max(T_hot,300.0))) *
             (1.0 + (d.salt_coeff-1.0)*salt) *
             (1.0 + 0.5*sand))   # sand abrades protective oxide
        return float(np.clip(r, 0.0, 0.005))

    def _crack(self, a: float, sigma_n: float) -> Tuple[float, float]:
        K   = 1.12*350.0*sigma_n*np.sqrt(np.pi*max(a,1e-9))
        da  = self.p.paris_C*K**self.p.paris_m * self.rng.lognormal(0.0, 0.12)
        new = a + float(np.clip(da, 0.0, 0.001))
        brisk = float(np.clip((new/self.p.a_crit)**2, 0.0, 1.0))
        return new, brisk

    def _apply_events(self, state: Dict, ev: RareEventConfig, cyc: int,
                       phase: FlightPhase) -> Tuple[Dict, Dict]:
        rng  = self.rng
        mods = dict(EGT_delta=0.0, vib_delta=0.0, D_fat_boost=0.0,
                    P3_scale=1.0, RPM_scale=1.0, oil_temp_delta=0.0,
                    fuel_contam_factor=0.0, event_name=None)
        s = state

        # Bird strike (more likely at takeoff/landing)
        phase_mult = 3.0 if phase in (FlightPhase.TAKEOFF, FlightPhase.LANDING) else 1.0
        if rng.random() < ev.bird_strike_prob * phase_mult:
            mods["EGT_delta"]   += ev.bird_EGT_spike
            mods["vib_delta"]   += ev.bird_vib_shock
            mods["D_fat_boost"] += ev.bird_fat_boost
            mods["RPM_scale"]   *= 0.97
            mods["event_name"]   = "bird_strike"
            s["events"].append({"c": cyc, "t": "bird_strike", "phase": int(phase)})

        if not s.get("in_stall", False) and rng.random() < ev.stall_prob:
            s["in_stall"]  = True
            s["stall_rem"] = ev.stall_cycles
            s["events"].append({"c": cyc, "t": "comp_stall"})
        if s.get("in_stall", False):
            ph2   = s["stall_rem"] / max(1, ev.stall_cycles)
            mods["P3_scale"]  *= (1.0 - ev.stall_P3_drop*np.sin(np.pi*(1-ph2)))
            mods["EGT_delta"] += 30.0
            mods["vib_delta"] += 0.08
            s["stall_rem"] -= 1
            if s["stall_rem"] <= 0:
                s["in_stall"] = False
                mods["D_fat_boost"] += 0.03

        if not s.get("in_oil_fail", False) and rng.random() < ev.oil_fail_prob:
            s["in_oil_fail"] = True
            s["oil_excess"]  = 0.0
            s["events"].append({"c": cyc, "t": "oil_failure"})
        if s.get("in_oil_fail", False):
            s["oil_excess"] = min(s["oil_excess"]+ev.oil_temp_rate,
                                   ev.oil_temp_rate*ev.oil_fail_max_cyc)
            mods["oil_temp_delta"] = s["oil_excess"]
            if s["oil_excess"] >= ev.oil_temp_rate*ev.oil_fail_max_cyc*0.8:
                s["in_oil_fail"] = False
                mods["EGT_delta"] += 25.0

        if not s.get("in_seal_leak", False) and rng.random() < ev.seal_prob:
            s["in_seal_leak"] = True
            s["seal_acc"]     = 0.0
            s["events"].append({"c": cyc, "t": "seal_leak"})
        if s.get("in_seal_leak", False):
            s["seal_acc"] = min(s["seal_acc"]+ev.seal_decay, ev.seal_decay*ev.seal_max_cyc)
            mods["P3_scale"] *= max(0.70, 1.0-s["seal_acc"])
            if s["seal_acc"] >= ev.seal_decay*ev.seal_max_cyc:
                s["in_seal_leak"] = False

        # Fuel contamination (Feature 6)
        if not s.get("in_fuel_contam", False) and rng.random() < ev.fuel_contam_prob:
            s["in_fuel_contam"]  = True
            s["fuel_contam_rem"] = ev.fuel_contam_dur
            s["events"].append({"c": cyc, "t": "fuel_contamination"})
        if s.get("in_fuel_contam", False):
            mods["fuel_contam_factor"] = ev.fuel_contam_SFC
            mods["EGT_delta"] += ev.fuel_contam_EGT * rng.uniform(0.6, 1.4)
            s["fuel_contam_rem"] -= 1
            if s["fuel_contam_rem"] <= 0:
                s["in_fuel_contam"] = False

        if rng.random() < ev.ash_prob:
            mods["D_fat_boost"] += 0.005
            mods["vib_delta"]   += 0.04
            s["events"].append({"c": cyc, "t": "volcanic_ash"})
            s["D_cor"] = float(min(1.0, s.get("D_cor",0.0)+0.003*ev.ash_cor_mult))

        if rng.random() < ev.sand_prob:
            s["eff_c"] = max(self.p.fouling_eff_min,
                              s.get("eff_c", self.p.eta_c0)-ev.sand_eta_drop)
            s["events"].append({"c": cyc, "t": "sand_erosion"})

        if rng.random() < ev.fod_prob:
            s["crack_len"] = float(min(self.p.a_crit,
                                        s.get("crack_len",self.p.a0_mean)+ev.fod_crack_boost))
            mods["vib_delta"] += ev.fod_vib
            s["events"].append({"c": cyc, "t": "fod_damage"})

        if rng.random() < ev.ice_prob:
            mods["EGT_delta"] += ev.ice_EGT
            mods["vib_delta"] += 0.03
            s["events"].append({"c": cyc, "t": "ice_ingestion"})

        if rng.random() < ev.rain_prob:
            mods["EGT_delta"] -= ev.rain_EGT_drop

        if rng.random() < ev.crosswind_prob:
            mods["vib_delta"] += ev.crosswind_vib

        return s, mods

    def step(self, state: Dict, cs: Dict, env: Dict,
             ev_cfg: RareEventConfig, cyc: int,
             phase: FlightPhase,
             mission: Dict) -> Tuple[Dict, Dict]:
        new          = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        new["drift"]     = state["drift"].copy()
        new["drift_cum"] = state["drift_cum"].copy()
        new["events"]    = list(state["events"])

        new, mods = self._apply_events(new, ev_cfg, cyc, phase)

        T4    = cs["T4"] + mods["EGT_delta"] * 0.3
        T3    = cs["T3"]
        N_rpm = cs["N_rpm"]
        alt   = float(env.get("altitude", 10000.0))
        hum   = float(np.clip(env.get("humidity", 0.4), 0.0, 1.0))
        salt  = float(env.get("salt_factor", 0.0))
        sand  = float(mission.get("sand_exposure_index", 0.2))
        sigma_n = max(0.0, (N_rpm/self.p.N_design)**2 *
                      (1.0+0.3*(T4/self.p.T4_max-0.7)))

        K_max = 1.12*350.0*sigma_n*np.sqrt(np.pi*max(state["crack_len"],1e-9))
        K_min = K_max*0.1
        DK    = K_max - K_min
        R     = K_min/max(K_max,1e-12)
        DT    = abs(T4 - state.get("last_T4", T4))

        dD_fat = self._fatigue(sigma_n, state.get("D_cor",0.0), DK, R, phase) + mods["D_fat_boost"]
        dD_crp = self._creep(sigma_n, T4, state["D_fat"])
        dD_th  = self._thermal_fatigue(DT, phase)
        dD_cor = self._corrosion(hum, 0.5*T3+0.5*T4, salt, sand)

        new["D_fat"] = float(min(1.0, state["D_fat"]  + dD_fat))
        new["D_crp"] = float(min(1.0, state["D_crp"]  + dD_crp))
        new["D_th"]  = float(min(1.0, state["D_th"]   + dD_th))
        new["D_cor"] = float(min(1.0, state.get("D_cor",0.0) + dD_cor))

        part     = hum * max(0.0, 1.0 - alt/20000.0) * (1.0 + sand)
        foul_r   = (self.d.k_foul *
                    max(0.0, state["eff_c"]-self.p.fouling_eff_min) *
                    part**self.d.foul_exp)
        new["eff_c"] = float(np.clip(state["eff_c"]-foul_r,
                                      self.p.fouling_eff_min, self.p.eta_c0))
        new["eff_t"] = float(max(0.72, state["eff_t"] -
                                  0.8e-6*(state["D_crp"]+0.5*state["D_fat"])*sigma_n))

        new["crack_len"], new["disk_burst_risk"] = self._crack(state["crack_len"], sigma_n)
        if new["crack_len"] >= self.p.a_crit:
            new["fracture"] = True

        inc = self.rng.normal(0.0, self.d.drift_vol, N_SENSORS)
        new["drift_cum"] += inc
        new["drift"]      = new["drift_cum"].copy()

        new["T_core"]        = cs.get("T_core", state["T_core"])
        new["last_T4"]       = T4
        new["last_EGT"]      = cs["EGT"] + mods["EGT_delta"]
        new["time_since_maint"] = state.get("time_since_maint", 0) + 1
        new["cycle"]         = state["cycle"] + 1
        return new, mods

    def health_index(self, s: Dict) -> float:
        eff_deg  = max(0.0, 1.0 - s["eff_c"]/self.p.eta_c0)
        sens_deg = float(np.clip(np.mean(np.abs(s["drift_cum"]))/0.1, 0.0, 1.0))
        crack_f  = float(s["crack_len"]/self.p.a_crit)
        comps    = np.array([s["D_fat"], s["D_crp"], s.get("D_cor",0.0), s["D_th"],
                             eff_deg, sens_deg, s["disk_burst_risk"],
                             float(s.get("fracture",False)), crack_f])
        w = self.hi_w / self.hi_w.sum()
        return float(np.clip(1.0 - np.dot(w, comps), 0.0, 1.0))

    def failure_probability(self, hi: float, cfg: UncertaintyConfig) -> float:
        """
        Feature 10: Compute instantaneous failure probability from health index.
        Uses a logistic function calibrated so P(fail) ≈ 0.5 at HI = hi_thresh.
        """
        thresh = cfg.failure_hi_thresh
        k = 10.0  # steepness
        pf = 1.0 / (1.0 + np.exp(k * (hi - thresh)))
        return float(np.clip(pf, 0.0, 1.0))

    def is_failed(self, s: Dict) -> bool:
        return (s["D_fat"] >= 0.98 or s["D_crp"] >= 0.98 or
                s.get("D_cor",0.0) >= 0.95 or
                s["eff_c"] <= self.p.fouling_eff_min + 0.002 or
                s.get("fracture", False) or
                self.health_index(s) < 0.02)

    def pinn_residuals(self, hist: List[Dict]) -> Dict[str, np.ndarray]:
        n = len(hist) - 1
        if n < 1:
            return {}
        keys = ["D_fat","D_crp","D_cor","D_th","eff_c","crack_len"]
        out  = {f"res_{k}": np.zeros(n, np.float32) for k in keys}
        for i in range(n):
            for k in keys:
                out[f"res_{k}"][i] = float(hist[i+1].get(k,0) - hist[i].get(k,0))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# §7  FADEC
# ══════════════════════════════════════════════════════════════════════════════

class FADEC:
    def __init__(self, cfg: FADECConfig, pcfg: PhysicsConfig,
                 pilot: Dict):
        self.cfg  = cfg
        self.p    = pcfg
        self.pilot = pilot
        self.reset()

    def reset(self):
        self._thr      = 0.85
        self._trim     = 0.0
        self._shutdown = False
        self._overtemp_count = 0

    def update(self, thr_demand: float, EGT: float, N_rpm: float,
               eff_c: float, surge_margin: float) -> Tuple[float,float,bool,Dict]:
        cfg = self.cfg; flags = {}
        # Pilot-style ramp rate modifies throttle lag
        eff_rate = cfg.throttle_rate_limit * self.pilot.get("ramp_rate", 1.0)
        d  = float(np.clip(thr_demand - self._thr, -eff_rate, eff_rate))
        self._thr = float(np.clip(self._thr + d/cfg.throttle_lag_tau, 0.4, 1.0))

        if EGT > cfg.EGT_redline:
            self._overtemp_count += 1
            if self._overtemp_count >= cfg.overtemp_cycles_limit:
                self._shutdown = True
                flags["emergency_shutdown"] = True
        else:
            self._overtemp_count = 0

        if EGT > cfg.EGT_amber and not self._shutdown:
            ratio = (EGT-cfg.EGT_amber)/(cfg.EGT_redline-cfg.EGT_amber)
            self._trim = -cfg.fuel_trim_auth * float(np.clip(ratio, 0, 1))
            flags["EGT_protect"] = True
        else:
            self._trim *= 0.85

        if N_rpm > cfg.N_overspeed * self.p.N_design:
            self._thr = min(self._thr, 0.90)
            flags["overspeed"] = True

        # Pilot surge margin preference
        sm_pref = self.pilot.get("surge_margin", cfg.surge_margin_min)
        bleed   = (eff_c/self.p.eta_c0 < cfg.bleed_eff_ratio or
                   surge_margin < sm_pref)
        if bleed:
            flags["bleed_valve"] = True

        return self._thr, self._trim, bleed, flags

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown


# ══════════════════════════════════════════════════════════════════════════════
# §8  MAINTENANCE MANAGER  [Feature 2 — extended]
# ══════════════════════════════════════════════════════════════════════════════

class MaintenanceManager:
    MAINT_TYPES = ["wash", "blade_repair", "module_replace", "repair",
                   "sensor_rep", "miscal"]

    def __init__(self, cfg: MaintenanceConfig, pcfg: PhysicsConfig,
                 aggressiveness: float, rng: np.random.Generator):
        self.cfg = cfg; self.p = pcfg
        self.agg = aggressiveness; self.rng = rng
        self.log: List[Dict] = []
        self._last_maint: int = 0

    def reset(self):
        self.log = []; self._last_maint = 0

    def apply(self, state: Dict, cyc: int) -> Tuple[Dict, List[str], str]:
        cfg = self.cfg; rng = self.rng; agg = self.agg
        new = dict(state); evts = []; maint_type_flag = "none"

        # Compressor wash
        if rng.random() < cfg.wash_prob * (1+agg):
            q   = float(np.clip(rng.lognormal(0, cfg.wash_quality_sig), 0.4, 2.0))
            rec = q * cfg.wash_recovery * max(0, self.p.eta_c0*0.97 - state["eff_c"])
            new["eff_c"] = float(min(self.p.eta_c0*0.98, state["eff_c"] + rec))
            self.log.append({"c": cyc, "t": "wash", "q": q})
            evts.append("wash"); maint_type_flag = "wash"
            new["time_since_maint"] = 0

        # Blade repair (Feature 2) — heals crack and fatigue damage
        if rng.random() < cfg.blade_repair_prob * agg:
            crack_heal = state["crack_len"] * cfg.blade_repair_crack_frac
            fat_heal   = state["D_fat"]     * cfg.blade_repair_fat_frac
            new["crack_len"] = float(max(self.p.a0_mean, state["crack_len"] - crack_heal))
            new["D_fat"]     = float(max(0.0, state["D_fat"] - fat_heal))
            self.log.append({"c": cyc, "t": "blade_repair"})
            evts.append("blade_repair"); maint_type_flag = "blade_repair"
            new["time_since_maint"] = 0

        # Module replacement (Feature 2) — major event
        if rng.random() < cfg.module_replace_prob:
            new["eff_c"] = float(min(self.p.eta_c0, state["eff_c"] +
                                      cfg.module_replace_eff_c_reset*(self.p.eta_c0 - state["eff_c"])))
            new["eff_t"] = float(min(self.p.eta_t0, state["eff_t"] +
                                      cfg.module_replace_eff_t_reset*(self.p.eta_t0 - state["eff_t"])))
            new["D_fat"] = float(state["D_fat"] * 0.3)
            new["D_crp"] = float(state["D_crp"] * 0.4)
            new["crack_len"] = self.p.a0_mean
            self.log.append({"c": cyc, "t": "module_replace"})
            evts.append("module_replace"); maint_type_flag = "module_replace"
            new["time_since_maint"] = 0

        if rng.random() < cfg.repair_prob * agg:
            q    = float(np.clip(rng.lognormal(0, cfg.repair_frac_sig), 0.2, 2.0))
            frac = float(rng.uniform(0.05, 0.30)) * q
            new["D_fat"] = float(max(0, state["D_fat"] - frac*0.3))
            new["D_crp"] = float(max(0, state["D_crp"] - frac*0.2))
            new["eff_t"] = float(min(self.p.eta_t0*0.97, state["eff_t"]+frac*0.04*q))
            self.log.append({"c": cyc, "t": "repair", "q": q})
            evts.append("repair")

        if rng.random() < cfg.sensor_rep_prob:
            chs = rng.choice(N_SENSORS, size=cfg.sensor_rep_n, replace=False)
            nd  = new["drift"].copy(); nc = new["drift_cum"].copy()
            nd[chs] = 0.0; nc[chs] = 0.0
            new["drift"] = nd; new["drift_cum"] = nc
            self.log.append({"c": cyc, "t": "sensor_rep"})
            evts.append("sensor_rep")

        if rng.random() < cfg.miscal_prob:
            n   = int(rng.integers(1, 5))
            chs = rng.choice(N_SENSORS, size=n, replace=False)
            mag = float(rng.normal(0, cfg.miscal_sig))
            nd  = new["drift"].copy(); nd[chs] += mag
            new["drift"] = nd
            self.log.append({"c": cyc, "t": "miscal"})
            evts.append("miscal")

        return new, evts, maint_type_flag


# ══════════════════════════════════════════════════════════════════════════════
# §9  SENSOR LAYER  [Feature 3 & 8 — 20 channels, gradual drift fault,
#                    per-engine calibration bias]
# ══════════════════════════════════════════════════════════════════════════════

class SensorLayer:
    NAMES = ["T2","T24","T30","T50","P2","P15","P30","Nf","Nc","EPR",
             "Ps30","phi","NRc","BPR","vib_rms","oil_temp","EGT_direct",
             "surge_margin","cross_delta_EGT","cross_delta_RPM"]
    UNITS = ["K","K","K","K","Pa","Pa","Pa","RPM","RPM","—",
             "Pa","kg/s","RPM","—","g","K","K","—","K","RPM"]
    N = N_SENSORS   # 20

    _SIGMA = np.array([
        0.80, 0.96, 1.44, 0.88,
        150.0, 120.0, 300.0,
        12.0, 12.0, 0.001, 135.0,
        0.15, 8.4, 0.01,
        0.015, 2.0, 1.0, 0.005,
        2.0, 8.0   # cross-engine deltas
    ])

    def __init__(self, fail_cfg: SensorFailConfig, rng: np.random.Generator):
        self.fc  = fail_cfg
        self.rng = rng
        self._stuck_val: Dict[int,float] = {}
        self._stuck_rem: Dict[int,int]   = {}
        self._delay_buf: Dict[int,List]  = {}
        self._sat_max: Optional[np.ndarray] = None
        # Feature 3: gradual drift fault channels
        self._drift_fault_ch: List[int]   = []
        self._drift_fault_rate: float     = 0.0
        self._drift_fault_accum: np.ndarray = np.zeros(self.N)
        # Feature 8: per-engine calibration bias
        self._cal_bias: np.ndarray = np.zeros(self.N)

    def reset(self):
        self._stuck_val = {}; self._stuck_rem = {}
        self._delay_buf = {}; self._sat_max = None
        self._drift_fault_accum = np.zeros(self.N)
        self._drift_fault_ch    = []

    def init_calibration_bias(self) -> np.ndarray:
        """Feature 8: unique per-engine calibration offset b_i."""
        b = self.rng.normal(0.0, 0.004, self.N)
        self._cal_bias = b
        return b

    def init_drift_faults(self):
        """Feature 3: assign gradual drift fault channels for this engine."""
        fc  = self.fc
        rng = self.rng
        if rng.random() < fc.drift_fault_prob:
            n_ch = int(rng.integers(1, fc.drift_fault_max_sensors+1))
            self._drift_fault_ch   = list(rng.choice(self.N, n_ch, replace=False))
            self._drift_fault_rate = float(rng.uniform(
                fc.drift_fault_rate*0.5, fc.drift_fault_rate*2.0))

    def bias(self) -> np.ndarray:
        """Standard noise bias + calibration bias."""
        return self.rng.normal(0.0, 0.003, self.N) + self._cal_bias

    def ground_truth(self, cs: Dict, ds: Dict,
                     cross_delta_EGT: float = 0.0,
                     cross_delta_RPM: float = 0.0) -> np.ndarray:
        T2=cs["T2"]; T3=cs["T3"]; T4=cs["T4"]; T5=cs["T5"]
        P2=cs["P2"]; P3=cs["P3"]; Nr=cs["N_rpm"]
        bpr  = 8.5*(1.0-0.02*ds["D_crp"])
        T24  = T2+0.4*(T3-T2)*(ds["eff_c"]/0.88)
        EPR  = P3/P2*0.98
        return np.array([
            T2, T24, T3, T5, P2, P2*1.6, P3,
            Nr*0.95, Nr, EPR, P3*0.97, cs["fuel_mdot"],
            Nr/np.sqrt(T3/288.15), bpr,
            cs["vibration"], cs["oil_temp"], cs["EGT"], cs["surge_margin"],
            cross_delta_EGT, cross_delta_RPM
        ], dtype=np.float64)

    def add_noise(self, gt: np.ndarray, bias: np.ndarray,
                  drift: np.ndarray, mods: Dict,
                  fault_flags_out: np.ndarray,
                  cycle: int) -> np.ndarray:
        rng = self.rng; fc = self.fc; n = len(gt)
        sig = self._SIGMA

        # Gradual drift fault accumulation (Feature 3)
        if self._drift_fault_ch:
            self._drift_fault_accum[self._drift_fault_ch] += self._drift_fault_rate
            fault_flags_out[0] = 1.0
            fault_flags_out[1] = float(len(self._drift_fault_ch))

        noise  = rng.normal(0.0, 1.0, n) * sig
        spk    = rng.random(n) < 0.05
        noise += rng.standard_t(6.0, n)*sig*0.8*spk

        noisy  = (gt + noise + drift*gt*0.05 + bias*gt
                  + self._drift_fault_accum)    # gradual drift added
        noisy[16] += mods.get("EGT_delta", 0.0)
        noisy[6]  *= mods.get("P3_scale",  1.0)
        noisy[14] += mods.get("vib_delta", 0.0)

        # Stuck sensor fault (Feature 3)
        for ch in range(n):
            if self._stuck_rem.get(ch, 0) > 0:
                noisy[ch] = self._stuck_val[ch]
                self._stuck_rem[ch] -= 1
                fault_flags_out[2] = 1.0
                fault_flags_out[3] = float(ch)
            elif rng.random() < fc.stuck_prob/n:
                self._stuck_val[ch] = float(noisy[ch])
                self._stuck_rem[ch] = int(rng.integers(fc.stuck_min, fc.stuck_max))

        for ch in range(n):
            if rng.random() < fc.delay_prob/n:
                buf = self._delay_buf.setdefault(ch, [])
                buf.append(float(noisy[ch]))
                delay = int(rng.integers(1, fc.delay_max+1))
                if len(buf) > delay:
                    noisy[ch] = buf.pop(0)

        for ch in range(n):
            if rng.random() < fc.quant_prob/n:
                step = (abs(gt[ch])*2.0+1e-6)/(2**fc.quant_bits)
                noisy[ch] = np.round(noisy[ch]/step)*step

        if self._sat_max is None:
            self._sat_max = np.abs(gt)*1.1
        for ch in range(n):
            if rng.random() < fc.sat_prob/n:
                cap = self._sat_max[ch]*fc.sat_frac
                noisy[ch] = float(np.clip(noisy[ch], -cap, cap))

        spk2 = rng.random(n) < fc.spike_prob
        if spk2.any():
            noisy[spk2] += rng.normal(0,1,spk2.sum())*sig[spk2]*fc.spike_mult
            fault_flags_out[4] = 1.0

        # Intermittent dropout (Feature 3)
        missing = rng.random(n) < fc.missing_prob
        noisy[missing] = np.nan
        if missing.any():
            fault_flags_out[5] = float(missing.sum())

        return noisy


# ══════════════════════════════════════════════════════════════════════════════
# §10  FLIGHT ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class FlightEnvironment:
    _OPC = {
        0: dict(alt=10600,as_=400, thr=0.83,ts=0.04,hum=0.30),
        1: dict(alt=8500, as_=600, thr=0.88,ts=0.05,hum=0.40),
        2: dict(alt=5000, as_=800, thr=0.92,ts=0.06,hum=0.50),
        3: dict(alt=11000,as_=300, thr=0.80,ts=0.03,hum=0.25),
        4: dict(alt=3000, as_=1200,thr=0.95,ts=0.04,hum=0.60),
        5: dict(alt=9000, as_=500, thr=0.85,ts=0.05,hum=0.35),
    }
    _RM = {
        "desert":  dict(hs=0.3, dT=+15, salt=0.0),
        "coastal": dict(hs=1.6, dT=+5,  salt=1.0),
        "arctic":  dict(hs=0.5, dT=-20, salt=0.0),
        "tropical":dict(hs=1.8, dT=+10, salt=0.5),
        "highland":dict(hs=0.7, dT=-5,  salt=0.0),
        "mixed":   dict(hs=1.0, dT=0,   salt=0.0),
    }

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def sample(self, op: int, cyc: int, route: str, util: float) -> Dict:
        cl   = self._OPC[op]
        mod  = self._RM.get(route, self._RM["mixed"])
        rng  = self.rng
        season = 5.0*np.sin(2*np.pi*cyc/365.0)
        alt  = float(np.clip(rng.normal(cl["alt"],cl["as_"]), 0, 12500))
        thr  = float(np.clip(rng.normal(cl["thr"],cl["ts"])*util, 0.55, 1.0))
        hum  = float(np.clip(rng.normal(cl["hum"]*mod["hs"]*max(0.1,1-alt/15000),0.08),0,1))
        dT   = float(rng.normal(5.0+season+mod["dT"], 12.0))
        return dict(altitude=alt, throttle=thr, humidity=hum,
                    dT_ISA=dT, salt_factor=float(mod["salt"]), op_cond=op,
                    route=route, utilization=util)


# ══════════════════════════════════════════════════════════════════════════════
# §11  TIMESTAMP GENERATOR  [Feature 9]
# ══════════════════════════════════════════════════════════════════════════════

class TimestampGenerator:
    """
    Generates realistic irregular timestamps for each cycle.
    Simulates ACARS/FOQA data acquisition:
      - Base interval: ~60 min between flight cycles
      - Jitter: ±8 min normally
      - Dropout: some cycles have no valid timestamp (NaN)
    """

    def __init__(self, cfg: SamplingConfig, rng: np.random.Generator,
                 engine_id: int):
        self.cfg = cfg
        self.rng = rng
        # Each engine starts at a random offset from epoch (fleet deployment spread)
        self._t = cfg.epoch_start + float(rng.uniform(0, 365*24*3600))

    def next(self) -> Tuple[float, float]:
        """Return (timestamp_unix, sampling_interval_min)."""
        cfg = self.cfg; rng = self.rng
        interval = float(np.clip(rng.normal(cfg.sample_interval_min,
                                             cfg.sample_interval_sigma), 10, 360))
        self._t += interval * 60.0
        ts = self._t if rng.random() > cfg.dropout_prob else np.nan
        return ts, interval


# ══════════════════════════════════════════════════════════════════════════════
# §12  UNCERTAINTY QUANTIFIER  [Feature 10 — extended]
# ══════════════════════════════════════════════════════════════════════════════

class UncertaintyQ:
    def __init__(self, cfg: UncertaintyConfig, rng: np.random.Generator):
        self.cfg = cfg; self.rng = rng

    def rul_dist(self, RUL: np.ndarray, HI: np.ndarray,
                  env_hum: np.ndarray,
                  fail_probs: np.ndarray) -> Dict[str, np.ndarray]:
        cfg   = self.cfg
        alea  = RUL*cfg.alea_frac*(1.0+0.3*(1.0-HI)) + 1.0
        epi   = RUL*cfg.epi_frac *(1.0+0.2*env_hum)
        total = np.sqrt(alea**2 + epi**2).astype(np.float32)
        out   = {
            "RUL_mean":  RUL.astype(np.float32),
            "RUL_std":   total,
            "RUL_alea":  alea.astype(np.float32),
            "RUL_epi":   epi.astype(np.float32),
            # Feature 10: explicit lower/upper bounds
            "RUL_lower": np.clip(RUL - 1.96*total, 0, None).astype(np.float32),
            "RUL_upper": (RUL + 1.96*total).astype(np.float32),
            "failure_prob": fail_probs.astype(np.float32),
        }
        for lv in cfg.ci_levels:
            z   = float(scipy_norm.ppf(lv))
            key = f"RUL_ci{int(lv*100):02d}"
            out[key] = np.clip(RUL + z*total, 0, None).astype(np.float32)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# §13  ENGINE SIMULATOR  (fully upgraded)
# ══════════════════════════════════════════════════════════════════════════════

class EngineSimulator:
    def __init__(self, cfg: DatasetConfig, rng: np.random.Generator):
        self.cfg     = cfg
        self.rng     = rng
        self.thermo  = TransientThermo(cfg.physics)
        self.degd    = DegradationEngine(cfg.physics, cfg.degrad, rng)
        self.sensor  = SensorLayer(cfg.sensor_fail, rng)
        self.env_gen = FlightEnvironment(rng)
        self.fadec_cfg = cfg.fadec
        self.uq      = UncertaintyQ(cfg.uncertainty, rng)
        self.phase_mgr = FlightPhaseManager(cfg.pilot, rng)
        self.mission_gen = MissionContext(cfg.fleet, rng)
        self.ts_gen_proto = cfg.sampling  # prototype, instantiated per engine
        self.causal_A    = CausalGraph.adjacency()

    def _mfg(self) -> Dict:
        p = self.cfg.physics
        return {
            "eta_c": float(self.rng.lognormal(0, p.mfg_eta_c_sig)),
            "eta_t": float(self.rng.lognormal(0, p.mfg_eta_t_sig)),
            "mdot":  float(self.rng.lognormal(0, p.mfg_mdot_sig)),
            "N":     float(self.rng.lognormal(0, p.mfg_N_sig)),
        }

    def _fleet(self, eid: int) -> Dict:
        fl  = self.cfg.fleet
        aid = eid % fl.n_airlines
        rid = (eid // fl.n_airlines) % len(fl.route_types)
        return {"airline_id": aid, "route_type": fl.route_types[rid],
                "maint_agg": fl.maint_agg[aid], "utilization": fl.utilization[aid],
                # Feature 4: engine pair
                "pair_id":   eid // 2,
                "pair_pos":  eid % 2}  # 0=left, 1=right

    def simulate(self, eid: int, max_cyc: int = None,
                 partial: bool = False,
                 paired_state: Optional[Dict] = None) -> Optional[Dict]:
        try:
            return self._sim_inner(eid, max_cyc, partial, paired_state)
        except Exception as exc:
            log.debug(f"Engine {eid} raised {type(exc).__name__}: {exc}")
            return None

    def _sim_inner(self, eid: int, max_cyc: int, partial: bool,
                    paired_state: Optional[Dict]) -> Optional[Dict]:
        cfg   = self.cfg
        if max_cyc is None:
            max_cyc = int(self.rng.integers(cfg.min_cycles, cfg.max_cycles+1))

        fleet = self._fleet(eid)
        mfg   = self._mfg()
        state = self.degd.init_state(mfg)
        op    = int(self.rng.integers(0, cfg.n_op_cond))

        # Per-engine: calibration bias, drift fault, pilot style
        self.sensor.reset()
        bias        = self.sensor.bias()
        self.sensor.init_calibration_bias()
        self.sensor.init_drift_faults()
        pilot_style = self.phase_mgr.sample_pilot_style()
        fadec       = FADEC(cfg.fadec, cfg.physics, pilot_style)

        # Timestamp generator for this engine
        ts_gen = TimestampGenerator(cfg.sampling, self.rng, eid)

        mm = MaintenanceManager(cfg.maint, cfg.physics, fleet["maint_agg"], self.rng)

        # Storage
        Sn=[]; Sc=[]; HI_list=[]; DL=[]; EL=[]; MF=[]; EF=[]
        FF=[]; CS=[]; FL=[]; ML=[]; TS=[]; SI=[]
        fault_sensor_rows = []  # Feature 3: per-cycle sensor fault flags

        actual = 0
        for cyc in range(max_cyc):
            if self.degd.is_failed(state) or fadec.is_shutdown:
                break

            # Base environment
            env = self.env_gen.sample(op, cyc, fleet["route_type"], fleet["utilization"])

            # Mission context (Feature 5)
            mission = self.mission_gen.sample(fleet["route_type"])

            # Flight phase for this cycle (Feature 1)
            phase, phase_env, thermal_delta = self.phase_mgr.representative_phase(
                cyc, max_cyc, env, pilot_style)
            # Merge phase overrides into env
            env["throttle"]       = phase_env["throttle"]
            env["altitude"]       = phase_env["altitude"]
            env["flight_phase"]   = int(phase)
            env["thermal_delta"]  = thermal_delta

            thr_eff, trim, bleed, fadec_fl = fadec.update(
                env["throttle"], state.get("last_EGT", 800.0),
                cfg.physics.N_design, state["eff_c"], 0.15)

            # Compute cycle with phase EGT bias and mission density
            cs = self.thermo.compute_cycle(
                alt=env["altitude"], dT_ISA=env["dT_ISA"], throttle=thr_eff,
                D_fat=state["D_fat"], D_crp=state["D_crp"], D_cor=state.get("D_cor",0.0),
                eff_c=state["eff_c"], eff_t=state["eff_t"],
                eta_comb=state["eta_comb"], mdot_scale=state["mdot_scale"],
                T_core_prev=state["T_core"], fuel_trim=trim, bleed_active=bleed,
                phase_EGT_bias=phase_env.get("EGT_phase_bias", 0.0),
                fuel_contam_factor=0.0,
                density_ratio=mission.get("air_density_ratio", 1.0))

            # Cross-engine delta (Feature 4)
            if paired_state is not None:
                cross_EGT = cs["EGT"] - paired_state.get("last_EGT", cs["EGT"])
                cross_RPM = cs["N_rpm"] - paired_state.get("last_RPM", cs["N_rpm"])
            else:
                cross_EGT = float(self.rng.normal(0, cfg.fleet.pair_egt_tolerance*0.3))
                cross_RPM = float(self.rng.normal(0, cfg.fleet.pair_rpm_tolerance*0.3))

            gt    = self.sensor.ground_truth(cs, state, cross_EGT, cross_RPM)

            # Sensor fault flags array: [drift_active, n_drift_ch, stuck_active, stuck_ch, spike, n_missing]
            fault_flags_arr = np.zeros(6, dtype=np.float32)

            state, mods = self.degd.step(state, cs, env, cfg.events, cyc, phase, mission)
            state, mev, maint_type = mm.apply(state, cyc)
            noisy = self.sensor.add_noise(gt, bias, state["drift"], mods,
                                           fault_flags_arr, cyc)

            # Timestamp (Feature 9)
            ts, samp_int = ts_gen.next()

            # Causal state vector
            idx = CausalGraph.IDX
            cg  = np.zeros(CausalGraph.N, dtype=np.float32)
            cg[idx["T4"]]          = float(cs["T4"]/2000.0)
            cg[idx["RPM"]]         = float(cs["N_rpm"]/cfg.physics.N_design)
            cg[idx["humidity"]]    = float(env["humidity"])
            cg[idx["altitude"]]    = float(env["altitude"]/12500.0)
            cg[idx["throttle"]]    = float(thr_eff)
            cg[idx["D_fat"]]       = float(state["D_fat"])
            cg[idx["D_crp"]]       = float(state["D_crp"])
            cg[idx["D_cor"]]       = float(state.get("D_cor",0.0))
            cg[idx["D_th"]]        = float(state["D_th"])
            cg[idx["eff_c"]]       = float(state["eff_c"]/cfg.physics.eta_c0)
            cg[idx["eff_t"]]       = float(state["eff_t"]/cfg.physics.eta_t0)
            cg[idx["crack_len"]]   = float(state["crack_len"]/cfg.physics.a_crit)
            cg[idx["EGT"]]         = float(cs["EGT"]/cfg.physics.T4_max)
            cg[idx["vibration"]]   = float(cs["vibration"]/2.0)
            cg[idx["oil_temp"]]    = float(cs["oil_temp"]/600.0)
            cg[idx["flight_phase"]]= float(int(phase)/5.0)
            cg[idx["fuel_contam"]] = float(state.get("in_fuel_contam", False))
            cg[idx["cross_EGT_delta"]] = float(cross_EGT / (cfg.fleet.pair_egt_tolerance*3))

            Sn.append(noisy.astype(np.float32))
            Sc.append(gt.astype(np.float32))
            HI_list.append(self.degd.health_index(state))
            DL.append(np.array([state["D_fat"],state["D_crp"],state.get("D_cor",0.0),
                                  state["D_th"],state["eff_c"],state["eff_t"],
                                  state["crack_len"],state["disk_burst_risk"],
                                  cs["T_core"],cs["surge_margin"],
                                  cs["bleed_active"],cs["stall_active"]],
                                 dtype=np.float32))
            EL.append(np.array([env["altitude"],env["throttle"],env["humidity"],
                                  env["dT_ISA"],env["salt_factor"],thr_eff,
                                  float(bleed),float(bool(mods.get("event_name"))),
                                  float(int(phase)),
                                  mission["route_distance_nm"],
                                  mission["flight_duration_min"],
                                  mission["airport_elevation_ft"],
                                  mission["sand_exposure_index"],
                                  mission["salt_exposure_index"],
                                  float(state.get("time_since_maint", 0)),
                                  thermal_delta],
                                 dtype=np.float32))
            MF.append(1 if mev else 0)
            ML.append(maint_type)
            EF.append(1 if mods.get("event_name") else 0)
            FF.append(np.array([float(fadec_fl.get("EGT_protect",False)),
                                  float(fadec_fl.get("bleed_valve",False)),
                                  float(fadec_fl.get("overspeed",False)),
                                  float(fadec_fl.get("emergency_shutdown",False))],
                                 dtype=np.float32))
            CS.append(cg)
            fault_sensor_rows.append(fault_flags_arr)
            TS.append(float(ts))
            SI.append(float(samp_int))
            actual += 1

        if actual < 10:
            return None

        n   = actual
        RUL = np.arange(n-1, -1, -1, dtype=np.float32)
        cut = n
        if partial:
            cut = max(10, int(float(self.rng.uniform(0.3, 0.8))*n))
            RUL = RUL[:cut]

        HI_arr   = np.array(HI_list[:cut], dtype=np.float32)
        env_arr  = np.array(EL[:cut],      dtype=np.float32)
        fail_probs = np.array([
            self.degd.failure_probability(float(HI_arr[i]), cfg.uncertainty)
            for i in range(cut)], dtype=np.float32)

        for i in range(cut):
            CS[i][CausalGraph.IDX["RUL"]] = float(RUL[i]/max(1, RUL[0]))

        rul_d = self.uq.rul_dist(RUL, HI_arr, env_arr[:, 2], fail_probs)
        pinn  = self.degd.pinn_residuals(
            [dict(D_fat=DL[i][0],D_crp=DL[i][1],D_cor=DL[i][2],
                  D_th=DL[i][3],eff_c=DL[i][4],crack_len=DL[i][6])
             for i in range(min(cut+1, n))])

        return {
            "engine_id":     eid,
            "op_cond":       op,
            "n_cycles":      cut,
            "n_cycles_eol":  n,
            "is_partial":    partial and cut < n,
            "fleet":         fleet,
            "pilot_style":   pilot_style,
            "maint_log":     mm.log,
            "event_log":     state["events"],
            "sensors":       np.array(Sn[:cut]),
            "sensors_clean": np.array(Sc[:cut]),
            "degrad":        np.array(DL[:cut]),
            "env":           env_arr,
            "fadec_flags":   np.array(FF[:cut]),
            "causal_state":  np.array(CS[:cut], dtype=np.float32),
            "sensor_faults": np.array(fault_sensor_rows[:cut], dtype=np.float32),
            "maint_flag":    np.array(MF[:cut], dtype=np.int8),
            "maint_type":    ML[:cut],
            "event_flag":    np.array(EF[:cut], dtype=np.int8),
            "timestamps":    np.array(TS[:cut], dtype=np.float64),
            "sampling_intervals": np.array(SI[:cut], dtype=np.float32),
            "RUL":           RUL,
            "RUL_dist":      rul_d,
            "health_index":  HI_arr,
            "op_setting":    np.full(cut, op, dtype=np.int32),
            "pinn":          pinn,
        }


# ══════════════════════════════════════════════════════════════════════════════
# §14  DOMAIN SHIFT
# ══════════════════════════════════════════════════════════════════════════════

def make_domain_cfg(base: DatasetConfig, domain: str) -> DatasetConfig:
    cfg = deepcopy(base)
    if domain == "val":
        cfg.sensor_fail.missing_prob *= 1.5
        cfg.sensor_fail.drift_fault_prob *= 1.3
    elif domain == "test":
        cfg.sensor_fail.missing_prob  *= 3.0
        cfg.sensor_fail.stuck_prob    *= 2.0
        cfg.sensor_fail.drift_fault_prob *= 2.5
        cfg.events.bird_strike_prob   *= 2.0
        cfg.events.stall_prob         *= 1.5
        cfg.events.fuel_contam_prob   *= 2.0
        cfg.degrad.k_cor              *= 1.8
        cfg.sampling.dropout_prob     *= 2.0
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# §15  HDF5 WRITER
# ══════════════════════════════════════════════════════════════════════════════

class HDF5Writer:
    SN = SensorLayer.N   # 20
    DN = 12; EN = 16; FN = 4; CN = CausalGraph.N  # 19
    SFN = 6  # sensor fault flag channels

    def __init__(self, path: str, cfg: DatasetConfig):
        self.h5   = h5py.File(path, "w")
        self.cfg  = cfg
        self._cur: Dict[str, int] = {}

    def _init(self, split: str):
        g  = self.h5.require_group(split)
        co = dict(compression="gzip", compression_opts=4)
        ck = (8192,)

        def mk(name, sh=(), dt=np.float32):
            g.create_dataset(name, shape=(0,)+sh, maxshape=(None,)+sh,
                              dtype=dt, chunks=ck+sh, **co)

        mk("sensors",       (self.SN,)); mk("sensors_clean", (self.SN,))
        mk("degrad",        (self.DN,)); mk("env",           (self.EN,))
        mk("fadec_flags",   (self.FN,)); mk("causal_state",  (self.CN,))
        mk("sensor_faults", (self.SFN,))
        mk("RUL"); mk("RUL_std"); mk("RUL_alea"); mk("RUL_epi")
        mk("RUL_lower"); mk("RUL_upper"); mk("failure_prob")
        mk("health_index")
        mk("maint_flag",    dt=np.int8)
        mk("event_flag",    dt=np.int8)
        mk("op_setting",    dt=np.int32)
        mk("engine_id",     dt=np.int32)
        mk("cycle_in_engine", dt=np.int32)
        mk("timestamps",    dt=np.float64)
        mk("sampling_intervals")
        for k in ["D_fat","D_crp","D_cor","D_th","eff_c","crack_len"]:
            mk(f"pinn_res_{k}")
        for lv in self.cfg.uncertainty.ci_levels:
            mk(f"RUL_ci{int(lv*100):02d}")
        self._cur[split] = 0

    def _app(self, ds: h5py.Dataset, arr: np.ndarray):
        n   = len(arr); old = ds.shape[0]
        ds.resize(old+n, axis=0); ds[old:old+n] = arr

    def write(self, split: str, rec: Dict):
        if split not in self._cur:
            self._init(split)
        g  = self.h5[split]
        n  = rec["n_cycles"]
        rd = rec["RUL_dist"]
        pn = rec.get("pinn", {})

        def app(key, arr):
            if key in g:
                self._app(g[key], arr)

        app("sensors",        rec["sensors"])
        app("sensors_clean",  rec["sensors_clean"])
        app("degrad",         rec["degrad"])
        app("env",            rec["env"])
        app("fadec_flags",    rec["fadec_flags"])
        app("causal_state",   rec["causal_state"])
        app("sensor_faults",  rec["sensor_faults"])
        app("RUL",            rec["RUL"])
        app("RUL_std",        rd["RUL_std"])
        app("RUL_alea",       rd["RUL_alea"])
        app("RUL_epi",        rd["RUL_epi"])
        app("RUL_lower",      rd["RUL_lower"])
        app("RUL_upper",      rd["RUL_upper"])
        app("failure_prob",   rd["failure_prob"])
        app("health_index",   rec["health_index"])
        app("maint_flag",     rec["maint_flag"])
        app("event_flag",     rec["event_flag"])
        app("op_setting",     rec["op_setting"])
        app("engine_id",      np.full(n, rec["engine_id"], np.int32))
        app("cycle_in_engine",np.arange(n, dtype=np.int32))
        app("timestamps",     rec["timestamps"])
        app("sampling_intervals", rec["sampling_intervals"])

        for k in ["D_fat","D_crp","D_cor","D_th","eff_c","crack_len"]:
            pk  = f"pinn_res_{k}"
            arr = pn.get(f"res_{k}", np.zeros(max(0,n-1), np.float32))
            if pk in g:
                self._app(g[pk], arr)

        for lv in self.cfg.uncertainty.ci_levels:
            ck = f"RUL_ci{int(lv*100):02d}"
            app(ck, rd.get(ck, rec["RUL"]))

        self._cur[split] += n

    def write_metadata(self):
        g  = self.h5.require_group("metadata")
        st = h5py.string_dtype()
        g.create_dataset("sensor_names",   data=np.array(SensorLayer.NAMES, dtype=st))
        g.create_dataset("sensor_units",   data=np.array(SensorLayer.UNITS, dtype=st))
        g.create_dataset("causal_nodes",   data=np.array(CausalGraph.NODES, dtype=st))
        g.create_dataset("causal_adjacency", data=CausalGraph.adjacency())
        g.create_dataset("env_col_names",  data=np.array([
            "altitude","throttle","humidity","dT_ISA","salt_factor","throttle_eff",
            "bleed_flag","event_present","flight_phase","route_distance_nm",
            "flight_duration_min","airport_elevation_ft","sand_exposure_index",
            "salt_exposure_index","time_since_maint","thermal_delta"], dtype=st))
        g.create_dataset("flight_phase_names",
                          data=np.array(PHASE_NAMES, dtype=st))
        g.create_dataset("sensor_fault_col_names", data=np.array([
            "drift_fault_active","n_drift_channels","stuck_active",
            "stuck_channel","spike_present","n_missing"], dtype=st))
        g.attrs["version"]       = "UTDTB_v5.0"
        g.attrs["sensor_count"]  = SensorLayer.N
        g.attrs["env_cols"]      = 16
        g.attrs["features_v5"]   = json.dumps([
            "flight_phase_dynamics", "rich_maintenance_events",
            "advanced_sensor_faults", "multi_engine_fleet_pairs",
            "mission_level_context", "rare_catastrophic_events",
            "pilot_operational_variation", "per_engine_calibration_bias",
            "timestamp_irregular_sampling", "full_rul_uncertainty"])
        g.attrs["config"] = json.dumps({
            "n_train": self.cfg.n_train, "n_val": self.cfg.n_val,
            "n_test":  self.cfg.n_test,
            "min_cycles": self.cfg.min_cycles,
            "max_cycles": self.cfg.max_cycles,
            "seed": self.cfg.seed,
            "generator": "UTDTB_v5.0_BEAST",
        })

    def total_rows(self, split: str) -> int:
        return self._cur.get(split, 0)

    def close(self):
        self.h5.flush(); self.h5.close()


# ══════════════════════════════════════════════════════════════════════════════
# §16  TABULAR WRITER
# ══════════════════════════════════════════════════════════════════════════════

class TabularWriter:
    _DCOLS = ["D_fat","D_crp","D_cor","D_th","eff_c","eff_t",
               "crack_len","disk_burst_risk","T_core","surge_margin",
               "bleed_active","stall_active"]
    _ECOLS = ["altitude","throttle","humidity","dT_ISA","salt_factor",
               "throttle_eff","bleed_flag","event_present","flight_phase",
               "route_distance_nm","flight_duration_min","airport_elevation_ft",
               "sand_exposure_index","salt_exposure_index",
               "time_since_maint","thermal_delta"]

    def to_df(self, recs: List[Dict], split: str) -> pd.DataFrame:
        rows = []
        for rec in recs:
            if rec is None:
                continue
            n  = rec["n_cycles"]; rd = rec["RUL_dist"]
            for i in range(n):
                r: Dict[str,Any] = {
                    "engine_id":   rec["engine_id"], "cycle": i,
                    "split":       split,
                    "airline_id":  rec["fleet"]["airline_id"],
                    "route_type":  rec["fleet"]["route_type"],
                    "pair_id":     rec["fleet"]["pair_id"],
                    "pair_pos":    rec["fleet"]["pair_pos"],
                    "op_setting":  int(rec["op_setting"][i]),
                    "timestamp":   float(rec["timestamps"][i]),
                    "sampling_interval_min": float(rec["sampling_intervals"][i]),
                }
                for j, sn in enumerate(SensorLayer.NAMES):
                    r[f"s_{sn}"] = float(rec["sensors"][i, j])
                for j, dn in enumerate(self._DCOLS):
                    r[dn] = float(rec["degrad"][i, j])
                for j, en in enumerate(self._ECOLS):
                    r[en] = float(rec["env"][i, j])
                for j, fn in enumerate(["fadec_EGT","fadec_bleed",
                                         "fadec_overspeed","fadec_eshutdown"]):
                    r[fn] = float(rec["fadec_flags"][i, j])
                sfcols = ["sf_drift_active","sf_n_drift_ch","sf_stuck_active",
                          "sf_stuck_ch","sf_spike","sf_n_missing"]
                for j, sfn in enumerate(sfcols):
                    r[sfn] = float(rec["sensor_faults"][i, j])
                r["RUL"]           = float(rec["RUL"][i])
                r["RUL_std"]       = float(rd["RUL_std"][i])
                r["RUL_lower"]     = float(rd["RUL_lower"][i])
                r["RUL_upper"]     = float(rd["RUL_upper"][i])
                r["failure_prob"]  = float(rd["failure_prob"][i])
                r["RUL_alea"]      = float(rd["RUL_alea"][i])
                r["RUL_epi"]       = float(rd["RUL_epi"][i])
                r["health_index"]  = float(rec["health_index"][i])
                r["maint_flag"]    = int(rec["maint_flag"][i])
                r["maint_type"]    = str(rec["maint_type"][i])
                r["event_flag"]    = int(rec["event_flag"][i])
                for lv in [5, 10, 25, 75, 90, 95]:
                    ck = f"RUL_ci{lv:02d}"
                    r[ck] = float(rd.get(ck, rec["RUL"])[i])
                rows.append(r)
        return pd.DataFrame(rows)

    def write(self, recs_map: Dict[str, List[Dict]], out: Path,
              do_parquet: bool, do_csv: bool, do_lf: bool):
        dfs = []
        for split, recs in recs_map.items():
            if recs:
                dfs.append(self.to_df(recs, split))
        if not dfs:
            log.warning("TabularWriter: no records to write!")
            return None
        full = pd.concat(dfs, ignore_index=True)
        if do_parquet:
            pq = out/"utdtb_v5.parquet"
            full.to_parquet(str(pq), index=False, compression="snappy")
            log.info(f"  Parquet → {pq}  ({pq.stat().st_size/1e6:.1f} MB)")
        if do_csv:
            csv = out/"utdtb_v5.csv"
            full.to_csv(str(csv), index=False)
            log.info(f"  CSV → {csv}  ({csv.stat().st_size/1e6:.1f} MB)")
        if do_lf:
            sc   = [f"s_{n}" for n in SensorLayer.NAMES[:14]]
            keep = [c for c in
                    ["engine_id","cycle","op_setting","altitude","throttle",
                     "humidity","flight_phase","RUL","RUL_lower","RUL_upper",
                     "failure_prob","health_index","split","timestamp"]+sc
                    if c in full.columns]
            (out/"utdtb_v5_lowfidelity.csv").write_text(
                full[keep].to_csv(index=False))
        return full


# ══════════════════════════════════════════════════════════════════════════════
# §17  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize(tr, va, te, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.gridspec as gs

    out = Path(out_dir)
    fig = plt.figure(figsize=(28, 22))
    fig.suptitle("UTDTB v5.0 — Universal Turbofan Digital Twin Benchmark (BEAST)",
                 fontsize=14, fontweight="bold")
    G   = gs.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.42)

    panels = [
        ("EGT [K]",           G[0,0]), ("P30 [bar]",          G[0,1]),
        ("Core RPM",          G[0,2]), ("Health Index",        G[0,3]),
        ("Fatigue D_fat",     G[1,0]), ("Creep D_crp",         G[1,1]),
        ("Thermal D_th",      G[1,2]), ("Crack Length [mm]",   G[1,3]),
        ("RUL [cycles]",      G[2,0]), ("Comp. Efficiency",    G[2,1]),
        ("Disk Burst Risk",   G[2,2]), ("RUL ± σ (Engine 0)", G[2,3]),
        ("Flight Phase Dist", G[3,0]), ("RUL Distribution",    G[3,1:3]),
        ("Failure Prob",      G[3,3]),
        ("Event Counts",      G[4,0]), ("Sensor Fault Flags",  G[4,1]),
        ("Cross-Engine EGT",  G[4,2]), ("Surge Margin",        G[4,3]),
    ]
    ax = {lbl: fig.add_subplot(sp) for lbl, sp in panels}
    sample = [r for r in tr[:6] if r is not None]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sample),1)))

    for i, rec in enumerate(sample):
        c_=colors[i]; n_=rec["n_cycles"]; cyc=np.arange(n_)
        s_=rec["sensors"]; d_=rec["degrad"]
        ax["EGT [K]"].plot(cyc, s_[:,16], alpha=0.7, color=c_, lw=1.1, label=f"E{rec['engine_id']}")
        ax["P30 [bar]"].plot(cyc, s_[:,6]/1e5, alpha=0.7, color=c_, lw=1.1)
        ax["Core RPM"].plot(cyc, s_[:,8], alpha=0.7, color=c_, lw=1.1)
        ax["Health Index"].plot(cyc, rec["health_index"], color=c_, lw=1.5, alpha=0.85)
        ax["Fatigue D_fat"].plot(cyc, d_[:,0], alpha=0.7, color=c_, lw=1.1)
        ax["Creep D_crp"].plot(cyc, d_[:,1], alpha=0.7, color=c_, lw=1.1)
        ax["Thermal D_th"].plot(cyc, d_[:,3], alpha=0.7, color=c_, lw=1.1)
        ax["Crack Length [mm]"].plot(cyc, d_[:,6]*1000, alpha=0.7, color=c_, lw=1.1)
        ax["RUL [cycles]"].plot(cyc, rec["RUL"], color=c_, lw=1.5, alpha=0.85)
        ax["Comp. Efficiency"].plot(cyc, d_[:,4], alpha=0.7, color=c_, lw=1.1)
        ax["Disk Burst Risk"].plot(cyc, d_[:,7], alpha=0.7, color=c_, lw=1.1)
        ax["Surge Margin"].plot(cyc, d_[:,9], alpha=0.6, color=c_, lw=1.1)
        ax["Failure Prob"].plot(cyc, rec["RUL_dist"]["failure_prob"], alpha=0.7, color=c_, lw=1.1)
        if rec["sensors"].shape[1] > 18:
            ax["Cross-Engine EGT"].plot(cyc, rec["sensors"][:,18], alpha=0.6, color=c_, lw=1)

    if sample:
        r0=sample[0]; c0=np.arange(r0["n_cycles"]); rd=r0["RUL_dist"]
        mu=rd["RUL_mean"]; sig=rd["RUL_std"]
        lo=rd["RUL_lower"]; hi=rd["RUL_upper"]
        ax["RUL ± σ (Engine 0)"].plot(c0, mu, color="steelblue", lw=1.5, label="Mean")
        ax["RUL ± σ (Engine 0)"].fill_between(c0, mu-sig, mu+sig, alpha=0.3,
                                                color="steelblue", label="±1σ")
        ax["RUL ± σ (Engine 0)"].fill_between(c0, lo, hi, alpha=0.15,
                                                color="steelblue", label="95% CI")
        ax["RUL ± σ (Engine 0)"].legend(fontsize=7)

        # Flight phase distribution
        ph_cols = r0["env"][:,8] if r0["env"].shape[1] > 8 else np.zeros(r0["n_cycles"])
        ph_counts = [np.sum(ph_cols==p) for p in range(6)]
        ax["Flight Phase Dist"].bar(PHASE_NAMES, ph_counts, color="steelblue", alpha=0.8)
        ax["Flight Phase Dist"].set_xticklabels(PHASE_NAMES, rotation=30, ha="right", fontsize=7)

    for spl_, recs_, col_ in [("train",tr,"steelblue"),("val",va,"orange"),("test",te,"firebrick")]:
        v_ = np.concatenate([r["RUL"] for r in recs_ if r is not None])
        ax["RUL Distribution"].hist(v_, bins=80, alpha=0.5, label=spl_, color=col_, density=True)
    ax["RUL Distribution"].legend()

    # Event counts
    ec = {}
    for recs_ in [tr,va,te]:
        for r in recs_:
            if r is None: continue
            for ev in r.get("event_log", []):
                ec[ev["t"]] = ec.get(ev["t"],0)+1
    if ec:
        nms=list(ec.keys()); vs=list(ec.values())
        ax["Event Counts"].bar(range(len(nms)), vs, color="crimson", alpha=0.8)
        ax["Event Counts"].set_xticks(range(len(nms)))
        ax["Event Counts"].set_xticklabels(nms, rotation=35, ha="right", fontsize=7)

    # Sensor fault flag presence
    sf_names = ["drift","n_drift","stuck","stuck_ch","spike","missing"]
    sf_totals = np.zeros(6)
    for recs_ in [tr,va,te]:
        for r in recs_:
            if r is not None and "sensor_faults" in r:
                sf_totals += r["sensor_faults"].sum(axis=0)
    ax["Sensor Fault Flags"].bar(sf_names, sf_totals, color="darkorange", alpha=0.8)
    ax["Sensor Fault Flags"].set_xticklabels(sf_names, rotation=30, ha="right", fontsize=8)

    ax["EGT [K]"].legend(fontsize=7)
    ax["Health Index"].axhline(0.02, color="red", ls="--", alpha=0.7)
    ax["Health Index"].set_ylim([0,1.05])
    for lbl, _ in panels:
        ax[lbl].set_title(lbl, fontsize=9, fontweight="bold")

    plt.savefig(str(out/"utdtb_v5_lifecycle.png"), dpi=110, bbox_inches="tight")
    plt.close()
    log.info("  Saved utdtb_v5_lifecycle.png")


# ══════════════════════════════════════════════════════════════════════════════
# §18  PYTORCH HOOK
# ══════════════════════════════════════════════════════════════════════════════

_TORCH_HOOK = '''\
"""UTDTB v5.0 — PyTorch Dataset Hook (all new features)"""
import torch, h5py, numpy as np
from torch.utils.data import Dataset

class UTDTBv5Dataset(Dataset):
    """
    Sliding-window Dataset over UTDTB v5.0 HDF5.
    Each item returns:
      sensors     (W, 20)  — noisy + drift-fault + cross-engine channels
      degrad      (12,)    — latent degradation state at window end
      env         (W, 16)  — environment + flight phase + mission context
      causal      (19,)    — causal state vector
      fadec       (W, 4)   — FADEC protection flags
      sensor_faults (W,6)  — fault flag channels
      RUL         scalar   — mean RUL target
      RUL_lower   scalar   — 95% lower bound (Feature 10)
      RUL_upper   scalar   — 95% upper bound
      failure_prob scalar  — instantaneous failure probability
      HI          scalar   — health index
    """
    def __init__(self, h5_path, split="train", window=30, stride=5, normalize=True):
        with h5py.File(h5_path, "r") as f:
            g = f[split]
            self.sensors  = torch.tensor(np.nan_to_num(g["sensors"][:]),  torch.float32)
            self.degrad   = torch.tensor(g["degrad"][:],                   torch.float32)
            self.env      = torch.tensor(g["env"][:],                      torch.float32)
            self.causal   = torch.tensor(g["causal_state"][:],             torch.float32)
            self.fadec    = torch.tensor(g["fadec_flags"][:],              torch.float32)
            self.sfaults  = torch.tensor(g["sensor_faults"][:],            torch.float32)
            self.RUL      = torch.tensor(g["RUL"][:],                      torch.float32)
            self.RUL_lo   = torch.tensor(g["RUL_lower"][:],                torch.float32)
            self.RUL_hi   = torch.tensor(g["RUL_upper"][:],                torch.float32)
            self.fail_p   = torch.tensor(g["failure_prob"][:],             torch.float32)
            self.HI       = torch.tensor(g["health_index"][:],             torch.float32)
            self.eids     = g["engine_id"][:]
            self.causal_A = torch.tensor(f["metadata/causal_adjacency"][:],torch.float32)
        if normalize:
            mu = self.sensors.nanmean(0); sd = self.sensors.std(0).clamp(1e-6)
            self.sensors = (self.sensors - mu) / sd
            mx = self.RUL.max().clamp(1)
            self.RUL /= mx; self.RUL_lo /= mx; self.RUL_hi /= mx
        self.windows = []
        for eid in np.unique(self.eids):
            idx = np.where(self.eids == eid)[0]
            for s in range(0, len(idx)-window, stride):
                self.windows.append((idx[s], idx[s+window-1]))

    def __len__(self): return len(self.windows)

    def __getitem__(self, i):
        s, e = self.windows[i]
        return {
            "sensors":      self.sensors[s:e+1],
            "degrad":       self.degrad[e],
            "env":          self.env[s:e+1],
            "causal":       self.causal[e],
            "causal_A":     self.causal_A,
            "fadec":        self.fadec[s:e+1],
            "sensor_faults":self.sfaults[s:e+1],
            "RUL":          self.RUL[e],
            "RUL_lower":    self.RUL_lo[e],
            "RUL_upper":    self.RUL_hi[e],
            "failure_prob": self.fail_p[e],
            "HI":           self.HI[e],
        }
'''

_README = """\
# UTDTB v5.0 — Universal Turbofan Digital Twin Benchmark (BEAST MODE)

## 10 New Features over v4.1
1. **Flight-phase dynamics** — taxi/takeoff/climb/cruise/descent/landing physics per cycle
2. **Rich maintenance** — blade repair, module replacement, with `maint_type` column
3. **Advanced sensor faults** — gradual drift fault, stuck, spike, dropout + `sensor_faults` matrix
4. **Multi-engine fleet pairs** — `pair_id`, `cross_delta_EGT`, `cross_delta_RPM` channels
5. **Mission context** — `route_distance_nm`, `flight_duration_min`, `airport_elevation_ft`, `sand_exposure_index`
6. **Rare events** — fuel contamination, FOD damage added to bird/stall/oil/seal/ash/sand/ice
7. **Pilot variation** — throttle ramp rate, climb profile, surge margin preference per engine
8. **Per-engine calibration bias** — unique `b_i` offset per sensor channel per engine
9. **Timestamps + irregular sampling** — `timestamp` (unix), `sampling_interval_min` (ACARS-style)
10. **Full RUL uncertainty** — `RUL_lower`, `RUL_upper` (95% CI), `failure_prob` (logistic)

## Quick load
```python
import h5py, numpy as np
with h5py.File('utdtb_v5.h5', 'r') as f:
    X     = f['train/sensors'][:]        # (N, 20) noisy sensors
    y     = f['train/RUL'][:]            # (N,) mean RUL
    y_lo  = f['train/RUL_lower'][:]      # 95% lower bound
    y_hi  = f['train/RUL_upper'][:]      # 95% upper bound
    pf    = f['train/failure_prob'][:]   # failure probability
    phase = f['train/env'][:, 8]         # flight phase (0-5)
    A     = f['metadata/causal_adjacency'][:]  # (19,19) causal graph
```

## Dataset columns (~85 total)
### Sensors (20 channels)
T2, T24, T30, T50, P2, P15, P30, Nf, Nc, EPR, Ps30, phi, NRc, BPR,
vib_rms, oil_temp, EGT_direct, surge_margin,
**cross_delta_EGT**, **cross_delta_RPM**  ← NEW cross-engine

### Degradation (12)
D_fat, D_crp, D_cor, D_th, eff_c, eff_t, crack_len, disk_burst_risk,
T_core, surge_margin, bleed_active, stall_active

### Environment (16)
altitude, throttle, humidity, dT_ISA, salt_factor, throttle_eff,
bleed_flag, event_present,
**flight_phase**, **route_distance_nm**, **flight_duration_min**,
**airport_elevation_ft**, **sand_exposure_index**, **salt_exposure_index**,
**time_since_maint**, **thermal_delta**

### FADEC flags (4)  |  Sensor fault matrix (6)  |  RUL labels (10)
RUL_mean, RUL_std, RUL_alea, RUL_epi,
**RUL_lower**, **RUL_upper**, **failure_prob**,
RUL_ci05..ci95

### Metadata
engine_id, cycle, split, airline_id, route_type, **pair_id**, **pair_pos**,
op_setting, **timestamp**, **sampling_interval_min**,
maint_flag, **maint_type**, event_flag, PINN residuals × 6

## Scale (BEAST mode)
| Split | Engines | ~Rows |
|-------|---------|-------|
| Train | 16,000  | ~11M  |
| Val   |  2,000  |  ~1.4M|
| Test  |  2,000  |  ~1.4M|
| Total | 20,000  | ~14M  |
"""


# ══════════════════════════════════════════════════════════════════════════════
# §19  MODE → CONFIG
# ══════════════════════════════════════════════════════════════════════════════

_SCALE = {
    "QUICK":  dict(n_train=400,   n_val=50,   n_test=50,
                   min_cycles=100, max_cycles=800,
                   save_csv=True, save_lf_csv=True),
    "MEDIUM": dict(n_train=2000,  n_val=300,  n_test=300,
                   min_cycles=120, max_cycles=1000,
                   save_csv=True, save_lf_csv=True),
    "FULL":   dict(n_train=8000,  n_val=1000, n_test=1000,
                   min_cycles=150, max_cycles=1200,
                   save_csv=False, save_lf_csv=False),
    "BEAST":  dict(n_train=16000, n_val=2000, n_test=2000,
                   min_cycles=200, max_cycles=1200,
                   save_csv=False, save_lf_csv=False),
}

def build_config(mode: str = "BEAST") -> DatasetConfig:
    mode = mode.upper()
    if mode not in _SCALE:
        raise ValueError(f"Unknown mode '{mode}'. Choose: {list(_SCALE)}")
    s = _SCALE[mode]
    return DatasetConfig(
        n_train      = s["n_train"],
        n_val        = s["n_val"],
        n_test       = s["n_test"],
        min_cycles   = s["min_cycles"],
        max_cycles   = s["max_cycles"],
        partial_frac = 0.20,
        output_dir   = "/content/",
        save_hdf5    = True,
        save_parquet = True,
        save_csv     = s["save_csv"],
        save_lf_csv  = s["save_lf_csv"],
        save_zip     = True,
        seed         = 42,
        verbose      = True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §20  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class UTDTBGenerator:
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.rng = default_rng(cfg.seed)
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _child_sim(self, dcfg: DatasetConfig) -> EngineSimulator:
        return EngineSimulator(dcfg, default_rng(int(self.rng.integers(0, 2**32))))

    def _run_split(self, label: str, n: int, dcfg: DatasetConfig,
                   id_off: int, pf: float, h5w: HDF5Writer) -> List[Dict]:
        sim  = self._child_sim(dcfg)
        recs = []; t0 = time.time(); skipped = 0
        log.info(f"\n  [{label.upper()}]  Simulating {n:,} engines...")
        for i in range(n):
            eid  = id_off + i
            part = sim.rng.random() < pf
            try:
                rec = sim.simulate(eid, partial=part)
            except Exception as e:
                log.debug(f"  [{label}] engine {eid}: {e}")
                rec = None

            if rec is not None:
                recs.append(rec)
                if self.cfg.save_hdf5:
                    try:
                        h5w.write(label, rec)
                    except Exception as e:
                        log.warning(f"  [{label}] HDF5 write {eid}: {e}")
            else:
                skipped += 1

            if self.cfg.verbose and (i+1) % max(1, n//20) == 0:
                el   = time.time()-t0; rate = (i+1)/el; eta = (n-i-1)/rate
                rows = h5w.total_rows(label)
                log.info(f"    {(i+1)/n*100:5.1f}%  {i+1:>7}/{n}  "
                         f"{rate:.1f} eng/s  ETA {eta/60:.1f}min  "
                         f"rows={rows:,}  skip={skipped}")

        total = h5w.total_rows(label)
        log.info(f"  [{label.upper()}] ✓  {len(recs):,} engines  "
                 f"{total:,} rows  ({skipped} skipped)")
        return recs

    def generate(self):
        cfg = self.cfg
        t0  = time.time()
        approx = (cfg.n_train+cfg.n_val+cfg.n_test)*(cfg.min_cycles+cfg.max_cycles)//2

        log.info("╔" + "═"*70 + "╗")
        log.info("║  UTDTB v5.0 — UNIVERSAL TURBOFAN DIGITAL TWIN BENCHMARK (BEAST)   ║")
        log.info("╠" + "═"*70 + "╣")
        log.info(f"║  MODE     : {MODE:<59} ║")
        log.info(f"║  Engines  : {cfg.n_train:,} train + {cfg.n_val:,} val + {cfg.n_test:,} test{'':<19} ║")
        log.info(f"║  ~Rows    : {approx:>12,}{'':<47} ║")
        log.info(f"║  Sensors  : {SensorLayer.N} channels (incl. cross-engine){'':<34} ║")
        log.info(f"║  Features : 10 new v5.0 upgrades active{'':<31} ║")
        log.info("╚" + "═"*70 + "╝\n")

        h5_path = self.out/"utdtb_v5.h5"
        h5w     = HDF5Writer(str(h5_path), cfg)

        tr = self._run_split("train", cfg.n_train,
                              make_domain_cfg(cfg,"train"), 0, 0.15, h5w)
        vr = self._run_split("val",   cfg.n_val,
                              make_domain_cfg(cfg,"val"),   cfg.n_train, 0.20, h5w)
        er = self._run_split("test",  cfg.n_test,
                              make_domain_cfg(cfg,"test"),
                              cfg.n_train+cfg.n_val, 0.30, h5w)

        h5w.write_metadata(); h5w.close()
        log.info(f"\n  HDF5  → {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")

        log.info("\n  Writing tabular outputs...")
        tw = TabularWriter()
        tw.write({"train":tr,"val":vr,"test":er}, self.out,
                 do_parquet=cfg.save_parquet,
                 do_csv=cfg.save_csv, do_lf=cfg.save_lf_csv)

        # Causal graph
        cg_p = self.out/"causal_graph_v5.json"
        with open(str(cg_p),"w") as f:
            json.dump(CausalGraph.to_dict(), f, indent=2)
        pd.DataFrame(CausalGraph.adjacency(),
                     index=CausalGraph.NODES,
                     columns=CausalGraph.NODES).to_csv(
                         str(self.out/"causal_adjacency_v5.csv"))
        log.info(f"  Causal graph → {cg_p}")

        log.info("\n  Generating plots...")
        try:
            visualize(tr, vr, er, str(self.out))
        except Exception as e:
            log.warning(f"  Plot skipped: {e}")

        (self.out/"utdtb_v5_torch.py").write_text(_TORCH_HOOK)
        (self.out/"README_utdtb_v5.md").write_text(_README)

        if cfg.save_zip:
            self._zip()

        self._summary(tr, vr, er, time.time()-t0)
        return tr, vr, er

    def _zip(self):
        zp    = self.out/"utdtb_v5_complete.zip"
        files = []
        for pat in ["*.h5","*.parquet","*.csv","*.json","*.png","*.py","*.md"]:
            files.extend(self.out.glob(pat))
        with zipfile.ZipFile(str(zp),"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in files:
                if f != zp:
                    z.write(str(f), arcname=f.name)
        log.info(f"  ZIP   → {zp}  ({zp.stat().st_size/1e6:.1f} MB)")

    def _summary(self, tr, vr, er, elapsed):
        log.info("\n" + "═"*65)
        log.info("  UTDTB v5.0 — BEAST MODE COMPLETE")
        log.info("═"*65)
        total = 0
        for name, recs in [("TRAIN",tr),("VAL",vr),("TEST",er)]:
            if not recs: continue
            cycs = [r["n_cycles"] for r in recs]
            evts = sum(len(r.get("event_log",[])) for r in recs)
            mnt  = sum(len(r.get("maint_log",[])) for r in recs)
            faults = sum(r["sensor_faults"].sum() for r in recs)
            total += sum(cycs)
            log.info(f"  {name:5}: {len(recs):>7,} engines  {sum(cycs):>10,} rows  "
                     f"events={evts:,}  maint={mnt:,}  sensor_faults={faults:.0f}")
        log.info(f"  TOTAL : {total:>10,} rows")
        log.info(f"  Time  : {elapsed/60:.1f} min  ({elapsed/3600:.2f} hr)")
        log.info("═"*65)
        log.info("\n  Files:")
        for f in sorted(Path(self.cfg.output_dir).glob("utdtb_v5*")):
            log.info(f"    {f.name:<50} {f.stat().st_size/1e6:8.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# §21  DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def download_all(output_dir: str = "/content/"):
    out = Path(output_dir)
    priority = [
        "utdtb_v5_complete.zip",
        "utdtb_v5.h5",
        "utdtb_v5.parquet",
        "utdtb_v5_lifecycle.png",
        "causal_graph_v5.json",
        "utdtb_v5_torch.py",
        "README_utdtb_v5.md",
    ]
    files = [out/f for f in priority if (out/f).exists()]
    for f in sorted(out.glob("utdtb_v5*")):
        if f not in files:
            files.append(f)

    try:
        from google.colab import files as cf
        print("\n" + "="*58)
        print("  Downloading UTDTB v5.0 BEAST dataset...")
        print("="*58)
        for path in files:
            sz = path.stat().st_size/1e6
            print(f"  ↓  {path.name}  ({sz:.1f} MB)")
            cf.download(str(path))
        print("\n  ✅ All downloads triggered.")
        print("  Re-download single file:")
        print(f"    from google.colab import files")
        print(f"    files.download('{out}/utdtb_v5_complete.zip')")
    except ImportError:
        print("\n  Local mode — files ready at:")
        for path in files:
            print(f"  ✅  {path.name:<50}  {path.stat().st_size/1e6:7.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# §22  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    mode = globals().get("MODE", "BEAST").upper()
    cfg  = build_config(mode)
    gen  = UTDTBGenerator(cfg)
    tr, vr, er = gen.generate()
    download_all(cfg.output_dir)
    return gen, tr, vr, er


if __name__ == "__main__":
    gen, tr, vr, er = main()
else:
    _auto_run = False
    try:
        import google.colab  # noqa
        _auto_run = True
    except ImportError:
        pass
    if _auto_run:
        gen, tr, vr, er = main()
