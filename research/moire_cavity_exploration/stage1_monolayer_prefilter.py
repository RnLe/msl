#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 (Monolayer) — Grid search + robust extremum tests + ranking
Refactor v3: UI polish, robust interrupts, multi-band, global optimization
========================================================================

Changes in this version
-----------------------
1) Single, tall 3D surface panel with **flatter view**; z-limits are **clipped to sampled bands**
   (not to band-diagram range). Candidate band is a surface; ±1 and ±2 neighbors are wireframes.
2) Right-side **report card column** (prevents clipping on the right).
3) **Ctrl+C** interrupts cleanly during scanning, optimization, and plotting.
4) **SciPy optimization** across all (k_label, band) pairs runs when `--optimize` is given;
   the best optimized candidate is appended; its stats are printed once via rich/console.
5) Plotting and CSV are executed **after** the candidate list is finalized.
6) **Parallelization** support for grid search and optimization using multiprocessing.

This stage is *monolayer-only* and twist-independent.
"""

from __future__ import annotations

import os, sys, math, argparse, contextlib, logging, itertools, signal
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, cast
from functools import partial
import multiprocessing as mp_proc
from multiprocessing import Pool, Manager

import numpy as np
import matplotlib
# Predeclare rich symbols to satisfy static analyzers
Console = cast(Any, None)
Progress = cast(Any, None)
BarColumn = cast(Any, None)
TextColumn = cast(Any, None)
TimeElapsedColumn = cast(Any, None)
TimeRemainingColumn = cast(Any, None)
RichHandler = cast(Any, None)
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------- Rich logging / progress ----------------------------------------------

OUTDIR = "stage1_mono_outputs"
os.makedirs(OUTDIR, exist_ok=True)
LOG_FILE = os.path.join(OUTDIR, "run.log")

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.logging import RichHandler
    RICH = True
except Exception:
    RICH = False
    # Fallback no-op classes to satisfy static analyzers and allow runtime without Rich
    class Progress:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def add_task(self, *args, **kwargs): return 0
        def start(self): pass
        def advance(self, *args, **kwargs): pass
        def stop(self): pass
    class BarColumn:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class TextColumn:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class TimeElapsedColumn:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class TimeRemainingColumn:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class Console:  # type: ignore
        def print(self, *args, **kwargs): pass
    class RichHandler(logging.StreamHandler):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__(stream=sys.stdout)

handlers: List[logging.Handler] = []
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
handlers.append(fh)
if RICH:
    ch = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)
else:
    ch = logging.StreamHandler(stream=sys.stdout); ch.setLevel(logging.INFO)
handlers.append(ch)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
logging.raiseExceptions = False  # avoid crashing when log sinks disappear mid-run
log = logging.getLogger("stage1_mono")
console = Console() if RICH else None

# ------------------------- Optional imports ------------------------------------------------------

mp = None
mpb = None
try:
    import meep as mp  # type: ignore
    from meep import mpb  # type: ignore
except Exception as e:
    log.error("Meep/MPB not importable. Install meep-python with MPB support. %s", e)
    sys.exit(2)

SCIPY_AVAILABLE = False
try:
    from scipy.optimize import minimize, Bounds
    SCIPY_AVAILABLE = True
except Exception as e:
    log.warning("SciPy not importable; continuous optimization will be skipped. %s", e)

# ------------------------- MPB quiet context -----------------------------------------------------

def mp_quiet(enabled: bool = True):
    class _Mute(contextlib.ContextDecorator):
        def __enter__(self):
            if not enabled:
                self._null = None; self._stdout = None; self._stderr = None; return self
            self._null = open(os.devnull, "w")
            self._stdout = contextlib.redirect_stdout(self._null)
            self._stderr = contextlib.redirect_stderr(self._null)
            self._stdout.__enter__(); self._stderr.__enter__()
            return self
        def __exit__(self, exc_type, exc, tb):
            if enabled:
                self._stderr.__exit__(exc_type, exc, tb)
                self._stdout.__exit__(exc_type, exc, tb)
                self._null.close()
            return False
    return _Mute()

# ------------------------- Config structure ------------------------------------------------------

@dataclass
class Config:
    lattice: str = "square"                # "square", "triangular", "rectangular", "oblique"
    a1: Tuple[float, float] = (1.0, 0.0)   # custom basis if needed
    a2: Tuple[float, float] = (0.0, 1.0)

    high_symmetry_only: bool = True
    polarization: str = "TM"               # "TE" or "TM"
    band_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    num_bands_scan: int = 7                # MPB bands for local scans
    num_bands_report: int = 10             # bands for full band diagram

    r_min: float = 0.1                     # hole radius / a
    r_max: float = 0.48
    Nr: int = 8
    eps_min: float = 2
    eps_max: float = 10.0
    Neps: int = 8

    # Neighborhood sampling in k-space (fractional coordinates)
    step_fraction: float = 0.14            # k-step for quadratic resolution
    second_step_multiplier: float = 2.0
    num_steps_dir: int = 3                 # keep points count fixed
    min_R2: float = 0.95                   # tighten quadratic fit quality
    grad_tol: float = 3e-3                 # penalty for linear slope
    min_gap_tol: float = 5e-3
    min_abs_curv: float = 2e-3             # demand curvature
    favor_min_or_max: Optional[str] = None # "min", "max", or None

    # Band diagram path density
    kpath_pts: int = 48

    # Surface neighborhood for the 3D plot (square grid around k0)
    surf_half_width: float = 0.12          # fractional coords
    surf_N: int = 17                       

    # Scoring weights
    w_gap_center: float = 1.5
    w_gap_neigh: float = 1.5
    w_R2: float = 0.6
    w_grad: float = 1.2                    # penalty for linearity
    w_eigs_same: float = 0.5
    w_curv_mag: float = 1.1                # reward curvature
    w_quadness: float = 1.6                # strengthen quadratic-ness term
    quad_target_ratio: float = 0.9         # require stronger curvature vs gradient

    # Selection & reports
    top_K: int = 32
    plot_reports: bool = True
    max_reports: Optional[int] = None

    # MPB
    resolution_grid: int = 36
    resolution_report: int = 48
    suppress_mpb_output: bool = True

# ------------------------- Lattice helpers & k-sets ----------------------------------------------

def canonical_lattice_vectors(lattice: str) -> Tuple[np.ndarray, np.ndarray]:
    L = lattice.lower()
    if L == "square":
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if L in ("triangular","hexagonal"):
        return np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3)/2.0])
    if L == "rectangular":
        return np.array([1.0, 0.0]), np.array([0.0, 1.4])
    if L == "oblique":
        return np.array([1.0, 0.0]), np.array([0.7, 0.9])
    raise ValueError(f"Unsupported lattice: {lattice}")

def lattice_to_mpb(a1: np.ndarray, a2: np.ndarray):
    return mp.Lattice(size=mp.Vector3(1,1,0),
                      basis1=mp.Vector3(float(a1[0]), float(a1[1]), 0),
                      basis2=mp.Vector3(float(a2[0]), float(a2[1]), 0))

def hs_points(lattice: str) -> Dict[str, np.ndarray]:
    L = lattice.lower()
    if L == "square":
        return {"Γ": np.array([0.0, 0.0]), "X": np.array([0.5, 0.0]), "M": np.array([0.5, 0.5])}
    if L in ("triangular","hexagonal"):
        return {"Γ": np.array([0.0, 0.0]), "M": np.array([0.5, 0.0]), "K": np.array([1/3, 1/3])}
    if L == "rectangular":
        return {"Γ": np.array([0.0,0.0]), "X": np.array([0.5,0.0]), "Y": np.array([0.0,0.5]), "M": np.array([0.5,0.5])}
    if L == "oblique":
        return {"Γ": np.array([0.0,0.0]), "B1": np.array([0.35,0.0]), "B2": np.array([0.0,0.35])}
    raise ValueError(f"Unsupported lattice: {lattice}")

def default_k_path(lattice: str) -> List[str]:
    L = lattice.lower()
    if L == "square":       return ["Γ","X","M","Γ"]
    if L in ("triangular","hexagonal"): return ["Γ","M","K","Γ"]
    if L == "rectangular":  return ["Γ","X","M","Y","Γ"]
    if L == "oblique":      return ["Γ","B1","B2","Γ"]
    raise ValueError(f"Unsupported lattice: {lattice}")

def neighbors_from_hs(lattice: str, k_label: str) -> List[Tuple[np.ndarray, str]]:
    P = hs_points(lattice)
    label = "Γ" if k_label in ("G","Gamma","gamma") else k_label
    if label not in P: return []
    origin = P[label]
    out: List[Tuple[np.ndarray, str]] = []
    for name, vec in P.items():
        if name == label: continue
        delta = vec - origin
        norm = np.linalg.norm(delta)
        if norm == 0: continue
        out.append((delta/norm, name))
    if len(out) <= 2: return out
    # pick two most orthogonal directions (simple heuristic)
    best = sorted(out, key=lambda t: -abs(np.linalg.det(np.c_[t[0], out[0][0]])))
    selected = [out[0]]
    for cand in best:
        if len(selected) == 2: break
        if np.linalg.norm(cand[0] - selected[0][0]) > 1e-6 and np.linalg.norm(cand[0] + selected[0][0]) > 1e-6:
            selected.append(cand)
    return selected

def build_kpath(P: Dict[str, np.ndarray], labels: List[str], pts_per_segment: int) -> Tuple[List[mp.Vector3], Dict[str,int]]:
    nodes = []
    for lbl in labels:
        if lbl not in P: raise ValueError(f"k-path label {lbl} not in HS set {list(P.keys())}")
        nodes.append(mp.Vector3(float(P[lbl][0]), float(P[lbl][1]), 0.0))
    kpts: List[mp.Vector3] = []
    idx_map: Dict[str,int] = {}
    idx = 0
    for i in range(len(nodes)-1):
        seg = mp.interpolate(pts_per_segment, [nodes[i], nodes[i+1]])
        if i > 0: seg = seg[1:]
        if i == 0: idx_map[labels[i]] = 0
        kpts.extend(seg)
        idx += len(seg)
        idx_map[labels[i+1]] = idx-1
    return kpts, idx_map

# ------------------------- Geometry --------------------------------------------------------------

def monolayer_geometry(a1: np.ndarray, a2: np.ndarray, radius: float, eps_bg: float):
    lat = lattice_to_mpb(a1, a2)
    geom = [mp.Cylinder(radius=float(radius), material=mp.air, center=mp.Vector3(0,0,0))]
    mat = mp.Medium(epsilon=float(eps_bg))
    return lat, geom, mat

# ------------------------- MPB computations ------------------------------------------------------

def run_mpb_single_k(a1: np.ndarray, a2: np.ndarray, r: float, eps: float, kvec: mp.Vector3,
                     pol: str, num_bands: int, resolution: int, suppress: bool) -> np.ndarray:
    lat, geom, mat = monolayer_geometry(a1, a2, r, eps)
    with mp_quiet(suppress):
        ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mat,
                            k_points=[kvec], resolution=int(resolution),
                            num_bands=int(num_bands), dimensions=2)
        p = pol.strip().lower()
        if p == "tm": ms.run_tm()
        elif p == "te": ms.run_te()
        else: raise ValueError("polarization must be TE or TM")
        return np.array(ms.all_freqs[0], float)

def band_gaps_at_k(a1: np.ndarray, a2: np.ndarray, r: float, eps: float, kvec: mp.Vector3,
                   pol: str, num_bands: int, resolution: int, suppress: bool) -> Tuple[np.ndarray, np.ndarray]:
    freqs = run_mpb_single_k(a1, a2, r, eps, kvec, pol, num_bands, resolution, suppress)
    Nb = len(freqs)
    gaps = np.full((Nb, 2), np.nan, float)
    for b in range(Nb):
        if b > 0: gaps[b,0] = freqs[b] - freqs[b-1]
        if b+1 < Nb: gaps[b,1] = freqs[b+1] - freqs[b]
    return freqs, gaps

# ------------------------- Neighborhood sampling & fits -----------------------------------------

def fit_quadratic_1d(s: np.ndarray, f: np.ndarray) -> Tuple[float,float,float,float]:
    X = np.column_stack([s**2, s, np.ones_like(s)])
    coef, *_ = np.linalg.lstsq(X, f, rcond=None)
    a,b,c = coef
    fhat = X @ coef
    ss_res = float(np.sum((f - fhat)**2))
    ss_tot = float(np.sum((f - np.mean(f))**2)) + 1e-16
    R2 = 1.0 - ss_res/ss_tot
    return float(a), float(b), float(c), float(R2)

def neighborhood_metrics(cfg: Config, a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int,
                         k0_label: str, r: float, eps: float, P: Dict[str,np.ndarray]) -> Dict[str,float]:
    dirs = neighbors_from_hs(cfg.lattice, k0_label)
    if len(dirs) == 0:
        dirs = [(np.array([1.0,0.0]), "ex"), (np.array([0.0,1.0]), "ey")]

    def seg_len(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(b - a))

    step = cfg.step_fraction
    step2 = cfg.second_step_multiplier * step
    k0 = P["Γ"] if k0_label in ("G","Gamma","gamma") else P[k0_label]
    k0_vec = mp.Vector3(float(k0[0]), float(k0[1]), 0.0)

    grad_max_abs = 0.0
    R2_min = 1.0
    kappas: Dict[str, float] = {}
    grads: Dict[str, float] = {}
    min_gap_neigh = float("inf")
    quad_ratios: List[float] = []
    fit_points: List[Tuple[float,float,float]] = []  # (Δkx, Δky, f) for plotting

    # 1D fits along two principal directions (±1..±N steps + center)
    for (d_hat, name) in dirs[:2]:
        seg = seg_len(k0, P[name]) if name in P else 1.0
        s_levels = np.arange(1, int(max(1, cfg.num_steps_dir)) + 1, dtype=float)
        s_signed = np.concatenate((-s_levels[::-1], s_levels)) * (step * seg)
        kvals = [ mp.Vector3(float(k0[0] + s*d_hat[0]), float(k0[1] + s*d_hat[1]), 0.0) for s in s_signed ]

        fvals = []
        for kv in kvals:
            freqs, gaps = band_gaps_at_k(a1, a2, r, eps, kv, pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
            f_b = float(freqs[band_index])
            fvals.append(f_b)
            # record for plotting: delta from k0
            fit_points.append((float(kv.x - k0[0]), float(kv.y - k0[1]), f_b))
            gap_local = min(gaps[band_index,0] if not np.isnan(gaps[band_index,0]) else float("inf"),
                            gaps[band_index,1] if not np.isnan(gaps[band_index,1]) else float("inf"))
            if gap_local < min_gap_neigh: min_gap_neigh = gap_local

        # include center point
        f0_arr, _ = band_gaps_at_k(a1, a2, r, eps, k0_vec, pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
        f0 = float(f0_arr[band_index])
        sfit = np.concatenate((s_signed, [0.0]))
        ffit = np.concatenate((np.array(fvals, float), [f0]))
        fit_points.append((0.0, 0.0, f0))

        a,b,c,R2 = fit_quadratic_1d(sfit, ffit)
        kappas[name] = a
        grads[name] = b
        R2_min = min(R2_min, R2)
        grad_max_abs = max(grad_max_abs, abs(b))

        # curvature-to-gradient ratio at characteristic step
        s_char = step * seg
        quad_ratios.append( (abs(a)*s_char) / (abs(b)+1e-12) )

    # Hessian reconstruction via directional curvatures + diagonal direction
    d1 = dirs[0][0]; d2 = dirs[1][0]
    e1 = d1 / np.linalg.norm(d1); e2 = d2 / np.linalg.norm(d2)
    kappa_e1 = list(kappas.values())[0] if len(kappas)>0 else 0.0
    kappa_e2 = list(kappas.values())[1] if len(kappas)>1 else 0.0
    ediag = (e1 + e2); ediag = ediag/np.linalg.norm(ediag)

    s_levels = np.arange(1, int(max(1, cfg.num_steps_dir)) + 1, dtype=float)
    s_signed = np.concatenate((-s_levels[::-1], s_levels)) * step
    kvals = [ mp.Vector3(float(k0[0] + s*ediag[0]), float(k0[1] + s*ediag[1]), 0.0) for s in s_signed ]
    fvals = []
    for kv in kvals:
        freqs, gaps = band_gaps_at_k(a1, a2, r, eps, kv, pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
        f_b = float(freqs[band_index])
        fvals.append(f_b)
        fit_points.append((float(kv.x - k0[0]), float(kv.y - k0[1]), f_b))
        gap_local = min(gaps[band_index,0] if not np.isnan(gaps[band_index,0]) else float("inf"),
                        gaps[band_index,1] if not np.isnan(gaps[band_index,1]) else float("inf"))
        if gap_local < min_gap_neigh: min_gap_neigh = gap_local
    f0_arr, _ = band_gaps_at_k(a1, a2, r, eps, mp.Vector3(float(k0[0]), float(k0[1]), 0.0),
                               pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
    f0 = float(f0_arr[band_index])
    sfit = np.concatenate((s_signed, [0.0]))
    ffit = np.concatenate((np.array(fvals, float), [f0]))
    fit_points.append((0.0, 0.0, f0))
    a_diag, b_diag, c_diag, R2_diag = fit_quadratic_1d(sfit, ffit)
    R2_min = min(R2_min, R2_diag)
    grad_max_abs = max(grad_max_abs, abs(b_diag))

    # update quadratic-ness with diagonal
    quad_ratios.append( (abs(a_diag) * step) / (abs(b_diag)+1e-12) )

    H11 = kappa_e1
    H22 = kappa_e2
    H12 = a_diag - 0.5*(H11 + H22)
    H = np.array([[H11, H12],[H12, H22]], float)
    lam, _ = np.linalg.eig(H)
    lam_min, lam_max = float(np.min(lam)), float(np.max(lam))
    same_sign = (lam_min*lam_max) > 0
    extremum_type = "min" if lam_min>0 and lam_max>0 else ("max" if lam_min<0 and lam_max<0 else "saddle")

    # center gaps at k0
    _, gaps0 = band_gaps_at_k(a1, a2, r, eps, mp.Vector3(float(k0[0]), float(k0[1]), 0.0),
                              pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
    gap_center = min(
        gaps0[band_index,0] if not np.isnan(gaps0[band_index,0]) else float("inf"),
        gaps0[band_index,1] if not np.isnan(gaps0[band_index,1]) else float("inf")
    )
    gap_center = float(gap_center)
    min_gap_neigh = float(min_gap_neigh) if math.isfinite(min_gap_neigh) else 0.0

    return dict(
        H11=H11, H22=H22, H12=H12,
        lam_min=lam_min, lam_max=lam_max,
        R2_min=float(R2_min),
        grad_max_abs=float(grad_max_abs),
        min_gap_center=gap_center,
        min_gap_neigh=min_gap_neigh,
        ok_same_sign=int(same_sign),
        extremum_type=extremum_type,
        f0=f0,
        quad_ratio_min=float(min(quad_ratios) if quad_ratios else 0.0),
        fit_points=np.array(fit_points, float) if fit_points else None,
    )

# ------------------------- Scoring ---------------------------------------------------------------

def composite_score(cfg: Config, metrics: Dict[str,float]) -> float:
    if metrics["min_gap_center"] < cfg.min_gap_tol:
        return 0.0
    if metrics["R2_min"] < cfg.min_R2:
        return 0.0
    q_gapc = metrics["min_gap_center"]
    q_gapn = metrics["min_gap_neigh"]
    q_R2   = metrics["R2_min"]**2
    q_grad = 1.0 / (1.0 + (metrics["grad_max_abs"]/max(cfg.grad_tol,1e-6))**2)
    q_sign = 1.0 if metrics["ok_same_sign"] else 0.4
    curv_mag = max(abs(metrics["lam_min"]), abs(metrics["lam_max"]))
    q_curv = 1.0 if curv_mag >= cfg.min_abs_curv else (curv_mag/max(cfg.min_abs_curv,1e-9))
    J = (cfg.w_gap_center*q_gapc + cfg.w_gap_neigh*q_gapn) * \
        (q_R2**cfg.w_R2) * (q_grad**cfg.w_grad) * (q_sign**cfg.w_eigs_same) * (q_curv**cfg.w_curv_mag)
    # new quadratic-ness term: penalize linear behavior
    q_quad = min(1.0, metrics.get("quad_ratio_min", 0.0) / max(cfg.quad_target_ratio, 1e-9))
    J *= (q_quad ** cfg.w_quadness)
    if cfg.favor_min_or_max is not None:
        want = cfg.favor_min_or_max.lower()
        if (want == "min" and metrics["extremum_type"] != "min") or \
           (want == "max" and metrics["extremum_type"] != "max"):
            J *= 0.4
    return float(max(0.0, J))

# ------------------------- Band diagram + surface plots ------------------------------------------

def compute_band_diagram(cfg: Config, a1: np.ndarray, a2: np.ndarray, r: float, eps: float,
                         pol: str, P: Dict[str,np.ndarray], klabels: List[str]) -> Tuple[np.ndarray, Dict[str,int]]:
    kpts, kidx = build_kpath(P, klabels, cfg.kpath_pts)
    lat, geom, mat = monolayer_geometry(a1, a2, r, eps)
    with mp_quiet(cfg.suppress_mpb_output):
        ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mat,
                            k_points=kpts, resolution=int(cfg.resolution_report),
                            num_bands=int(cfg.num_bands_report), dimensions=2)
        if cfg.polarization.strip().lower() == "tm": ms.run_tm()
        else: ms.run_te()
        freqs = np.array(ms.all_freqs, float)
    return freqs, kidx

def surface_sample_multi(cfg: Config, a1: np.ndarray, a2: np.ndarray, r: float, eps: float,
                         pol: str, k0: np.ndarray, center_band: int) -> Tuple[np.ndarray, np.ndarray, Dict[int,np.ndarray]]:
    """
    Sample ω over a square grid around k0 for the center band and up to two neighbors above/below.
    Returns (xs, ys, Zs_dict) where Zs_dict maps band_index -> Z grid.
    """
    half = cfg.surf_half_width
    N = int(cfg.surf_N)
    xs = np.linspace(-half, half, N)
    ys = np.linspace(-half, half, N)
    Zs: Dict[int,np.ndarray] = {}
    b_list = [b for b in range(max(0, center_band-2), min(cfg.num_bands_scan-1, center_band+2)+1)]
    for b in b_list:
        Zs[b] = np.zeros((N,N), float)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            kv = mp.Vector3(float(k0[0] + x), float(k0[1] + y), 0.0)
            freqs = run_mpb_single_k(a1, a2, r, eps, kv, pol, cfg.num_bands_scan, cfg.resolution_grid, cfg.suppress_mpb_output)
            for b in b_list:
                Zs[b][iy, ix] = freqs[b]
    return xs, ys, Zs

def plot_report_png(path: str, cfg: Config, lattice: str, P: Dict[str,np.ndarray], klabels: List[str],
                    freqs: np.ndarray, kidx: Dict[str,int], highlight_label: str, center_band: int,
                    xs: np.ndarray, ys: np.ndarray, Zs: Dict[int,np.ndarray],
                    metrics: Dict[str,float], r: float, eps: float):
    try:
        Nk, Nb = freqs.shape
        x = np.arange(Nk)
        fig = plt.figure(figsize=(13.6, 7.0), constrained_layout=False)
        gs = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[2.2, 1.6, 0.9])
        axb = fig.add_subplot(gs[0,0])
        axm = fig.add_subplot(gs[0,1], projection="3d")
        axt = fig.add_subplot(gs[0,2])

        # Band diagram
        for b in range(Nb):
            lw = 2.0 if b == center_band else 0.9
            alpha = 1.0 if b == center_band else 0.9
            axb.plot(x, freqs[:, b], lw=lw, alpha=alpha, color="0.15")
        ticks = []; ticklabels = []
        for lbl in klabels:
            xi = kidx[lbl]; axb.axvline(xi, color="0.8", lw=0.8)
            ticks.append(xi); ticklabels.append(lbl)
        if highlight_label in kidx:
            xi = kidx[highlight_label]; yi = freqs[xi, center_band]
            axb.scatter([xi], [yi], s=70, c="crimson", zorder=10)
        axb.set_xlim(0, Nk-1); axb.set_xticks(ticks); axb.set_xticklabels(ticklabels)
        axb.set_xlabel("k-path"); axb.set_ylabel("Frequency (a/λ)")
        y_max = float(np.max(freqs))*1.10
        axb.set_ylim(0.0, y_max)
        axb.set_title(f"{lattice} — pol={cfg.polarization}, bands={cfg.band_indices}, r={r:.3f}, ε={eps:.2f}")

        # Surface panel: flatter angle and z clipped to sampled band surfaces
        X, Y = np.meshgrid(xs, ys)
        all_vals = []
        for Z in Zs.values():
            all_vals.append(Z.min()); all_vals.append(Z.max())
        z_min = float(min(all_vals)); z_max = float(max(all_vals))
        pad = 0.05*(z_max - z_min + 1e-12)
        axm.set_zlim(z_min - pad, z_max + pad)
        axm.view_init(elev=25, azim=-55)

        if center_band in Zs:
            axm.plot_surface(X, Y, Zs[center_band], linewidth=0.1, antialiased=True, shade=True)
        for delta in [-2, -1, 1, 2]:
            b = center_band + delta
            if b in Zs:
                axm.plot_wireframe(X, Y, Zs[b], rstride=2, cstride=2, linewidth=0.7, alpha=0.85)
        axm.set_xlabel("Δk_x"); axm.set_ylabel("Δk_y"); axm.set_zlabel("ω")
        axm.set_title(f"Local dispersion around {highlight_label}, band {center_band}")

        # Right report card
        axt.axis("off")
        lines = [
            f"extremum: {metrics['extremum_type']}",
            f"ω0: {metrics['f0']:.6f}",
            f"gaps:",
            f"  center = {metrics['min_gap_center']:.5f}",
            f"  neigh  = {metrics['min_gap_neigh']:.5f}",
            f"λ_min/max:",
            f"  {metrics['lam_min']:.3e} / {metrics['lam_max']:.3e}",
            f"R²_min: {metrics['R2_min']:.5f}",
            f"|b|max:  {metrics['grad_max_abs']:.2e}",
            f"score J: {metrics['J']:.6f}",
            f"params:",
            f"  r = {r:.4f}",
            f"  ε = {eps:.3f}",
            f"  k = {highlight_label}",
            f"  b = {center_band}",
        ]
        axt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

        fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.08, wspace=0.28)
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as e:
        log.warning("Report plot failed for %s: %s", path, e)

# ------------------------- Parallelization control ----------------------------------------------

PARALLEL_ENABLED = True
NUM_CORES = max(1, mp_proc.cpu_count() - 1)
START_METHOD: Optional[str] = None
POOL_CHUNKSIZE: Optional[int] = None

# Additional controls for report parallelism
PARALLEL_REPORTS: bool = True
REPORT_CORES: Optional[int] = None        # default resolves to min(NUM_CORES, 4)
REPORT_CHUNKSIZE: Optional[int] = None

# Globals for worker reuse
_CFG_G: Optional[Config] = None
_A1_G: Optional[np.ndarray] = None
_A2_G: Optional[np.ndarray] = None
_P_G: Optional[Dict[str, np.ndarray]] = None

def _init_pool_globals(cfg_in: Config, a1_in: np.ndarray, a2_in: np.ndarray, P_in: Dict[str, np.ndarray]):
    # Avoid oversubscription when running many processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    global _CFG_G, _A1_G, _A2_G, _P_G
    _CFG_G, _A1_G, _A2_G, _P_G = cfg_in, a1_in, a2_in, P_in

# ------------------------- Parallel worker functions --------------------------------------------

def _grid_worker(params: Tuple[float, float, str, int]) -> Optional[Tuple[Candidate, Dict[str, float]]]:
	"""Worker function for parallel grid search evaluation."""
	r, eps, k_label, b = params
	try:
		assert _CFG_G is not None and _A1_G is not None and _A2_G is not None and _P_G is not None
		met = neighborhood_metrics(_CFG_G, _A1_G, _A2_G, _CFG_G.polarization, int(b), k_label, float(r), float(eps), _P_G)
		J = composite_score(_CFG_G, met)
		met["J"] = J
		cand = Candidate(float(r), float(eps), k_label, int(b), _CFG_G.polarization)
		return (cand, met)
	except Exception as e:
		log.debug("Eval failed at r=%.3f eps=%.2f k=%s b=%d: %s", r, eps, k_label, b, e)
		return None

def _opt_worker(params: Tuple[str, int, Optional[Tuple[float, float]]]) -> Optional[Tuple[Candidate, Dict[str, float]]]:
	"""Worker function for parallel optimization."""
	k_label, b, x0_hint = params
	assert _CFG_G is not None and _A1_G is not None and _A2_G is not None and _P_G is not None

	if x0_hint is not None:
		x0 = np.array(x0_hint, float)
	else:
		x0 = np.array([(_CFG_G.r_min + _CFG_G.r_max) / 2.0, (_CFG_G.eps_min + _CFG_G.eps_max) / 2.0], float)

	def negJ(x: np.ndarray) -> float:
		r, eps = float(x[0]), float(x[1])
		try:
			met = neighborhood_metrics(_CFG_G, _A1_G, _A2_G, _CFG_G.polarization, int(b), k_label, r, eps, _P_G)
			return -composite_score(_CFG_G, met)
		except Exception:
			return 1e3

	from scipy.optimize import minimize, Bounds
	bounds = Bounds(np.array([_CFG_G.r_min, _CFG_G.eps_min], float), np.array([_CFG_G.r_max, _CFG_G.eps_max], float))

	with mp_quiet(_CFG_G.suppress_mpb_output):
		res = minimize(negJ, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=80, ftol=1e-4))

	r_opt, eps_opt = float(res.x[0]), float(res.x[1])
	try:
		met = neighborhood_metrics(_CFG_G, _A1_G, _A2_G, _CFG_G.polarization, int(b), k_label, r_opt, eps_opt, _P_G)
		met["J"] = composite_score(_CFG_G, met)
		cand_opt = Candidate(r_opt, eps_opt, k_label, int(b), _CFG_G.polarization)
		return (cand_opt, met)
	except Exception as e:
		log.debug("Optimization eval failed for k=%s b=%d: %s", k_label, b, e)
		return None

def _report_worker(params: Tuple[int, Tuple['Candidate', Dict[str,float]]]) -> Optional[str]:
    """Worker to render and save a single report; returns output path or None."""
    idx, item = params
    assert _CFG_G is not None and _A1_G is not None and _A2_G is not None and _P_G is not None
    cand, met = item
    try:
        klabels = default_k_path(_CFG_G.lattice)
        freqs, kidx = compute_band_diagram(_CFG_G, _A1_G, _A2_G, cand.r, cand.eps, cand.pol, _P_G, klabels)
        k0 = _P_G[cand.k_label]
        xs, ys, Zs = surface_sample_multi(_CFG_G, _A1_G, _A2_G, cand.r, cand.eps, cand.pol, k0, cand.band_index)
        out_png = os.path.join(OUTDIR, f"report_{idx:03d}_{_CFG_G.lattice}_b{cand.band_index}_{cand.k_label}.png")
        plot_report_png(path=out_png, cfg=_CFG_G, lattice=_CFG_G.lattice, P=_P_G, klabels=klabels,
                        freqs=freqs, kidx=kidx, highlight_label=cand.k_label, center_band=cand.band_index,
                        xs=xs, ys=ys, Zs=Zs, metrics=met, r=cand.r, eps=cand.eps)
        return out_png
    except Exception as e:
        log.warning("Report generation failed for candidate %d: %s", idx, e)
        return None

# ------------------------- Grid search & (optional) optimization ---------------------------------

@dataclass
class Candidate:
    r: float
    eps: float
    k_label: str
    band_index: int
    pol: str

def grid_search(cfg: Config) -> List[Tuple[Candidate, Dict[str,float]]]:
    a1, a2 = canonical_lattice_vectors(cfg.lattice) if cfg.lattice.lower() in ("square","triangular","hexagonal","rectangular","oblique") else (np.array(cfg.a1), np.array(cfg.a2))
    P = hs_points(cfg.lattice)
    k_labels = list(P.keys())
    if not cfg.high_symmetry_only:
        interior = {"C1": np.array([0.25, 0.15]), "C2": np.array([0.33, 0.22])}
        P = {**P, **interior}
        k_labels = list(P.keys())

    r_list = np.linspace(cfg.r_min, cfg.r_max, int(cfg.Nr))
    eps_list = np.linspace(cfg.eps_min, cfg.eps_max, int(cfg.Neps))

    total = len(r_list) * len(eps_list) * len(k_labels) * len(cfg.band_indices)
    log.info("Grid size: r(%d) × eps(%d) × k(%d) × bands(%d) = %d points",
             len(r_list), len(eps_list), len(k_labels), len(cfg.band_indices), total)

    results: List[Tuple[Candidate, Dict[str,float]]] = []
    failures = 0

    # Prepare all parameter combinations (small tuple to reduce IPC)
    param_list = [(r, eps, k_label, b) for r, eps, k_label, b in itertools.product(r_list, eps_list, k_labels, cfg.band_indices)]

    # Initialize globals for sequential execution and to warm imports
    _init_pool_globals(cfg, a1, a2, P)

    if PARALLEL_ENABLED and NUM_CORES > 1:
        log.info("Running parallel grid search with %d cores", NUM_CORES)
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold blue]Scan (parallel)"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("grid", total=total); progress.start()

        # Use appropriate start method context
        ctx = mp_proc.get_context(START_METHOD) if START_METHOD else mp_proc
        # Tuned chunksize to reduce scheduling overhead
        chunksize = POOL_CHUNKSIZE if POOL_CHUNKSIZE is not None else max(1, total // (NUM_CORES * 8))

        try:
            with ctx.Pool(processes=NUM_CORES, initializer=_init_pool_globals, initargs=(cfg, a1, a2, P)) as pool:
                for result in pool.imap_unordered(_grid_worker, param_list, chunksize=chunksize):
                    if result is not None:
                        results.append(result)
                    else:
                        failures += 1
                    if progress and task is not None:
                        progress.advance(task)
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl+C) during grid scan.")
            raise
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass
    else:
        # Sequential fallback
        log.info("Running sequential grid search")
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold blue]Scan"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("grid", total=total); progress.start()
        try:
            for params in param_list:
                result = _grid_worker(params)
                if result is not None:
                    results.append(result)
                else:
                    failures += 1
                if progress and task is not None: progress.advance(task)
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl+C) during grid scan.")
            raise
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass

    log.info("Scan done: %d ok, %d failures", len(results), failures)
    return results

def deduplicate_top(cands: List[Tuple[Candidate, Dict[str,float]]],
                    dr: float = 1e-3, deps: float = 1e-2) -> List[Tuple[Candidate, Dict[str,float]]]:
    kept: List[Tuple[Candidate, Dict[str,float]]] = []
    for cand, met in sorted(cands, key=lambda cm: cm[1]["J"], reverse=True):
        def close(a,b,t): return abs(a-b) <= t
        if any(close(cand.r, c.r, dr) and close(cand.eps, c.eps, deps) and
               cand.k_label==c.k_label and cand.band_index==c.band_index for (c,_m) in kept):
            continue
        kept.append((cand, met))
    return kept

def optimize_best_across_all(cfg: Config,
                             pool: List[Tuple[Candidate, Dict[str,float]]]) -> Optional[Tuple[Candidate, Dict[str,float]]]:
    """
    For each (k_label, band) combination present in the configuration, run a bounded
    L-BFGS-B optimization in (r, eps) to maximize J. Initialize from the best grid
    point for that pair if available; otherwise from mid-rectangle. Return the single
    best optimized candidate across ALL pairs.
    """
    if not SCIPY_AVAILABLE:
        log.warning("SciPy unavailable; skipping global optimization.")
        return None

    a1, a2 = canonical_lattice_vectors(cfg.lattice)
    P = hs_points(cfg.lattice)
    if not cfg.high_symmetry_only:
        P = {**P, **{"C1": np.array([0.25,0.15]), "C2": np.array([0.33,0.22])}}
    k_labels = list(P.keys())

    # Seed globals for workers
    _init_pool_globals(cfg, a1, a2, P)

    # best initializers from the pool
    bucket: Dict[Tuple[str,int], Tuple[Candidate, Dict[str,float]]] = {}
    for cand, met in pool:
        key = (cand.k_label, cand.band_index)
        if key not in bucket or met["J"] > bucket[key][1]["J"]:
            bucket[key] = (cand, met)

    # Prepare optimization tasks (small tuples)
    opt_tasks = []
    for k_label in k_labels:
        for b in cfg.band_indices:
            if (k_label, b) in bucket:
                x0_hint = (bucket[(k_label,b)][0].r, bucket[(k_label,b)][0].eps)
            else:
                x0_hint = None
            opt_tasks.append((k_label, b, x0_hint))

    best_pair: Optional[Tuple[Candidate, Dict[str,float]]] = None
    total = len(opt_tasks)

    if PARALLEL_ENABLED and NUM_CORES > 1:
        log.info("Running parallel optimization with %d cores", NUM_CORES)
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold green]Optimize (parallel)"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("opt", total=total); progress.start()

        ctx = mp_proc.get_context(START_METHOD) if START_METHOD else mp_proc
        chunksize = POOL_CHUNKSIZE if POOL_CHUNKSIZE is not None else max(1, total // (NUM_CORES * 8))

        try:
            with ctx.Pool(processes=NUM_CORES, initializer=_init_pool_globals, initargs=(cfg, a1, a2, P)) as pool_mp:
                for result in pool_mp.imap_unordered(_opt_worker, opt_tasks, chunksize=chunksize):
                    if result is not None:
                        if best_pair is None or result[1]["J"] > best_pair[1]["J"]:
                            best_pair = result
                    if progress and task is not None:
                        progress.advance(task)
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl+C) during optimization.")
            raise
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass
    else:
        # Sequential fallback
        log.info("Running sequential optimization")
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold green]Optimize"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("opt", total=total); progress.start()
        try:
            for params in opt_tasks:
                result = _opt_worker(params)
                if result is not None:
                    if best_pair is None or result[1]["J"] > best_pair[1]["J"]:
                        best_pair = result
                if progress and task is not None: progress.advance(task)
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl+C) during optimization.")
            raise
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass

    return best_pair

# ------------------------- CSV & reports ---------------------------------------------------------

def export_csv(path: str, items: List[Tuple[Candidate, Dict[str,float]]], K: int, cfg: Config):
    import csv
    cols = ["lattice","r","eps_bg","k_label","band_index","pol",
            "J","f0","min_gap_center","min_gap_neigh","R2_min","grad_max_abs",
            "H11","H12","H22","lam_min","lam_max","extremum_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (c, m) in sorted(items, key=lambda cm: cm[1]["J"], reverse=True)[:K]:
            w.writerow(dict(lattice=cfg.lattice, r=c.r, eps_bg=c.eps, k_label=c.k_label,
                            band_index=c.band_index, pol=c.pol,
                            J=m["J"], f0=m["f0"], min_gap_center=m["min_gap_center"],
                            min_gap_neigh=m["min_gap_neigh"],
                            R2_min=m["R2_min"], grad_max_abs=m["grad_max_abs"],
                            H11=m["H11"], H12=m["H12"], H22=m["H22"],
                            lam_min=m["lam_min"], lam_max=m["lam_max"],
                            extremum_type=m["extremum_type"]))

def generate_reports(cfg: Config, items: List[Tuple[Candidate, Dict[str,float]]], max_reports: Optional[int] = None):
    if not cfg.plot_reports: return
    a1, a2 = canonical_lattice_vectors(cfg.lattice)
    P = hs_points(cfg.lattice)
    if not cfg.high_symmetry_only:
        P = {**P, **{"C1": np.array([0.25,0.15]), "C2": np.array([0.33,0.22])}}

    # seed globals for workers
    _init_pool_globals(cfg, a1, a2, P)

    sorted_items = sorted(items, key=lambda cm: cm[1]["J"], reverse=True)
    if max_reports is not None:
        sorted_items = sorted_items[:max_reports]
    total = len(sorted_items)
    if total == 0:
        log.info("No reports to generate.")
        return

    if PARALLEL_REPORTS and (REPORT_CORES or NUM_CORES) > 1:
        cores = REPORT_CORES if REPORT_CORES is not None else NUM_CORES
        log.info("Generating %d reports in parallel with %d cores", total, cores)
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold magenta]Reports (parallel)"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("reports", total=total); progress.start()

        ctx = mp_proc.get_context(START_METHOD) if START_METHOD else mp_proc
        chunksize = REPORT_CHUNKSIZE if REPORT_CHUNKSIZE is not None else max(1, total // (cores * 4))
        try:
            with ctx.Pool(processes=cores, initializer=_init_pool_globals, initargs=(cfg, a1, a2, P)) as pool:
                for out in pool.imap_unordered(_report_worker, list(enumerate(sorted_items, start=1)), chunksize=chunksize):
                    if out:
                        log.info("Saved report: %s", out)
                    if progress and task is not None:
                        progress.advance(task)
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass
    else:
        log.info("Generating %d reports sequentially", total)
        progress = None; task = None
        if RICH:
            progress = Progress(
                TextColumn("[bold magenta]Reports"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>5.1f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                console=console,
            )
            task = progress.add_task("reports", total=total); progress.start()
        try:
            for i, itm in enumerate(sorted_items, start=1):
                _ = _report_worker((i, itm))
                if progress and task is not None:
                    progress.advance(task)
        finally:
            if progress:
                try: progress.stop()
                except Exception: pass

# ------------------------- CLI -------------------------------------------------------------------

def parse_bands(value: str) -> List[int]:
    """Parse comma-separated band indices from string."""
    try:
        return [int(x.strip()) for x in value.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid band indices: {value}") from e

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 1 (Monolayer) — robust extremum search & scoring")
    ap.add_argument("--lattice", default="square", choices=["square","triangular","rectangular","oblique"])
    ap.add_argument("--hs-only", action="store_true", help="Consider only high-symmetry points (default follows config).")
    ap.add_argument("--no-hs-only", action="store_true", help="Allow interior k-points too.")
    ap.add_argument("--pol", default="TM", choices=["TE","TM"])
    ap.add_argument("--bands", type=parse_bands, default=None, help="Comma-separated list of 0-based band indices, e.g. '2,3,4'. If omitted, uses Config default [0..6].")
    ap.add_argument("--num-bands-scan", type=int, default=8)
    ap.add_argument("--num-bands-report", type=int, default=10)

    ap.add_argument("--r-min", type=float, default=0.12); ap.add_argument("--r-max", type=float, default=0.48); ap.add_argument("--Nr", type=int, default=7)
    ap.add_argument("--eps-min", type=float, default=2.5); ap.add_argument("--eps-max", type=float, default=10.0); ap.add_argument("--Neps", type=int, default=8)

    ap.add_argument("--favor", choices=["min","max"], help="Prefer minima or maxima.")
    ap.add_argument("--K", type=int, default=None, help="Top-K seeds to export (default: Config.top_K).")
    ap.add_argument("--out", default="seeds_stage1_monolayer.csv")

    ap.add_argument("--verbose", action="store_true", help="Show MPB output (unsuppressed).")
    ap.add_argument("--no-plots", action="store_true", help="Disable PNG report generation.")
    ap.add_argument("--max-reports", type=int, help="Maximum number of reports to generate.")

    ap.add_argument("--optimize", action="store_true", help="Run final global SciPy optimization over all (k_label, band) pairs.")
    # Parallelization arguments
    ap.add_argument("--parallel", action="store_true", default=True, help="Enable parallel processing (default: True)")
    ap.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    ap.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use (default: cpu_count()-1)")
    ap.add_argument("--start-method", choices=["auto","fork","spawn","forkserver"], default="auto",
                    help="Multiprocessing start method (default: auto)")
    ap.add_argument("--chunksize", type=int, default=None, help="Pool chunksize override (advanced).")
    # Report-parallelization arguments
    ap.add_argument("--parallel-reports", action="store_true", default=True, help="Enable parallel report generation (default: True)")
    ap.add_argument("--no-parallel-reports", action="store_true", help="Disable parallel report generation")
    ap.add_argument("--report-cores", type=int, default=None, help="Cores for report generation (default: min(4, cores))")
    ap.add_argument("--report-chunksize", type=int, default=None, help="Chunksize for report pool (advanced).")
    
    return ap.parse_args()

def main():
    global cfg, PARALLEL_ENABLED, NUM_CORES, START_METHOD, POOL_CHUNKSIZE
    global PARALLEL_REPORTS, REPORT_CORES, REPORT_CHUNKSIZE
    args = parse_args()

    # Configure parallelization
    if args.no_parallel:
        PARALLEL_ENABLED = False
    elif args.parallel:
        PARALLEL_ENABLED = True

    if args.cores is not None:
        NUM_CORES = max(1, min(args.cores, mp_proc.cpu_count()))

    START_METHOD = None if args.start_method == "auto" else args.start_method
    POOL_CHUNKSIZE = args.chunksize

    PARALLEL_REPORTS = False if args.no_parallel_reports else True
    REPORT_CORES = None if args.report_cores is None else max(1, min(args.report_cores, mp_proc.cpu_count()))
    REPORT_CHUNKSIZE = args.report_chunksize

    if PARALLEL_ENABLED:
        log.info("Parallelization enabled with %d cores%s", NUM_CORES, f", start={START_METHOD}" if START_METHOD else "")
    else:
        log.info("Running in sequential mode")

    cfg = Config(
        lattice=args.lattice,
        high_symmetry_only=True if args.hs_only else (False if args.no_hs_only else Config.high_symmetry_only),
        polarization=args.pol, band_indices=(args.bands if args.bands is not None else Config().band_indices),
        num_bands_scan=args.num_bands_scan, num_bands_report=args.num_bands_report,
        r_min=args.r_min, r_max=args.r_max, Nr=args.Nr,
        eps_min=args.eps_min, eps_max=args.eps_max, Neps=args.Neps,
        favor_min_or_max=args.favor,
        top_K=(args.K if args.K is not None else Config().top_K),
        plot_reports=not args.no_plots,
        max_reports=args.max_reports,
        suppress_mpb_output=not args.verbose
    )

    try:
        # Grid
        t_results = grid_search(cfg)
        t_results = deduplicate_top(t_results)

        # Global continuous optimization across all (k_label, band) pairs
        if args.optimize:
            try:
                opt_pair = optimize_best_across_all(cfg, t_results)
                if opt_pair is not None:
                    t_results.append(opt_pair)
                    t_results = deduplicate_top(t_results)
                    j = opt_pair[1]["J"]; c = opt_pair[0]
                    msg = f"[opt] best J={j:.6f}  r={c.r:.5f}  eps={c.eps:.4f}  k={c.k_label}  band={c.band_index}"
                    if RICH and console:
                        console.print(f"[bold green]{msg}[/bold green]")
                    else:
                        print(msg)
                else:
                    log.info("Optimization finished, but produced no valid candidate.")
            except KeyboardInterrupt:
                log.warning("Interrupted by user (Ctrl+C) during optimization.")
                raise
            except Exception as e:
                log.warning("Global optimization failed: %s", e)
        else:
            log.info("Global optimization disabled. Use --optimize to enable.")

        # Export (after candidate list is finalized)
        export_csv(args.out, t_results, cfg.top_K, cfg)
        log.info("Exported top-%d seeds to %s", cfg.top_K, args.out)

        # Reports
        try:
            generate_reports(cfg, t_results[:cfg.top_K], cfg.max_reports)
        except KeyboardInterrupt:
            log.warning("Interrupted by user (Ctrl+C) during plotting.")
            raise
        except Exception as e:
            log.warning("Report generation skipped: %s", e)

    except KeyboardInterrupt:
        log.warning("Interrupted by user. Exiting.")
        sys.exit(130)

if __name__ == "__main__":
    # Ensure proper cleanup on interrupt
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
