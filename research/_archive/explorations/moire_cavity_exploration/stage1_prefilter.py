#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 — Seed generation & validation with robust extremum tests
=================================================================
This file *generates* Stage-1 seeds (grid + optional continuous optimization) and
optionally *validates* an existing CSV of seeds using a more stringent notion of a
"good local extremum" at the chosen high-symmetry k-point.

Key improvements (vs. earlier version):
- Neighborhood sampling: points at ±s and ±2s along at least two principal directions
  that connect k0 to adjacent high-symmetry nodes; s is ≥10% of the HS-segment length.
- Robust 1D quadratic fits f(s) = a s^2 + b s + c per direction with R^2 and slope check.
- 2D Hessian reconstruction from directional curvatures (x, y, diag), eigenanalysis.
- Neighborhood spectral isolation: min gap to adjacent bands across all sampled points.
- Composite quality score J2 that down-weights sharp corners, poor fits, or weak gaps.

Design notes
------------
- Hard-coded CONFIG at top for future relaxation (lattice types, k-sets, etc.).
- Square lattice, high-symmetry only (Γ, X, M) for now.
- Registry mini-cells are 1× at stackings (AA/AB/BA).
- All diagnostics remain in reciprocal space.

Outputs
-------
- seeds_stage1.csv (top-K by objective J2), run log in stage1_outputs/run.log
- Optional band-diagram PNGs in stage1_outputs/
- Validation mode: --validate-csv in.csv --out-validated validated.csv enriches metrics and flags

Code & comments are in English and avoid the 2nd person.
"""
from __future__ import annotations

import os, sys, math, time, argparse, contextlib, logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from matplotlib.patches import Rectangle

# ================================== CONFIG =======================================================

CONFIG = dict(
    # Geometry / lattice
    lattice_type = "square",
    a1 = (1.0, 0.0),
    a2 = (0.0, 1.0),

    # Search spaces
    grid = dict(
        hole_radius_list = np.linspace(0.10, 0.48, 5).tolist(),   # conservative r sweep
        eps_bg_list      = np.linspace(2.5, 8.0, 5).tolist(),
        polarizations    = ["TM"],
        band_indices     = [2,3,4,5],            # target mid bands often host edges
        k_labels         = ["M", "X", "Γ"],
        # two-phase scan knobs
        top_fraction_for_full = 0.15,  # evaluate full neighborhood for top 15% by J1
        min_full = 32,
        max_full = 128,
    ),

    # Optimization (continuous) over (hole_radius_a, eps_bg)
    optimization = dict(
        enable           = True,
        polarizations    = ["TM"],
        band_indices     = [2,3,4,5],
        k_labels         = ["M", "X", "Γ"],
        initial_guess    = (0.38, 4.0),
        bounds           = ((0.08, 0.49), (2.5, 10.0)),
        maxiter          = 50,
    ),

    # Local neighborhood sampling (for extremum tests)
    neighborhood = dict(
        step_fraction = 0.06,       # relax step to probe closer neighborhood
        second_step_multiplier = 2, # sample also at 2× step
        directions = "auto",        # "auto" for HS-dependent star, or ["x","y","diag"]
        min_R2 = 0.8,               # slightly relaxed fit quality
        grad_tol = 3e-2,            # allow small residual slope
        min_gap_tol = 0.005,        # allow smaller neighborhood isolation
        min_abs_curv = 5e-4,        # accept gentler curvature
        favor_min_or_max = None,    # "min", "max", or None (accept either)
    ),

    # Envelope curvature estimator (legacy central differences used as a baseline)
    finite_diff = dict(
        delta_k = 0.02,
        isotropic_average = True,
    ),

    # MPB solver knobs (separate fast scan vs report)
    mpb = dict(
        resolution_grid   = 36,   # faster for scanning
        resolution_report = 48,   # higher quality for plots
        num_bands_scan    = 8,
        num_bands_report  = 10,
        suppress_output   = True,
    ),

    # Objective weights (base J1) and extremum-quality multipliers
    objective = dict(
        w_contrast  = 1.0,   # |ω_AB - ω_AA|
        w_gapmin    = 0.8,   # min gap at k0
        w_absalpha  = 0.2,   # |α| (legacy curvature proxy)
        # Neighborhood quality multipliers (0..1). J2 = J1 * Q
        weight_R2        = 0.6,   # influences how strongly fit quality matters
        weight_grad      = 0.6,
        weight_gap_neigh = 1.0,
        weight_sign_cons = 0.5,
        weight_eigs_same = 0.5,
    ),

    # Seed selection
    seeds = dict(
        K = 16,
        include_optimized = True,
        min_param_separation = dict(hole_radius=1e-3, eps_bg=1e-2),
    ),

    # Visualization toggles
    visuals = dict(
        plot_reciprocal_sample = True,
        plot_for_first_only    = True,
    ),

    # Band-diagram reports
    reports = dict(
        enable = True,
        kpath_labels = ["Γ", "X", "M", "Γ"],
        points_per_segment = 48
    ),
)

# ================================== Logging / Rich ==============================================

LOG_DIR = "stage1_outputs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "run.log")

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.logging import RichHandler
    RICH = True
except Exception:
    RICH = False

handlers: List[logging.Handler] = []
fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.DEBUG); handlers.append(fh)
if RICH:
    ch = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)
else:
    ch = logging.StreamHandler(stream=sys.stdout); ch.setLevel(logging.INFO)
handlers.append(ch)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
log = logging.getLogger("stage1")
console = Console() if RICH else None

# ================================== Imports that may fail =======================================

# Predeclare modules to avoid "possibly unbound" warnings
mp = None  # type: ignore
mpb = None  # type: ignore
ml = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception as e:
    log.error("Matplotlib required: %s", e); raise

MEEP_AVAILABLE = False
try:
    import meep as mp
    from meep import mpb
    MEEP_AVAILABLE = True
except Exception as e:
    log.warning("Meep/MPB not importable; computations are disabled. %s", e)

ML_AVAILABLE = False
try:
    import moire_lattice_py as ml
    ML_AVAILABLE = True
except Exception as e:
    log.warning("moire_lattice_py not importable; reciprocal plots will be minimal. %s", e)

SCIPY_AVAILABLE = False
try:
    from scipy.optimize import minimize, Bounds
    SCIPY_AVAILABLE = True
except Exception as e:
    log.warning("SciPy not importable; continuous optimization will be skipped. %s", e)

# ================================== Utilities ====================================================

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

@dataclass
class Candidate:
    lattice_type: str
    a1x: float; a1y: float
    a2x: float; a2y: float
    hole_radius_a: float
    eps_bg: float
    polarization: str
    band_index: int
    k_label: str
    twist_deg: float = 1.10

# Lattice helpers
def mp_lattice_from_vectors(a1: np.ndarray, a2: np.ndarray):
    return mp.Lattice(size=mp.Vector3(1,1,0),
                      basis1=mp.Vector3(float(a1[0]), float(a1[1]), 0),
                      basis2=mp.Vector3(float(a2[0]), float(a2[1]), 0))

def k0_from_label_square(k_label: str) -> mp.Vector3:
    if k_label in ("Γ","G","Gamma","gamma"): return mp.Vector3(0,0,0)
    if k_label.upper() == "X": return mp.Vector3(0.5,0.0,0.0)
    if k_label.upper() == "M": return mp.Vector3(0.5,0.5,0.0)
    raise ValueError(f"Unsupported k_label for square: {k_label!r}")

# Geometry
def monolayer_geometry(a1: np.ndarray, a2: np.ndarray, radius: float, eps_bg: float):
    lat = mp_lattice_from_vectors(a1, a2)
    geom = [mp.Cylinder(radius=float(radius), material=mp.air, center=mp.Vector3(0,0,0))]
    mat = mp.Medium(epsilon=float(eps_bg))
    return lat, geom, mat

def registry_basis_uv(reg: str) -> List[Tuple[float, float]]:
    if reg == "AA": return [(0.0, 0.0), (0.0, 0.0)]
    if reg == "AB": return [(0.0, 0.0), (0.5, 0.5)]
    if reg == "BA": return [(0.0, 0.0), (0.5, -0.5)]
    raise ValueError(f"Unknown registry: {reg}")

def mpb_registry_geometry(a1: np.ndarray, a2: np.ndarray, radius: float, eps_bg: float, reg: str):
    lat = mp_lattice_from_vectors(a1, a2)
    uv = registry_basis_uv(reg)
    geom = [mp.Cylinder(radius=float(radius), material=mp.air, center=mp.Vector3(float(u),float(v),0.0))
            for (u,v) in uv]
    mat = mp.Medium(epsilon=float(eps_bg))
    return lat, geom, mat

# Solver
def run_mpb_single_k(lat, geom, mat, kvec, num_bands: Optional[int] = None, pol: str = "TM", res: Optional[int] = None) -> np.ndarray:
    nb = int(num_bands if num_bands is not None else CONFIG["mpb"]["num_bands_scan"]) 
    rr = int(res if res is not None else CONFIG["mpb"]["resolution_grid"]) 
    ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mat,
                        k_points=[kvec], resolution=rr,
                        num_bands=nb, dimensions=2)
    p = pol.strip().lower()
    if p == "tm": ms.run_tm()
    elif p == "te": ms.run_te()
    else: raise ValueError("polarization must be TE or TM")
    return np.array(ms.all_freqs[0], float)

# Legacy curvature (central diff) at the node
def curvature_alpha_at_k0(a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int, k0, radius: float, eps_bg: float, delta: float) -> Tuple[float,float]:
    lat, geom, mat = monolayer_geometry(a1, a2, radius, eps_bg)
    ex, ey = mp.Vector3(1.0,0.0,0.0), mp.Vector3(0.0,1.0,0.0)
    with mp_quiet(CONFIG["mpb"]["suppress_output"]):
        freqs0 = run_mpb_single_k(lat, geom, mat, k0, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) 
        f0 = freqs0[band_index]
        fxp = run_mpb_single_k(lat, geom, mat, k0 + delta*ex, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
        fxm = run_mpb_single_k(lat, geom, mat, k0 - delta*ex, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
        fyp = run_mpb_single_k(lat, geom, mat, k0 + delta*ey, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
        fym = run_mpb_single_k(lat, geom, mat, k0 - delta*ey, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
    d2x = (fxp + fxm - 2.0*f0) / (delta**2)
    d2y = (fyp + fym - 2.0*f0) / (delta**2)
    alpha = 0.5*(d2x+d2y) if CONFIG["finite_diff"]["isotropic_average"] else (d2x, d2y)
    return float(f0), float(alpha)

def band_gaps_at_k(a1: np.ndarray, a2: np.ndarray, pol: str, kvec, radius: float, eps_bg: float) -> np.ndarray:
    """Return gaps above/below for all bands at this kvec: array shape (num_bands, 2)."""
    lat, geom, mat = monolayer_geometry(a1, a2, radius, eps_bg)
    with mp_quiet(CONFIG["mpb"]["suppress_output"]):
        freqs = run_mpb_single_k(lat, geom, mat, kvec, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) 
    Nb = len(freqs)
    gaps = np.full((Nb, 2), np.nan, float)
    for b in range(Nb):
        if b > 0: gaps[b,0] = freqs[b] - freqs[b-1]
        if b+1 < Nb: gaps[b,1] = freqs[b+1] - freqs[b]
    return freqs, gaps

def band_gaps_at_k0(a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int, k0, radius: float, eps_bg: float) -> Tuple[float, float]:
    freqs, gaps = band_gaps_at_k(a1, a2, pol, k0, radius, eps_bg)
    bi = band_index
    gap_below = gaps[bi,0]
    gap_above = gaps[bi,1]
    return float(gap_below), float(gap_above)

def registry_edges_at_k0(a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int, k0, radius: float, eps_bg: float) -> Dict[str,float]:
    edges = {}
    for reg in ("AA","AB","BA"):
        lat, geom, mat = mpb_registry_geometry(a1, a2, radius, eps_bg, reg)
        with mp_quiet(CONFIG["mpb"]["suppress_output"]):
            freqs = run_mpb_single_k(lat, geom, mat, k0, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) 
        edges[reg] = float(freqs[band_index])
    return edges

# ================================== Neighborhood around k0 ======================================

def hs_neighbors_square(k_label: str) -> List[Tuple[np.ndarray, str]]:
    """Return unit directions (fractional basis) from k_label toward adjacent HS nodes with their names."""
    G = np.array([0.0, 0.0])
    X = np.array([0.5, 0.0])
    M = np.array([0.5, 0.5])
    if k_label in ("Γ","G","Gamma","gamma"):
        dirs = [("X", X-G), ("M", M-G)]
    elif k_label.upper() == "X":
        dirs = [("Γ", G-X), ("M", M-X)]
    elif k_label.upper() == "M":
        dirs = [("X", X-M), ("Γ", G-M)]
    else:
        raise ValueError(f"Unsupported k_label for square: {k_label!r}")
    out: List[Tuple[np.ndarray,str]] = []
    for name, vec in dirs:
        norm = np.linalg.norm(vec)
        if norm == 0: continue
        out.append((vec/norm, name))
    return out

def segment_length(k_from: str, k_to: str) -> float:
    P = dict(Γ=np.array([0.0,0.0]), X=np.array([0.5,0.0]), M=np.array([0.5,0.5]))
    a = P["Γ"] if k_from in ("Γ","G","Gamma","gamma") else P[k_from]
    b = P[k_to]
    return float(np.linalg.norm(b - a))

def fit_quadratic_1d(s: np.ndarray, f: np.ndarray) -> Tuple[float,float,float,float]:
    """
    Fit f(s) = a s^2 + b s + c (least squares). Return (a,b,c,R2).
    Uses both ±s and ±2s samples to stabilize against noise and odd asymmetries.
    """
    X = np.column_stack([s**2, s, np.ones_like(s)])
    coef, *_ = np.linalg.lstsq(X, f, rcond=None)
    a,b,c = coef
    fhat = X @ coef
    ss_res = float(np.sum((f - fhat)**2))
    ss_tot = float(np.sum((f - np.mean(f))**2)) + 1e-16
    R2 = 1.0 - ss_res/ss_tot
    return float(a), float(b), float(c), float(R2)

def quick_evaluate_candidate(a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int, k_label: str, r: float, eps: float) -> Dict[str, float]:
    """Cheap metrics at k0 only: gaps and registry contrast. Returns a dict with J1 and placeholders for others."""
    k0 = k0_from_label_square(k_label)
    # Gaps at k0
    try:
        freqs0, gaps0 = band_gaps_at_k(a1, a2, pol, k0, r, eps)
        gap_below = float(gaps0[band_index,0])
        gap_above = float(gaps0[band_index,1])
        f0 = float(freqs0[band_index])
    except Exception as e:
        log.debug("quick gaps failed at k0 (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        gap_below = 0.0; gap_above = 0.0; f0 = float("nan")
    # Registry contrast (AA vs AB only for speed)
    try:
        lat, geomAA, mat = mpb_registry_geometry(a1, a2, r, eps, "AA")
        with mp_quiet(CONFIG["mpb"]["suppress_output"]):
            fAA = run_mpb_single_k(lat, geomAA, mat, k0, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
        lat, geomAB, mat = mpb_registry_geometry(a1, a2, r, eps, "AB")
        with mp_quiet(CONFIG["mpb"]["suppress_output"]):
            fAB = run_mpb_single_k(lat, geomAB, mat, k0, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) [band_index]
        contrast = float(fAB - fAA)
    except Exception as e:
        log.debug("quick contrast failed at k0 (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        contrast = 0.0; fAA = float("nan"); fAB = float("nan")
    alpha = 0.0  # skip curvature in quick pass
    J1 = objective_J1(contrast, gap_below, gap_above, alpha)
    return dict(
        omega0=f0, alpha=alpha, gap_below=gap_below, gap_above=gap_above,
        omega_AA=(locals().get("fAA") if "fAA" in locals() else float("nan")),
        omega_AB=(locals().get("fAB") if "fAB" in locals() else float("nan")),
        omega_BA=float("nan"), contrast_AA_AB=abs(contrast), J1=J1,
        Q=0.5, J2=J1*0.5,
    )

def evaluate_candidate(a1: np.ndarray, a2: np.ndarray, pol: str, band_index: int, k_label: str, r: float, eps: float
                       ) -> Dict[str,float]:
    k0 = k0_from_label_square(k_label)

    # Legacy single-node metrics with robust fallbacks
    try:
        f0, alpha = curvature_alpha_at_k0(a1, a2, pol, band_index, k0, r, eps, CONFIG["finite_diff"]["delta_k"])
    except Exception as e:
        log.debug("alpha@k0 failed (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        with mp_quiet(CONFIG["mpb"]["suppress_output"]):
            lat, geom, mat = monolayer_geometry(a1, a2, r, eps)
            freqs0 = run_mpb_single_k(lat, geom, mat, k0, CONFIG["mpb"]["num_bands_scan"], pol, CONFIG["mpb"]["resolution_grid"]) 
            f0 = float(freqs0[band_index])
        alpha = 0.0
    try:
        gap_below, gap_above = band_gaps_at_k0(a1, a2, pol, band_index, k0, r, eps)
    except Exception as e:
        log.debug("gaps@k0 failed (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        gap_below = 0.0; gap_above = 0.0
    try:
        edges = registry_edges_at_k0(a1, a2, pol, band_index, k0, r, eps)
    except Exception as e:
        log.debug("registry edges failed (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        edges = {"AA": f0, "AB": f0, "BA": f0}
    contrast = edges["AB"] - edges["AA"]
    J1 = objective_J1(contrast, gap_below, gap_above, alpha)

    # Robust neighborhood metrics with fallback
    try:
        neigh = neighborhood_star_metrics(a1, a2, pol, band_index, k_label, r, eps)
        Q = extremum_quality_multiplier(neigh)
    except Exception as e:
        log.debug("neighborhood metrics failed (%s,b%d,%s,r=%.3f,eps=%.2f): %s", pol, band_index, k_label, r, eps, e)
        neigh = dict(kappa_x=float("nan"), kappa_y=float("nan"), kappa_diag=float("nan"), Hxx=float("nan"), Hyy=float("nan"), Hxy=float("nan"), lam_min=float("nan"), lam_max=float("nan"), R2_min=0.0, grad_max_abs=1.0, min_gap_neigh=0.0, ok_minmax_sign=0, extremum_type="indeterminate")
        Q = 0.5
    J2 = J1 * Q

    out = dict(
        omega0=f0, alpha=alpha, gap_below=gap_below, gap_above=gap_above,
        omega_AA=edges.get("AA", float("nan")), omega_AB=edges.get("AB", float("nan")), omega_BA=edges.get("BA", float("nan")),
        contrast_AA_AB=abs(contrast), J1=J1, Q=Q, J2=J2,
    )
    out.update(neigh)
    return out

def objective_J1(contrast: float, gap_lower: float, gap_upper: float, alpha: float) -> float:
    """Primary merit at k0 combining registry contrast, min gap, and curvature proxy."""
    obj_any: Any = CONFIG.get("objective", {})
    obj: Dict[str, Any] = obj_any if isinstance(obj_any, dict) else {}
    w_contrast = float(obj.get("w_contrast", 1.0))
    w_gapmin   = float(obj.get("w_gapmin", 0.8))
    w_absalpha = float(obj.get("w_absalpha", 0.2))
    gaps = [g for g in (gap_lower, gap_upper) if not (g is None or math.isnan(g))]
    gmin = min(gaps) if gaps else 0.0
    J = w_contrast*abs(contrast) + w_gapmin*gmin + w_absalpha*abs(alpha)
    return float(J)


def extremum_quality_multiplier(neigh: Dict[str,float]) -> float:
    """Return Q in [0,1] that penalizes poor fits, non-stationary slopes, weak gaps, or mixed-sign Hessian."""
    th_any: Any = CONFIG.get("neighborhood", {})
    thd: Dict[str, Any] = th_any if isinstance(th_any, dict) else {}
    obj_any: Any = CONFIG.get("objective", {})
    obj: Dict[str, Any] = obj_any if isinstance(obj_any, dict) else {}

    # Fit quality: clamp R2_min to [0,1]
    R2 = max(0.0, min(1.0, float(neigh.get("R2_min", 0.0))))
    q_R2 = R2**2  # more discriminative near 1

    # Gradient at center should be ~0
    grad_tol = float(thd.get("grad_tol", 3e-2))
    g = abs(float(neigh.get("grad_max_abs", 1.0)))
    q_grad = 1.0 / (1.0 + (g / max(grad_tol, 1e-6))**2)

    # Neighborhood gap
    min_gap_tol = float(thd.get("min_gap_tol", 0.005))
    gapn = float(neigh.get("min_gap_neigh", 0.0))
    q_gap = 1.0 if gapn >= min_gap_tol else (gapn / max(min_gap_tol, 1e-6))

    # Hessian eigen signs & curvature magnitude
    min_abs_curv = float(thd.get("min_abs_curv", 5e-4))
    lam_min = float(neigh.get("lam_min", 0.0)); lam_max = float(neigh.get("lam_max", 0.0))
    same_sign = (lam_min*lam_max) > 0
    q_sign = 1.0 if same_sign else 0.4
    curv_mag = max(abs(lam_min), abs(lam_max))
    q_curv = 1.0 if curv_mag >= min_abs_curv else (curv_mag / max(min_abs_curv, 1e-9))

    # Combine with weights
    wr2  = float(obj.get("weight_R2", 0.6))
    wgr  = float(obj.get("weight_grad", 0.6))
    wgap = float(obj.get("weight_gap_neigh", 1.0))
    wsame = float(obj.get("weight_eigs_same", 0.5))
    Q = (q_R2**wr2) * (q_grad**wgr) * (q_gap**wgap) * (q_sign**wsame) * (q_curv**0.5)

    # Enforce preferred extremum type if requested
    prefer = thd.get("favor_min_or_max", None)
    if prefer is not None and not bool(int(neigh.get("ok_minmax_sign", 0))):
        Q *= 0.4
    return float(max(0.0, min(1.0, Q)))

# ================================== Reciprocal viz & band diagrams ===============================

def reciprocal_rectangle_area(L_py) -> float:
    (g1, g2) = L_py.reciprocal_vectors()
    g1 = np.array(g1); g2 = np.array(g2)
    return abs(g1[0]*g2[1] - g1[1]*g2[0])

def plot_reciprocal_registry_square(theta_deg: float, outfile: str):
    try:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.axhline(0, lw=0.5, c="0.7"); ax.axvline(0, lw=0.5, c="0.7")
        if ML_AVAILABLE:
            L = ml.create_square_lattice(1.0)  # type: ignore[attr-defined]
            mo = ml.py_twisted_bilayer(L, math.radians(theta_deg))  # type: ignore[attr-defined]
            L1_py, L2_py = mo.lattice_1(), mo.lattice_2()
            side = math.sqrt(reciprocal_rectangle_area(L1_py))
            ax.add_patch(Rectangle((-0.5*side,-0.5*side), side, side, fill=False, lw=1.2))
            ax.scatter([0.0],[0.0], s=30, c="k", label="Γ")
            ax.set_title(f"Reciprocal registry (θ={theta_deg:.2f}°)")
        else:
            ax.add_patch(Rectangle((-math.pi,-math.pi), 2*math.pi, 2*math.pi, fill=False, lw=1.2))
            ax.scatter([0.0],[0.0], s=30, c="k", label="Γ")
        ax.set_aspect("equal","box"); ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); fig.savefig(outfile, dpi=160); plt.close(fig)
    except Exception as e:
        log.warning("Reciprocal registry plot failed: %s", e)

def kvec_from_label_square(label: str) -> mp.Vector3:
    return k0_from_label_square(label)

def build_kpath_square(labels: List[str], points_per_segment: int) -> Tuple[List[mp.Vector3], Dict[str, int]]:
    nodes = [kvec_from_label_square(lbl) for lbl in labels]
    kpts: List[mp.Vector3] = []
    idx_map: Dict[str, int] = {}
    idx = 0
    for i in range(len(nodes) - 1):
        seg = mp.interpolate(points_per_segment, [nodes[i], nodes[i+1]])
        if i > 0: seg = seg[1:]  # avoid duplicate node
        if i == 0: idx_map[labels[i]] = 0
        kpts.extend(seg)
        idx += len(seg)
        idx_map[labels[i+1]] = idx - 1
    return kpts, idx_map

def compute_band_diagram(a1: np.ndarray, a2: np.ndarray, radius: float, eps_bg: float, pol: str,
                         num_bands: Optional[int], kpts: List[mp.Vector3]) -> np.ndarray:  # type: ignore[name-defined]
    mpb_conf = CONFIG.get("mpb", {}) if isinstance(CONFIG.get("mpb", {}), dict) else {}
    nb = int(num_bands if num_bands is not None else int(mpb_conf.get("num_bands_report", 10)))
    lat, geom, mat = monolayer_geometry(a1, a2, radius, eps_bg)
    sup = bool(mpb_conf.get("suppress_output", True))
    res = int(mpb_conf.get("resolution_report", 48))
    with mp_quiet(sup):
        ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mat,  # type: ignore[attr-defined]
                            k_points=kpts, resolution=res,
                            num_bands=nb, dimensions=2)
        if pol.strip().lower() == "tm":
            ms.run_tm()
        else:
            ms.run_te()
        freqs = np.array(ms.all_freqs, float)
    return freqs

def plot_band_diagram_report(out_png: str, labels: List[str], kidx: Dict[str,int],
                             freqs: np.ndarray, highlight_band: int, highlight_label: str,
                             cand: Candidate, metrics: Dict[str,float]):
    try:
        Nk, Nb = freqs.shape
        x = np.arange(Nk)
        fig = plt.figure(figsize=(9.5, 4.8))
        gs = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[3.3, 1.0])
        ax = fig.add_subplot(gs[0,0])
        axr = fig.add_subplot(gs[0,1])

        for b in range(Nb):
            ax.plot(x, freqs[:, b], lw=0.8, alpha=0.9, color="0.15")
        ticks = []; ticklabels = []
        for lbl in labels:
            xi = kidx[lbl]
            ax.axvline(xi, color="0.8", lw=0.8)
            ticks.append(xi); ticklabels.append(lbl)
        if 0 <= highlight_band < Nb and highlight_label in kidx:
            xi = kidx[highlight_label]
            yi = freqs[xi, highlight_band]
            ax.scatter([xi], [yi], s=60, c="crimson", zorder=10, label=f"{highlight_label}, b{highlight_band}")
            ax.legend(loc="best", fontsize=8)
        ax.set_xlim(0, Nk-1)
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
        ax.set_xlabel("k-path")
        ax.set_ylabel("Frequency (a/λ)")
        ax.set_title(f"Band diagram — pol={cand.polarization}, r={cand.hole_radius_a:.3f}, ε={cand.eps_bg:.2f}")

        # Right panel with richer stats
        lines = [
            f"band: {cand.band_index} @ {cand.k_label} ({metrics.get('extremum_type','?')})",
            f"ω0: {metrics.get('omega0', float('nan')):.5f}",
            f"α(central): {metrics.get('alpha', float('nan')):.5f}",
            f"κx: {metrics.get('kappa_x', float('nan')):.5f}",
            f"κy: {metrics.get('kappa_y', float('nan')):.5f}",
            f"κdiag: {metrics.get('kappa_diag', float('nan')):.5f}",
            f"H: [[{metrics.get('Hxx',float('nan')):.4f},{metrics.get('Hxy',float('nan')):.4f}],",
            f"    [{metrics.get('Hxy',float('nan')):.4f},{metrics.get('Hyy',float('nan')):.4f}]]",
            f"λmin/max: {metrics.get('lam_min', float('nan')):.5f} / {metrics.get('lam_max', float('nan')):.5f}",
            f"R²_min: {metrics.get('R2_min', float('nan')):.5f}",
            f"|b|max: {metrics.get('grad_max_abs', float('nan')):.5e}",
            f"min gap(neigh): {metrics.get('min_gap_neigh', float('nan')):.5f}",
            f"|ΔV|: {metrics.get('contrast_AA_AB', float('nan')):.5f}",
            f"J1: {metrics.get('J1', float('nan')):.5f}",
            f"Q:  {metrics.get('Q', float('nan')):.5f}",
            f"J2: {metrics.get('J2', float('nan')):.5f}",
        ]
        axr.axis("off")
        axr.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
        fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
    except Exception as e:
        log.warning("Band report plot failed for %s: %s", out_png, e)

# ================================== Grid search & optimization ==================================

def grid_search() -> List[Tuple[Candidate, Dict[str,float]]]:
    a1 = np.array(CONFIG["a1"], float); a2 = np.array(CONFIG["a2"], float)
    Gconf_any = CONFIG.get("grid", {})
    Gconf: Dict[str, Any] = Gconf_any if isinstance(Gconf_any, dict) else {}
    r_list = list(Gconf.get("hole_radius_list", []))
    eps_list = list(Gconf.get("eps_bg_list", []))
    pols = list(Gconf.get("polarizations", []))
    bands = list(Gconf.get("band_indices", []))
    klabels = list(Gconf.get("k_labels", []))
    results: List[Tuple[Candidate, Dict[str,float]]] = []
    total = len(r_list) * len(eps_list) * len(pols) * len(bands) * len(klabels)
    log.info("Grid across %d points ...", total)
    progress = None; task = None
    if RICH:
        progress = Progress(TextColumn("[bold blue]Quick scan"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), transient=True, console=console)
        task = progress.add_task("grid_quick", total=total); progress.start()

    quick_pool: List[Tuple[Candidate, Dict[str,float]]] = []
    failures = 0

    for pol in pols:
        for bi in bands:
            for k_label in klabels:
                for r in r_list:
                    for eps in eps_list:
                        try:
                            met = quick_evaluate_candidate(a1, a2, pol, bi, k_label, r, eps)
                            cand = Candidate(CONFIG["lattice_type"], a1[0], a1[1], a2[0], a2[1], r, eps, pol, bi, k_label, 1.10)
                            quick_pool.append((cand, met))
                        except Exception as e:
                            failures += 1
                            log.debug("Quick eval failed r=%.3f eps=%.2f pol=%s band=%d k=%s: %s", r, eps, pol, bi, k_label, e)
                        if progress: progress.advance(task)
    if progress: progress.stop()

    log.info("Quick scan done: %d ok, %d failures. Selecting top for full eval...", len(quick_pool), failures)

    # Select top-M by J1
    quick_sorted = sorted(quick_pool, key=lambda cm: cm[1]["J1"], reverse=True)
    M = max(int(Gconf.get("min_full", 32)), int(Gconf.get("top_fraction_for_full", 0.15) * len(quick_sorted)))
    M = min(M, int(Gconf.get("max_full", 128)), len(quick_sorted))
    top_for_full = quick_sorted[:M]

    # Full evaluation on top-M
    progress = None; task = None
    if RICH:
        progress = Progress(TextColumn("[bold green]Full eval"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), transient=True, console=console)
        task = progress.add_task("grid_full", total=M); progress.start()

    for cand, _met0 in top_for_full:
        try:
            met = evaluate_candidate(a1, a2, cand.polarization, cand.band_index, cand.k_label, cand.hole_radius_a, cand.eps_bg)
            results.append((cand, met))
        except Exception as e:
            log.debug("Full eval failed for %s: %s", cand, e)
        if progress: progress.advance(task)
    if progress: progress.stop()

    # Optionally also keep remaining quick candidates (with downweighted Q) to ensure we have enough seeds
    remaining = quick_sorted[M:]
    results.extend(remaining)

    return results

def deduplicate(cands: List[Tuple[Candidate, Dict[str,float]]]) -> List[Tuple[Candidate, Dict[str,float]]]:
    sep_r = CONFIG["seeds"]["min_param_separation"]["hole_radius"]
    sep_eps = CONFIG["seeds"]["min_param_separation"]["eps_bg"]
    kept: List[Tuple[Candidate, Dict[str,float]]] = []
    for cand, met in sorted(cands, key=lambda cm: cm[1]["J2"], reverse=True):
        def is_close(x, y, tol): return abs(x-y) <= tol
        if any(is_close(cand.hole_radius_a, c.hole_radius_a, sep_r) and
               is_close(cand.eps_bg, c.eps_bg, sep_eps) and
               cand.polarization == c.polarization and
               cand.band_index == c.band_index and
               cand.k_label == c.k_label
               for (c,_m) in kept):
            continue
        kept.append((cand, met))
    return kept

def optimize_continuous(grid_seed_pool: Optional[List[Tuple[Candidate, Dict[str,float]]]] = None) -> Optional[Tuple[Candidate, Dict[str,float]]]:
    if not SCIPY_AVAILABLE:
        log.warning("SciPy not available; skipping continuous optimization.")
        return None
    a1 = np.array(CONFIG["a1"], float); a2 = np.array(CONFIG["a2"], float)
    O_any: Any = CONFIG.get("optimization", {})
    O: Dict[str, Any] = O_any if isinstance(O_any, dict) else {}
    bounds_cfg = O.get("bounds", ((0.08, 0.49), (2.5, 10.0)))
    bounds = Bounds([bounds_cfg[0][0], bounds_cfg[1][0]], [bounds_cfg[0][1], bounds_cfg[1][1]])

    def x0_for(pol: str, bi: int, k_label: str) -> np.ndarray:
        if grid_seed_pool:
            filtered = [(c, m) for (c, m) in grid_seed_pool if c.polarization==pol and c.band_index==bi and c.k_label==k_label]
            if filtered:
                (c_best, _m_best) = max(filtered, key=lambda cm: cm[1]["J2"])
                return np.array([c_best.hole_radius_a, c_best.eps_bg], float)
        x0_cfg = O.get("initial_guess", (0.38, 4.0))
        return np.array(list(x0_cfg), float)

    best_pair: Optional[Tuple[Candidate, Dict[str,float]]] = None
    best_J = -math.inf
    pols = list(O.get("polarizations", []))
    bands = [int(b) for b in O.get("band_indices", [])]
    klabels = list(O.get("k_labels", []))
    combos = [(pol, bi, k_label) for pol in pols for bi in bands for k_label in klabels]
    log.info("Continuous optimization over %d (pol,band,k) combos ...", len(combos))

    # Progress bar for optimization combos (transient, like grid search)
    progress = None; task = None
    if RICH and len(combos) > 0:
        progress = Progress(TextColumn("[bold magenta]Optimize"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), transient=True, console=console)
        task = progress.add_task("optimize", total=len(combos)); progress.start()

    mpb_conf = CONFIG.get("mpb", {}) if isinstance(CONFIG.get("mpb", {}), dict) else {}
    sup = bool(mpb_conf.get("suppress_output", True))
    maxiter = int(O.get("maxiter", 50))

    for (pol, bi, k_label) in combos:
        def neg_objective(x):
            r, eps = float(x[0]), float(x[1])
            try:
                metrics = evaluate_candidate(a1, a2, pol, int(bi), k_label, r, eps)
                return -metrics["J2"]
            except Exception:
                return 1e3

        x0 = x0_for(pol, int(bi), k_label)
        with mp_quiet(sup):
            res = minimize(neg_objective, x0, method="Powell", bounds=bounds,
                           options=dict(maxiter=maxiter, xtol=1e-3, ftol=1e-3, disp=False))
        r_opt, eps_opt = float(res.x[0]), float(res.x[1])
        try:
            metrics = evaluate_candidate(a1, a2, pol, int(bi), k_label, r_opt, eps_opt)
        except Exception as e:
            log.debug("Eval failed after opt for (pol=%s, band=%d, k=%s): %s", pol, bi, k_label, e)
            if progress is not None and task is not None:
                progress.advance(task)
            continue
        J = metrics["J2"]
        if J > best_J:
            best_J = J
            cand = Candidate(str(CONFIG.get("lattice_type", "square")), a1[0], a1[1], a2[0], a2[1], r_opt, eps_opt, pol, int(bi), k_label, 1.10)
            best_pair = (cand, metrics)
        if progress is not None and task is not None:
            progress.advance(task)

    if progress is not None:
        progress.stop()

    if best_pair is not None:
        log.info("Optimization best: pol=%s band=%d k=%s r=%.4f eps=%.4f J2=%.6f",
                 best_pair[0].polarization, best_pair[0].band_index, best_pair[0].k_label,
                 best_pair[0].hole_radius_a, best_pair[0].eps_bg, best_pair[1]["J2"])
    else:
        log.warning("Optimization did not yield a valid candidate.")
    return best_pair

def export_seeds(cands: List[Tuple[Candidate, Dict[str,float]]], path_csv: str, K: int):
    import csv
    columns = ["lattice_type","a1x","a1y","a2x","a2y","hole_radius_a","eps_bg","polarization","band_index","k_label","twist_deg",
               "omega0","alpha","gap_below","gap_above","omega_AA","omega_AB","omega_BA","contrast_AA_AB",
               "kappa_x","kappa_y","kappa_diag","Hxx","Hyy","Hxy","lam_min","lam_max",
               "R2_min","grad_max_abs","min_gap_neigh","ok_minmax_sign","extremum_type",
               "J1","Q","J2"]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for (cand, met) in sorted(cands, key=lambda cm: cm[1]["J2"], reverse=True)[:K]:
            row = {
                "lattice_type":cand.lattice_type, "a1x":cand.a1x, "a1y":cand.a1y, "a2x":cand.a2x, "a2y":cand.a2y,
                "hole_radius_a":cand.hole_radius_a, "eps_bg":cand.eps_bg, "polarization":cand.polarization,
                "band_index":cand.band_index, "k_label":cand.k_label, "twist_deg":cand.twist_deg,
                **{k: met.get(k, float("nan")) for k in columns if k not in {
                    "lattice_type","a1x","a1y","a2x","a2y","hole_radius_a","eps_bg","polarization","band_index","k_label","twist_deg"
                }}
            }
            w.writerow(row)

# ================================== Reporting & CSV validation ==================================

def generate_band_diagram_reports(selected: List[Tuple[Candidate, Dict[str,float]]], labels: Optional[List[str]] = None, points_per_segment: Optional[int] = None):
    if not CONFIG.get("reports", {}).get("enable", True): return
    if not selected: return
    labels = labels or CONFIG["reports"]["kpath_labels"]
    pts_per = int(points_per_segment or CONFIG["reports"]["points_per_segment"])
    a1 = np.array(CONFIG["a1"], float); a2 = np.array(CONFIG["a2"], float)
    kpts, kidx = build_kpath_square(labels, pts_per)
    log.info("Generating band diagram reports for %d candidates...", len(selected))
    for i, (cand, met) in enumerate(selected, start=1):
        try:
            freqs = compute_band_diagram(a1, a2, cand.hole_radius_a, cand.eps_bg, cand.polarization,
                                         CONFIG["mpb"]["num_bands_report"], kpts)
            out_png = os.path.join(LOG_DIR, f"banddiag_{i:03d}_{cand.polarization}_b{cand.band_index}_k{cand.k_label}.png")
            plot_band_diagram_report(out_png, labels, kidx, freqs, cand.band_index, cand.k_label, cand, met)
        except Exception as e:
            log.warning("Band-diagram report failed for candidate #%d: %s", i, e)

def validate_csv(in_csv: str, out_csv: str, K: Optional[int] = None, reports_dir: Optional[str] = None):
    """Re-evaluate seeds from CSV with neighborhood tests, write enriched CSV and band diagrams."""
    import csv
    if not MEEP_AVAILABLE:
        log.error("Meep/MPB not available – cannot validate CSV.")
        return
    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    a1 = np.array(CONFIG["a1"], float); a2 = np.array(CONFIG["a2"], float)
    pool: List[Tuple[Candidate, Dict[str,float]]] = []
    for r in rows:
        cand = Candidate(
            lattice_type=r["lattice_type"],
            a1x=float(r["a1x"]), a1y=float(r["a1y"]),
            a2x=float(r["a2x"]), a2y=float(r["a2y"]),
            hole_radius_a=float(r["hole_radius_a"]),
            eps_bg=float(r["eps_bg"]),
            polarization=r["polarization"],
            band_index=int(r["band_index"]),
            k_label=r["k_label"],
            twist_deg=float(r.get("twist_deg", 1.10)),
        )
        # Re-evaluate with full neighborhood
        met = evaluate_candidate(a1, a2, cand.polarization, cand.band_index, cand.k_label, cand.hole_radius_a, cand.eps_bg)
        pool.append((cand, met))

    pool = deduplicate(pool)
    K = K or CONFIG["seeds"]["K"]
    export_seeds(pool, out_csv, K)
    log.info("Validated and wrote top-%d seeds to %s", K, out_csv)

    # Optional band reports
    if reports_dir is not None:
        os.makedirs(reports_dir, exist_ok=True)
        sel = sorted(pool, key=lambda cm: cm[1]["J2"], reverse=True)[:K]
        # Use a slightly denser path to visualize curvature near k0
        generate_band_diagram_reports(sel, labels=CONFIG["reports"]["kpath_labels"], points_per_segment=64)

# ================================== CLI ==========================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Stage 1 — Seed generation with robust extremum tests")
    ap.add_argument("--out", default="seeds_stage1.csv", help="Output CSV for seeds (default: seeds_stage1.csv).")
    ap.add_argument("--K", type=int, default=CONFIG["seeds"]["K"], help="Number of seeds to export.")
    ap.add_argument("--verbose", action="store_true", help="Show MPB output (default suppressed).")
    ap.add_argument("--reports-from-csv", metavar="CSV", help="Generate band reports from existing CSV (no optimization).")
    ap.add_argument("--reports-dir", help="Output directory for band reports (default: stage1_outputs).")
    ap.add_argument("--max-reports", type=int, help="Maximum number of reports to generate from CSV.")
    ap.add_argument("--validate-csv", metavar="CSV", help="Validate an existing seeds CSV using neighborhood tests.")
    ap.add_argument("--out-validated", default="seeds_stage1_validated.csv", help="Path for validated CSV output.")
    return ap.parse_args()

def main():
    args = parse_args()

    # Band reports only?
    if args.reports_from_csv:
        if not MEEP_AVAILABLE:
            log.error("Meep/MPB not available – cannot generate band diagrams."); sys.exit(2)
        outdir = args.reports_dir or LOG_DIR
        os.makedirs(outdir, exist_ok=True)
        # Reuse older simpler path: parse CSV and plot
        import csv
        rows = []
        with open(args.reports_from_csv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd: rows.append(r)
        labels = CONFIG["reports"]["kpath_labels"]
        kpts, kidx = build_kpath_square(labels, CONFIG["reports"]["points_per_segment"])
        for i, r in enumerate(rows, start=1):
            try:
                cand = Candidate(r["lattice_type"], float(r["a1x"]), float(r["a1y"]), float(r["a2x"]), float(r["a2y"]),
                                 float(r["hole_radius_a"]), float(r["eps_bg"]), r["polarization"], int(r["band_index"]), r["k_label"],
                                 float(r.get("twist_deg", 1.10)))
                freqs = compute_band_diagram(np.array(CONFIG["a1"],float), np.array(CONFIG["a2"],float),
                                             cand.hole_radius_a, cand.eps_bg, cand.polarization,
                                             CONFIG["mpb"]["num_bands_report"], kpts)
                met = {k: (float(r[k]) if k in r and r[k]!="" else float("nan")) for k in ("omega0","alpha","gap_below","gap_above","omega_AA","omega_AB","omega_BA","contrast_AA_AB","J1","Q","J2","kappa_x","kappa_y","kappa_diag","Hxx","Hyy","Hxy","lam_min","lam_max","R2_min","grad_max_abs","min_gap_neigh")}
                out_png = os.path.join(outdir, f"banddiag_csv_{i:03d}_{cand.polarization}_b{cand.band_index}_k{cand.k_label}.png")
                plot_band_diagram_report(out_png, labels, kidx, freqs, cand.band_index, cand.k_label, cand, met)
                log.info("Saved band-diagram report: %s", out_png)
                if args.max_reports and i >= args.max_reports: break
            except Exception as e:
                log.warning("Band-diagram report failed for row %d: %s", i, e)
        sys.exit(0)

    # Validation mode
    if args.validate_csv:
        if not MEEP_AVAILABLE:
            log.error("Meep/MPB not available – cannot validate CSV."); sys.exit(2)
        validate_csv(args.validate_csv, args.out_validated, K=args.K, reports_dir=args.reports_dir)
        sys.exit(0)

    # Full generation pipeline
    if not MEEP_AVAILABLE:
        log.error("Meep/MPB not available – cannot run Stage 1 computations."); sys.exit(2)
    CONFIG["mpb"]["suppress_output"] = not args.verbose

    if CONFIG["visuals"]["plot_reciprocal_sample"]:
        try:
            plot_reciprocal_registry_square(1.10, os.path.join(LOG_DIR, "reciprocal_registry_1p10deg.png"))
        except Exception as e:
            log.warning("Reciprocal registry plotting skipped: %s", e)

    # 1) Grid search
    t0 = time.time()
    grid_res = grid_search()
    grid_res = deduplicate(grid_res)
    t1 = time.time()
    log.info("Grid search produced %d unique candidates in %.1fs", len(grid_res), t1-t0)

    # 2) Continuous optimization
    opt_pair = None
    if CONFIG["seeds"]["include_optimized"] and CONFIG["optimization"]["enable"]:
        try:
            opt_pair = optimize_continuous(grid_res)
        except Exception as e:
            log.warning("Optimization failed: %s", e)

    # Pool & export
    pool = list(grid_res)
    if opt_pair is not None:
        pool.append(opt_pair)
        pool = deduplicate(pool)
        log.info("After adding optimized seed: %d unique candidates", len(pool))

    export_seeds(pool, args.out, args.K)
    log.info("Wrote top-%d seeds to %s", args.K, args.out)
    log.info("Full log at %s", LOG_FILE)

    # 3) Band-diagram reports for top seeds
    try:
        selected = sorted(pool, key=lambda cm: cm[1]["J2"], reverse=True)[:args.K]
        generate_band_diagram_reports(selected, labels=CONFIG["reports"]["kpath_labels"], points_per_segment=64)
    except Exception as e:
        log.warning("Band report generation skipped: %s", e)

if __name__ == "__main__":
    main()
