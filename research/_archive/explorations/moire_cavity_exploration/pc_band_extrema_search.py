"""
Photonic crystal band-extrema search & parameter optimization (enhanced)
------------------------------------------------------------------------
Key upgrades:
- FIX: Robust handling of polarization="both" (candidates + plots now always emitted).
- FIX: No reliance on globals inside detection; r/eps passed explicitly.
- Continuous shape optimization:
    * rectangular: aspect ratio rho = b/a (excludes square line via margin penalty).
    * oblique: rho=b/a and angle gamma (deg), excluding γ≈60°,90°,120° with margin penalty.
- Oblique lattice gets a 2D k-mesh search over the first Brillouin Zone using `moire_lattice_py`.
  Includes extra plots (BZ heatmap + extrema markers).
- Output folder gets suffix "_<latticetype>".

Notes
-----
* Requires Meep/MPB. For oblique BZ mesh, requires `moire_lattice_py` Python bindings.
* Tunable parameters are grouped in the CONFIG section below.
"""

import os
import csv
import math
import json
import time
import warnings
from dataclasses import dataclass, asdict, replace
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

# Optional progress (rich)
try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )
    from rich.text import Text
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False

# Optional SciPy optimization
try:
    from scipy.optimize import minimize as _minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# MPB / Meep (lazy import handles environments without Meep)
mp: Any = None
mpb: Any = None
try:
    import meep as _mp
    from meep import mpb as _mpb
    mp = _mp
    mpb = _mpb
    MEEP_AVAILABLE = True
except Exception:
    MEEP_AVAILABLE = False

# Lattice & BZ (user's Rust/Py bindings)
_ml_ok = True
try:
    import moire_lattice_py as ml  # user's binding for lattice & BZ
except Exception:
    _ml_ok = False

# ---------------------------
# CONFIG
# ---------------------------

@dataclass
class SearchConfig:
    # General
    output_dir: str = "pc_extrema_results"
    csv_name: str = "candidates.csv"
    log_name: str = "run_log.json"
    seed: int = 1

    # Lattice (base)
    lattice_type: str = "oblique"       # "triangular" | "square" | "rectangular" | "oblique"
    a: float = 1.0                          # lattice constant (arbitrary units)

    # Rectangular shape defaults (continuous param rho=b/a)
    rect_rho_min: float = 1.05
    rect_rho_max: float = 3.0
    rect_rho_exclude_margin: float = 0.02   # soft-penalize rho near 1 (square)

    # Oblique shape defaults (continuous params rho=b/a, gamma in degrees)
    obl_rho_min: float = 1.05
    obl_rho_max: float = 3.0
    obl_gamma_min: float = 60.5            # exclude 60° boundary (triangular)
    obl_gamma_max: float = 119.5           # exclude 120° boundary (triangular)
    obl_gamma_exclude: Tuple[float, ...] = (90.0,)  # avoid rectangular line
    obl_gamma_margin: float = 0.4          # soft margin around excluded angles
    oblique_mesh: bool = True              # enable 2D BZ mesh for oblique
    oblique_use_mesh_in_objective: bool = True  # optimize using mesh-based detector
    oblique_mesh_n: int = 25               # coarse grid per axis for objective/sweep
    oblique_mesh_n_final: int = 35         # finer grid for final confirm plot
    oblique_window: int = 1                # neighborhood radius (in grid steps) for 2D quad fit

    # MPB
    resolution: int = 40                    # grid resolution (pixels / a)
    num_bands: int = 8
    polarization: str = "tm"                # "tm" | "te" | "both"
    k_points_per_segment: int = 18

    # Geometry (air hole in dielectric background)
    r_min: float = 0.08                     # hole radius (in units of a)
    r_max: float = 0.45
    eps_bg_min: float = 2.0                 # background epsilon
    eps_bg_max: float = 16.0

    # Sweep (grid) across r/eps only (shape handled by optimizer to keep runs feasible)
    do_grid_sweep: bool = True
    r_grid: Optional[List[float]] = None    # if None: auto linspace
    eps_grid: Optional[List[float]] = None
    r_grid_points: int = 5
    eps_grid_points: int = 5

    # Optimizer
    do_optimizer: bool = True               # enable local optimization (Nelder-Mead)
    # x0 meaning depends on lattice:
    #  triangular/square: (r, eps)
    #  rectangular:       (r, eps, rho)
    #  oblique:           (r, eps, rho, gamma_deg)
    opt_initial: Tuple[float, ...] = (0.25, 12.0)  
    opt_maxiter: int = 60

    # Candidate detection (1D path)
    high_symmetry_points_only: bool = True
    target_extremum: str = "either"         # "min" | "max" | "either"
    window_halfwidth: int = 3               # points on each side for local quadratic fit
    curvature_min: float = 0.05             # |a| lower bound for 1D quad fit
    curvature_max: float = 15.0             # |a| upper bound (filter noisy spikes)
    linear_term_tol: float = 0.08           # |b| tolerance
    residual_tol: float = 2e-3              # MSE tolerance of quadratic fit in the local window
    min_gap_to_neighbors: float = 0.02      # minimum frequency separation to adjacent bands

    # Candidate detection (2D mesh, oblique)
    mesh_grad_tol: float = 5e-3             # ||grad|| tolerance near extremum
    mesh_curv_min: float = 0.02             # min |eig(H)|
    mesh_curv_max: float = 40.0             # max |eig(H)|
    mesh_residual_tol: float = 4e-3         # LS residual tolerance

    # Candidate scoring (for optimizer)
    w_curv: float = 1.0
    w_gap: float = 1.0
    w_lin: float = 0.5
    w_res: float = 0.5

    # Plotting
    save_plots: bool = True
    dpi: int = 150


# ---------------------------
# Utility
# ---------------------------

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def clamp(v: float, vmin: float, vmax: float) -> float:
    return float(max(vmin, min(vmax, v)))


def soft_barrier(dist: float, width: float) -> float:
    """Smooth penalty ~0 far away, grows as 1/(dist^2) within margin 'width'."""
    if dist <= 0:
        # inside forbidden
        return 1e2
    if dist > width:
        return 0.0
    # quadratic blow-up near zero
    return (width / dist) ** 2


# ---------------------------
# LATTICE & K-PATH HELPERS
# ---------------------------

def lattice_basis_from_params(cfg: SearchConfig,
                              shape: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Return (a1, a2, mp.Lattice) using lattice_type + optional 'shape' params.
    shape keys:
      rectangular: {'rho': ...}
      oblique:     {'rho': ..., 'gamma_deg': ...}
    """
    lt = cfg.lattice_type.lower()
    if lt == "triangular":
        a1 = np.array([cfg.a, 0.0])
        a2 = np.array([0.5 * cfg.a, (math.sqrt(3)/2) * cfg.a])
    elif lt == "square":
        a1 = np.array([cfg.a, 0.0])
        a2 = np.array([0.0, cfg.a])
    elif lt == "rectangular":
        rho = 1.5 if shape is None else float(shape.get("rho", 1.5))
        a1 = np.array([cfg.a, 0.0])
        a2 = np.array([0.0, rho * cfg.a])
    elif lt == "oblique":
        rho = 1.2 if shape is None else float(shape.get("rho", 1.2))
        gamma_deg = 100.0 if shape is None else float(shape.get("gamma_deg", 100.0))
        gamma = math.radians(gamma_deg)
        a1 = np.array([cfg.a, 0.0])
        a2 = rho * cfg.a * np.array([math.cos(gamma), math.sin(gamma)])
    else:
        raise ValueError(f"Unknown lattice_type: {cfg.lattice_type}")

    lat = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(float(a1[0]), float(a1[1]), 0),
                     basis2=mp.Vector3(float(a2[0]), float(a2[1]), 0))
    return a1, a2, lat


def high_symmetry_path(cfg: SearchConfig) -> Tuple[List[Any], List[str], List[int]]:
    """
    Build a canonical k-path and labels in *reciprocal-lattice fractional coordinates*.
    Returns: (k_points, labels, label_indices)
    """
    lt = cfg.lattice_type.lower()
    pts: List[Any] = []
    labels: List[str] = []
    label_indices: List[int] = []

    def add_segment(p_from, p_to, lbl_from, lbl_to):
        nonlocal pts, labels, label_indices
        seg = mp.interpolate(cfg.k_points_per_segment, [p_from, p_to])
        if len(pts) > 0:
            seg = seg[1:]
        start_idx = len(pts)
        pts += seg
        if lbl_from is not None:
            if len(labels) == 0:
                labels.append(lbl_from)
                label_indices.append(start_idx)
        labels.append(lbl_to)
        label_indices.append(len(pts)-1)

    if lt == "triangular":
        G = mp.Vector3(0, 0); M = mp.Vector3(0.5, 0.0); K = mp.Vector3(1/3, 1/3)
        add_segment(G, M, "Γ", "M")
        add_segment(M, K, "M", "K")
        add_segment(K, G, "K", "Γ")
    elif lt in ("square", "rectangular"):
        G = mp.Vector3(0, 0); X = mp.Vector3(0.5, 0.0); M = mp.Vector3(0.5, 0.5)
        add_segment(G, X, "Γ", "X")
        add_segment(X, M, "X", "M")
        add_segment(M, G, "M", "Γ")
    elif lt == "oblique":
        # Oblique has no canonical HS path; provide a basic Γ-edge-Γ for quick 1D scans.
        # The thorough 2D BZ mesh is handled elsewhere.
        G = mp.Vector3(0, 0); B1 = mp.Vector3(0.5, 0.0); B2 = mp.Vector3(0.0, 0.5)
        add_segment(G, B1, "Γ", "b1/2")
        add_segment(B1, B2, "b1/2", "b2/2")
        add_segment(B2, G, "b2/2", "Γ")
    else:
        raise ValueError(f"Unknown lattice_type for HS path: {cfg.lattice_type}")

    return pts, labels, label_indices


def path_arclength(kpts: List[Any]) -> np.ndarray:
    ks = np.array([[kp.x, kp.y] for kp in kpts], dtype=float)
    if len(ks) < 2:
        return np.zeros(len(ks), dtype=float)
    ds = np.linalg.norm(np.diff(ks, axis=0), axis=1, ord=2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


# ---------------------------
# MPB RUNNERS
# ---------------------------

def run_mpb_path(cfg: SearchConfig,
                 r: float,
                 eps_bg: float,
                 shape: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Run an MPB band calculation along a 1D k-path."""
    if not MEEP_AVAILABLE:
        raise RuntimeError("Meep / MPB not available.")

    a1, a2, lat = lattice_basis_from_params(cfg, shape)

    geom = [mp.Cylinder(radius=r * cfg.a, material=mp.air, center=mp.Vector3(0, 0, 0))]
    kpts, k_labels, k_label_inds = high_symmetry_path(cfg)

    ms_kwargs = dict(
        geometry_lattice=lat,
        geometry=geom,
        default_material=mp.Medium(epsilon=eps_bg),
        k_points=kpts,
        resolution=cfg.resolution,
        num_bands=cfg.num_bands,
        dimensions=2
    )
    ms = mpb.ModeSolver(**ms_kwargs)

    pol = cfg.polarization.lower()
    if pol == "tm":
        ms.run_tm(); freqs = np.array(ms.all_freqs)
        out = {"freqs": freqs}
    elif pol == "te":
        ms.run_te(); freqs = np.array(ms.all_freqs)
        out = {"freqs": freqs}
    elif pol == "both":
        ms.run_tm(); freqs_tm = np.array(ms.all_freqs)
        ms.reset_meep(); ms = mpb.ModeSolver(**ms_kwargs); ms.run_te(); freqs_te = np.array(ms.all_freqs)
        out = {"freqs_tm": freqs_tm, "freqs_te": freqs_te}
    else:
        raise ValueError("polarization must be 'tm', 'te' or 'both'")

    out.update({
        "s": path_arclength(kpts),
        "k_labels": k_labels,
        "k_label_inds": np.array(k_label_inds, dtype=int),
        "a1": a1, "a2": a2,
        "shape": shape or {},
        "mode": "path"
    })
    return out


def _bz_polygon_fractional(lattice: Any) -> np.ndarray:
    """Return BZ polygon in *fractional reciprocal coordinates* given ml.PyLattice2D lattice."""
    # ml returns reciprocal basis in 2π convention; columns are b1,b2,b3
    if hasattr(lattice, "inner"):
        # If it's wrapped, get the inner reciprocal basis
        if hasattr(lattice.inner, "reciprocal"):
            B = np.array(lattice.inner.reciprocal)
        else:
            # Try reciprocal_basis method
            basis = lattice.inner.reciprocal_basis()
            B = np.array([list(basis[0]), list(basis[1]), list(basis[2])]).T
    else:
        # Direct PyLattice2D object - use reciprocal_basis method
        basis = lattice.reciprocal_basis()
        B = np.array([list(basis[0]), list(basis[1]), list(basis[2])]).T
    
    B2 = B[:2, :2]  # 2D
    # Vertices in physical reciprocal coords (x,y,z); convert to fractional: B2^{-1} * kvec
    verts = []
    if hasattr(lattice, "inner"):
        # If it's wrapped, get the inner brillouin_zone
        if hasattr(lattice.inner, "brillouin_zone"):
            if hasattr(lattice.inner.brillouin_zone, "vertices"):
                bz_vertices = lattice.inner.brillouin_zone.vertices
            else:
                bz_vertices = lattice.inner.brillouin_zone.vertices()
        else:
            bz_vertices = lattice.inner.brillouin_zone().vertices()
    else:
        # Direct PyLattice2D object - use brillouin_zone method
        bz_vertices = lattice.brillouin_zone().vertices()
    
    for v in bz_vertices:
        kv = np.array([v[0], v[1]]) if isinstance(v, (list, tuple)) else np.array([v.x, v.y])
        frac = np.linalg.solve(B2, kv)
        verts.append(frac)
        kv = np.array([v[0], v[1]]) if isinstance(v, (list, tuple)) else np.array([v.x, v.y])
        frac = np.linalg.solve(B2, kv)
        verts.append(frac)
    return np.array(verts, dtype=float)


def _mesh_inside_polygon(poly_frac: np.ndarray, N: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return fractional k-points inside polygon and grid axes (for plotting)."""
    minx, miny = poly_frac.min(axis=0)
    maxx, maxy = poly_frac.max(axis=0)
    gx = np.linspace(minx, maxx, N)
    gy = np.linspace(miny, maxy, N)
    XX, YY = np.meshgrid(gx, gy, indexing="xy")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    mask = MplPath(poly_frac).contains_points(pts)
    pts_in = pts[mask]
    return pts_in, (gx, gy)


def run_mpb_mesh_oblique(cfg: SearchConfig,
                         r: float,
                         eps_bg: float,
                         rho: float,
                         gamma_deg: float,
                         N: int) -> Dict[str, Any]:
    """Run MPB on a 2D mesh of k-points inside the oblique BZ (fractional coords)."""
    if not (MEEP_AVAILABLE and _ml_ok):
        raise RuntimeError("Meep/MPB and moire_lattice_py are required for oblique mesh.")

    # Build lattice through ml for BZ & also create mp lattice
    lat_ml = ml.PyLattice2D("oblique", a=cfg.a, b=rho * cfg.a, angle=gamma_deg)
    poly_frac = _bz_polygon_fractional(lat_ml)  # fractional polygon in (b1,b2) basis

    a1, a2, lat = lattice_basis_from_params(cfg, {"rho": rho, "gamma_deg": gamma_deg})
    geom = [mp.Cylinder(radius=r * cfg.a, material=mp.air, center=mp.Vector3(0, 0, 0))]

    pts_frac, (gx, gy) = _mesh_inside_polygon(poly_frac, N)
    kpts = [mp.Vector3(float(px), float(py), 0) for (px, py) in pts_frac]

    ms_kwargs = dict(
        geometry_lattice=lat,
        geometry=geom,
        default_material=mp.Medium(epsilon=eps_bg),
        k_points=kpts,
        resolution=cfg.resolution,
        num_bands=cfg.num_bands,
        dimensions=2
    )
    ms = mpb.ModeSolver(**ms_kwargs)

    pol = cfg.polarization.lower()
    if pol == "tm":
        ms.run_tm(); freqs = np.array(ms.all_freqs)
        out = {"freqs": freqs}
    elif pol == "te":
        ms.run_te(); freqs = np.array(ms.all_freqs)
        out = {"freqs": freqs}
    elif pol == "both":
        ms.run_tm(); freqs_tm = np.array(ms.all_freqs)
        ms.reset_meep(); ms = mpb.ModeSolver(**ms_kwargs); ms.run_te(); freqs_te = np.array(ms.all_freqs)
        out = {"freqs_tm": freqs_tm, "freqs_te": freqs_te}
    else:
        raise ValueError("polarization must be 'tm', 'te' or 'both'")

    out.update({
        "k_frac": pts_frac,          # [nk,2]
        "grid_axes": (gx, gy),       # for heatmap convenience
        "a1": a1, "a2": a2,
        "rho": rho, "gamma_deg": gamma_deg,
        "bz_poly_frac": poly_frac,
        "mode": "mesh"
    })
    return out


# ---------------------------
# EXTREMA DETECTORS
# ---------------------------

@dataclass
class Candidate:
    lattice_type: str
    a1x: float
    a1y: float
    a2x: float
    a2y: float
    hole_radius: float
    eps_bg: float
    polarization: str
    band_index: int
    k_index: int
    k_label: str
    s_value: float
    freq: float
    extremum_type: str
    curvature_a: float
    linear_b: float
    quad_residual: float
    gap_below: float
    gap_above: float
    plot_file: str
    # Extra for mesh
    kx_frac: float = float("nan")
    ky_frac: float = float("nan")
    mode: str = "path"  # 'path' or 'mesh'


def _fit_local_quadratic_1d(s: np.ndarray, f: np.ndarray, i0: int, halfw: int) -> Tuple[float, float, float, float]:
    lo = max(0, i0 - halfw)
    hi = min(len(s), i0 + halfw + 1)
    ss = s[lo:hi]
    ff = f[lo:hi]
    s0 = s[i0]
    x = ss - s0
    coeffs = np.polyfit(x, ff, deg=2)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    fit = a * x**2 + b * x + c
    mse = float(np.mean((fit - ff)**2))
    return a, b, c, mse


def detect_candidates_path(cfg: SearchConfig,
                           result: Dict[str, Any],
                           r_val: float,
                           eps_val: float) -> List[Candidate]:
    """1D path detector using local quadratic fits."""
    s = result["s"]
    a1, a2 = result["a1"], result["a2"]
    k_labels, k_inds = result["k_labels"], result["k_label_inds"]

    all_cands: List[Candidate] = []

    def process(freqs: np.ndarray, pol_name: str):
        nonlocal all_cands
        nk, nb = freqs.shape
        cand_k = list(k_inds) if cfg.high_symmetry_points_only else list(range(nk))

        for b in range(nb):
            fb = freqs[:, b]
            for ki in cand_k:
                a, b_lin, c, mse = _fit_local_quadratic_1d(s, fb, ki, cfg.window_halfwidth)
                # checks
                if cfg.target_extremum == "min" and not (a > 0): continue
                if cfg.target_extremum == "max" and not (a < 0): continue
                if not (cfg.curvature_min <= abs(a) <= cfg.curvature_max): continue
                if abs(b_lin) > cfg.linear_term_tol: continue
                if mse > cfg.residual_tol: continue
                f0 = fb[ki]
                gap_below = f0 - freqs[ki, b-1] if b-1 >= 0 else float("inf")
                gap_above = freqs[ki, b+1] - f0 if b+1 < nb else float("inf")
                if min(gap_below, gap_above) < cfg.min_gap_to_neighbors: continue

                klabel = k_labels[list(k_inds).index(ki)] if ki in k_inds else ""
                all_cands.append(Candidate(
                    lattice_type=cfg.lattice_type,
                    a1x=float(a1[0]), a1y=float(a1[1]),
                    a2x=float(a2[0]), a2y=float(a2[1]),
                    hole_radius=float(r_val),
                    eps_bg=float(eps_val),
                    polarization=pol_name,
                    band_index=b,
                    k_index=ki,
                    k_label=klabel,
                    s_value=float(s[ki]),
                    freq=float(f0),
                    extremum_type="min" if a > 0 else "max",
                    curvature_a=float(a),
                    linear_b=float(b_lin),
                    quad_residual=float(mse),
                    gap_below=float(0.0 if math.isinf(gap_below) else gap_below),
                    gap_above=float(0.0 if math.isinf(gap_above) else gap_above),
                    plot_file="",
                    mode="path"
                ))

    pol = cfg.polarization.lower()
    if pol == "both":
        for pol_name in ("TM", "TE"):
            freqs = result["freqs_tm"] if pol_name == "TM" else result["freqs_te"]
            process(freqs, pol_name)
    else:
        freqs = result["freqs"]
        process(freqs, pol.upper())

    return all_cands


def _fit_local_quadratic_2d(kx: np.ndarray, ky: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit f(kx,ky) = 0.5 * [kx ky] H [kx ky]^T + g^T [kx ky] + c
    Returns (H[2x2], g[2], mse).
    """
    X = np.column_stack([0.5*kx*kx, kx*ky, 0.5*ky*ky, kx, ky, np.ones_like(kx)])
    # Solve least squares
    beta, *_ = np.linalg.lstsq(X, f, rcond=None)
    H = np.array([[beta[0], beta[1]], [beta[1], beta[2]]])
    g = np.array([beta[3], beta[4]])
    fhat = X @ beta
    mse = float(np.mean((fhat - f)**2))
    return H, g, mse


def detect_candidates_mesh(cfg: SearchConfig,
                           result: Dict[str, Any],
                           r_val: float,
                           eps_val: float) -> List[Candidate]:
    """2D mesh detector over BZ: local quadratic fit -> Hessian eigen-test + small gradient."""
    kfrac = result["k_frac"]       # [nk,2]
    a1, a2 = result["a1"], result["a2"]
    poly = result["bz_poly_frac"]  # polygon (unused here, but could be plotted)

    # Build a regular grid index to get neighbor windows quickly.
    # We assume k_frac came from a regular mesh masked by polygon, so grid spacing is uniform.
    # Compute spacing approximately from unique sorted coordinates.
    xs = np.unique(np.round(kfrac[:,0], 6))
    ys = np.unique(np.round(kfrac[:,1], 6))
    dx = np.min(np.diff(xs)) if len(xs) > 1 else 1.0
    dy = np.min(np.diff(ys)) if len(ys) > 1 else 1.0
    tol = 0.25 * min(dx, dy)

    # Map fractional coords -> indices
    def key(ix: int, iy: int) -> Tuple[int, int]: return (ix, iy)
    x_to_i = {val: i for i, val in enumerate(xs)}
    y_to_j = {val: j for j, val in enumerate(ys)}

    # Bucket points onto grid indices (nearest neighbor)
    pts_idx = {}
    for idx, (px, py) in enumerate(kfrac):
        ix = int(round((px - xs[0]) / dx))
        iy = int(round((py - ys[0]) / dy))
        pts_idx[(ix, iy)] = idx

    window = cfg.oblique_window

    all_cands: List[Candidate] = []

    def process(freqs: np.ndarray, pol_name: str):
        nonlocal all_cands
        nk, nb = freqs.shape

        # For inner grid points that have full (2w+1)^2 neighborhoods
        # We'll iterate over the index dictionary but require neighbors present
        present = set(pts_idx.keys())
        for (ix, iy) in list(present):
            neigh = [(ix+u, iy+v) for u in range(-window, window+1) for v in range(-window, window+1)]
            if not all((n in present) for n in neigh):
                continue  # skip boundary / missing neighbors

            idxs = [pts_idx[n] for n in neigh]
            kxloc = kfrac[idxs, 0] - kfrac[pts_idx[(ix,iy)], 0]
            kyloc = kfrac[idxs, 1] - kfrac[pts_idx[(ix,iy)], 1]

            for b in range(nb):
                ff = freqs[idxs, b]
                H, g, mse = _fit_local_quadratic_2d(kxloc, kyloc, ff)

                # eigen-analysis
                w, _ = np.linalg.eigh(H)
                lam_min = float(np.min(np.abs(w)))
                # classify
                if cfg.target_extremum == "min" and not np.all(w > 0): 
                    continue
                if cfg.target_extremum == "max" and not np.all(w < 0): 
                    continue
                if not (cfg.mesh_curv_min <= lam_min <= cfg.mesh_curv_max):
                    continue
                if np.linalg.norm(g) > cfg.mesh_grad_tol:
                    continue
                if mse > cfg.mesh_residual_tol:
                    continue

                center_idx = pts_idx[(ix, iy)]
                f0 = float(freqs[center_idx, b])
                gap_below = f0 - freqs[center_idx, b-1] if b-1 >= 0 else float("inf")
                gap_above = freqs[center_idx, b+1] - f0 if b+1 < nb else float("inf")
                if min(gap_below, gap_above) < cfg.min_gap_to_neighbors:
                    continue

                all_cands.append(Candidate(
                    lattice_type=cfg.lattice_type,
                    a1x=float(a1[0]), a1y=float(a1[1]),
                    a2x=float(a2[0]), a2y=float(a2[1]),
                    hole_radius=float(r_val),
                    eps_bg=float(eps_val),
                    polarization=pol_name,
                    band_index=b,
                    k_index=center_idx,
                    k_label="",
                    s_value=float("nan"),
                    freq=f0,
                    extremum_type="min" if np.all(w > 0) else "max",
                    curvature_a=lam_min,
                    linear_b=float(np.linalg.norm(g)),
                    quad_residual=float(mse),
                    gap_below=float(0.0 if math.isinf(gap_below) else gap_below),
                    gap_above=float(0.0 if math.isinf(gap_above) else gap_above),
                    plot_file="",
                    kx_frac=float(kfrac[center_idx, 0]),
                    ky_frac=float(kfrac[center_idx, 1]),
                    mode="mesh"
                ))

    pol = cfg.polarization.lower()
    if pol == "both":
        for pol_name in ("TM", "TE"):
            freqs = result["freqs_tm"] if pol_name == "TM" else result["freqs_te"]
            process(freqs, pol_name)
    else:
        freqs = result["freqs"]
        process(freqs, pol.upper())

    return all_cands


# ---------------------------
# PLOTTING
# ---------------------------

def plot_bands_with_extrema(cfg: SearchConfig,
                            result: Dict[str, Any],
                            candidates: List[Candidate],
                            out_png: str):
    """Plot 1D band diagram with highlighted candidate extrema."""
    s = result["s"]
    k_labels = result["k_labels"]
    k_inds = result["k_label_inds"]

    plt.figure(figsize=(10, 6))
    if ("freqs_tm" in result) or ("freqs_te" in result) or (cfg.polarization.lower() == "both"):
        blues = plt.get_cmap("Blues")
        reds = plt.get_cmap("Reds")
        legend_handles: List[Line2D] = []
        if "freqs_tm" in result:
            freqs_tm = result["freqs_tm"]; nb_tm = freqs_tm.shape[1]
            colors_tm = blues(np.linspace(0.35, 0.9, max(nb_tm, 1)))
            for b in range(nb_tm): plt.plot(s, freqs_tm[:, b], lw=1.6, color=colors_tm[b])
            legend_handles.append(Line2D([0], [0], color=blues(0.65), lw=2, label="TM"))
        if "freqs_te" in result:
            freqs_te = result["freqs_te"]; nb_te = freqs_te.shape[1]
            colors_te = reds(np.linspace(0.35, 0.9, max(nb_te, 1)))
            for b in range(nb_te): plt.plot(s, freqs_te[:, b], lw=1.6, color=colors_te[b])
            legend_handles.append(Line2D([0], [0], color=reds(0.65), lw=2, label="TE"))
        if legend_handles: plt.legend(handles=legend_handles, loc="best")
    else:
        freqs = result["freqs"]
        for b in range(freqs.shape[1]): plt.plot(s, freqs[:, b], lw=1.8)

    for xi in result["k_label_inds"]:
        plt.axvline(float(s[xi]), ls="--", lw=0.8)

    plt.xticks([float(s[i]) for i in k_labels and result["k_label_inds"]], list(k_labels))
    plt.xlabel("k-path"); plt.ylabel("Frequency (c/a)")

    # Title params
    pol_label = "TE+TM" if (("freqs_tm" in result) and ("freqs_te" in result)) else cfg.polarization.upper()
    # pull r/eps from candidates if available; else from result['shape'] context (not stored), so we skip
    r_val = candidates[0].hole_radius if len(candidates) else float("nan")
    eps_val = candidates[0].eps_bg if len(candidates) else float("nan")
    plt.title(f"{cfg.lattice_type} lattice | r/a={r_val:.3f} ε_bg={eps_val:.2f} | {pol_label}")

    if len(candidates) > 0:
        xs = [float(c.s_value) for c in candidates if not np.isnan(c.s_value)]
        ys = [float(c.freq) for c in candidates if not np.isnan(c.s_value)]
        colors = ["tab:green" if c.extremum_type == "min" else "tab:red" for c in candidates if not np.isnan(c.s_value)]
        if len(xs) > 0:
            plt.scatter(xs, ys, s=60, marker="o", facecolors="none", edgecolors=colors, lw=2, zorder=5)

    plt.tight_layout(); plt.savefig(out_png, dpi=cfg.dpi); plt.close()


def plot_oblique_mesh_maps(cfg: SearchConfig,
                           result: Dict[str, Any],
                           candidates: List[Candidate],
                           out_prefix: str):
    """Extra plots for oblique mesh: heatmap of a chosen band + candidate markers over BZ polygon."""
    kfrac = result["k_frac"]; poly = result["bz_poly_frac"]
    gx, gy = result["grid_axes"]
    # Choose which freq array to visualize (first available)
    if "freqs_tm" in result:
        freqs = result["freqs_tm"]
    elif "freqs_te" in result:
        freqs = result["freqs"]
    else:
        freqs = result["freqs"]

    # We'll visualize the first band
    f0 = freqs[:, 0]

    # Re-grid onto (gx,gy) with NaNs outside polygon
    NX, NY = len(gx), len(gy)
    Z = np.full((NY, NX), np.nan, dtype=float)  # imshow expects [rows(y), cols(x)]
    # Map points to nearest grid cell
    xs0, ys0 = gx[0], gy[0]
    dx = gx[1] - gx[0] if len(gx) > 1 else 1.0
    dy = gy[1] - gy[0] if len(gy) > 1 else 1.0
    for (pt, val) in zip(kfrac, f0):
        ix = int(round((pt[0] - xs0) / dx))
        iy = int(round((pt[1] - ys0) / dy))
        if 0 <= ix < NX and 0 <= iy < NY:
            Z[iy, ix] = val

    # Heatmap
    plt.figure(figsize=(7, 6))
    extent = [gx.min(), gx.max(), gy.min(), gy.max()]
    im = plt.imshow(Z, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(im, label="freq (c/a)")
    # Polygon
    poly_closed = np.vstack([poly, poly[0]])
    plt.plot(poly_closed[:,0], poly_closed[:,1], "k-", lw=1.0, alpha=0.7)
    # Candidates
    if len(candidates) > 0:
        xs = [c.kx_frac for c in candidates if not math.isnan(c.kx_frac)]
        ys = [c.ky_frac for c in candidates if not math.isnan(c.ky_frac)]
        colors = ["green" if c.extremum_type == "min" else "red" for c in candidates if not math.isnan(c.kx_frac)]
        plt.scatter(xs, ys, s=30, facecolors="none", edgecolors=colors, lw=1.8)
    plt.xlabel("k (fractional b1)"); plt.ylabel("k (fractional b2)")
    plt.title(f"Oblique BZ mesh — band 1")
    plt.tight_layout()
    png1 = f"{out_prefix}_meshmap.png"
    plt.savefig(png1, dpi=cfg.dpi)
    plt.close()

    # Scatter-only map
    plt.figure(figsize=(7,6))
    plt.plot(poly_closed[:,0], poly_closed[:,1], "k-", lw=1.0, alpha=0.7)
    if len(candidates) > 0:
        xs = [c.kx_frac for c in candidates if not math.isnan(c.kx_frac)]
        ys = [c.ky_frac for c in candidates if not math.isnan(c.ky_frac)]
        colors = ["green" if c.extremum_type == "min" else "red" for c in candidates if not math.isnan(c.kx_frac)]
        plt.scatter(xs, ys, s=36, facecolors="none", edgecolors=colors, lw=1.8)
    plt.xlabel("k (fractional b1)"); plt.ylabel("k (fractional b2)")
    plt.title("Oblique BZ — extrema")
    plt.tight_layout()
    png2 = f"{out_prefix}_extrema.png"
    plt.savefig(png2, dpi=cfg.dpi)
    plt.close()
    return [png1, png2]


# ---------------------------
# SCORING & OBJECTIVE
# ---------------------------

def candidate_score(cfg: SearchConfig, cand: Candidate) -> float:
    # Higher is better; same form for path & mesh candidates
    curv_term = min(abs(cand.curvature_a), max(cfg.curvature_max, cfg.mesh_curv_max)) / max(cfg.curvature_max, cfg.mesh_curv_max)
    gap_term = min(cand.gap_above, cand.gap_below) / max(cfg.min_gap_to_neighbors, 1e-6)
    lin_penalty = abs(cand.linear_b) / max(cfg.linear_term_tol, 1e-6)
    res_penalty = cand.quad_residual / max(cfg.residual_tol, 1e-9)
    score = cfg.w_curv * curv_term + cfg.w_gap * gap_term - cfg.w_lin * lin_penalty - cfg.w_res * res_penalty
    return float(score)


def _shape_from_x(cfg: SearchConfig, x: np.ndarray) -> Dict[str, float]:
    lt = cfg.lattice_type.lower()
    if lt in ("triangular", "square"):
        return {}
    if lt == "rectangular":
        return {"rho": float(x[2])}
    if lt == "oblique":
        return {"rho": float(x[2]), "gamma_deg": float(x[3])}
    raise ValueError("Unknown lattice type")


def _apply_bounds_and_penalties(cfg: SearchConfig, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Clip x to legal ranges and compute soft penalties for canonical boundaries."""
    lt = cfg.lattice_type.lower()
    r = clamp(x[0], cfg.r_min, cfg.r_max)
    eps = clamp(x[1], cfg.eps_bg_min, cfg.eps_bg_max)
    pen = 0.0

    if lt == "rectangular":
        rho = clamp(x[2], cfg.rect_rho_min, cfg.rect_rho_max)
        # Soft barrier away from rho~1 (square)
        pen += soft_barrier(abs(rho - 1.0) - cfg.rect_rho_exclude_margin, cfg.rect_rho_exclude_margin)
        return np.array([r, eps, rho]), pen

    if lt == "oblique":
        rho = clamp(x[2], cfg.obl_rho_min, cfg.obl_rho_max)
        gamma = clamp(x[3], cfg.obl_gamma_min, cfg.obl_gamma_max)
        # Avoid rectangular line and also away from triangular boundaries
        for g0 in (list(cfg.obl_gamma_exclude) + [60.0, 120.0]):
            pen += soft_barrier(abs(gamma - g0) - cfg.obl_gamma_margin, cfg.obl_gamma_margin)
        return np.array([r, eps, rho, gamma]), pen

    return np.array([r, eps]), pen


def objective(cfg: SearchConfig, x: np.ndarray) -> float:
    """Minimize negative best candidate score + penalties (so optimizer maximizes score)."""
    xclip, pen = _apply_bounds_and_penalties(cfg, x)
    lt = cfg.lattice_type.lower()

    try:
        if lt == "oblique" and cfg.oblique_use_mesh_in_objective:
            # Mesh-based objective (coarser N during iterations)
            r, eps, rho, gamma = float(xclip[0]), float(xclip[1]), float(xclip[2]), float(xclip[3])
            result = run_mpb_mesh_oblique(cfg, r, eps, rho, gamma, N=cfg.oblique_mesh_n)
            cands = detect_candidates_mesh(cfg, result, r, eps)
        else:
            # Path-based objective
            r = float(xclip[0]); eps = float(xclip[1])
            shape = _shape_from_x(cfg, xclip)
            result = run_mpb_path(cfg, r, eps, shape=shape)
            cands = detect_candidates_path(cfg, result, r, eps)

        if len(cands) == 0:
            return 5.0 + pen  # penalty when no viable candidate
        best = max(candidate_score(cfg, c) for c in cands)
        return -best + pen
    except Exception as e:
        warnings.warn(f"objective failed @ x={xclip}: {e}")
        return 10.0 + pen


# ---------------------------
# CSV I/O
# ---------------------------

def write_csv_header(csv_path: str):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "lattice_type",
            "a1x","a1y","a2x","a2y",
            "hole_radius_a",
            "eps_bg",
            "polarization",
            "band_index",
            "k_index",
            "k_label",
            "s_value",
            "frequency",
            "extremum_type",
            "curvature_a",
            "linear_b",
            "quad_residual",
            "gap_below",
            "gap_above",
            "kx_frac",
            "ky_frac",
            "mode",
            "plot_file"
        ])


def append_candidate(csv_path: str, cand: Candidate):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            cand.lattice_type,
            f"{cand.a1x:.8f}", f"{cand.a1y:.8f}", f"{cand.a2x:.8f}", f"{cand.a2y:.8f}",
            f"{cand.hole_radius:.8f}",
            f"{cand.eps_bg:.8f}",
            cand.polarization,
            cand.band_index,
            cand.k_index,
            cand.k_label,
            f"{cand.s_value:.8f}" if not math.isnan(cand.s_value) else "",
            f"{cand.freq:.8f}",
            cand.extremum_type,
            f"{cand.curvature_a:.8f}",
            f"{cand.linear_b:.8f}",
            f"{cand.quad_residual:.8e}",
            f"{cand.gap_below:.8f}",
            f"{cand.gap_above:.8f}",
            f"{cand.kx_frac:.8f}" if not math.isnan(cand.kx_frac) else "",
            f"{cand.ky_frac:.8f}" if not math.isnan(cand.ky_frac) else "",
            cand.mode,
            cand.plot_file
        ])


# ---------------------------
# MAIN SEARCH PIPELINE
# ---------------------------

def main():
    cfg = SearchConfig()

    # Suffix output dir with lattice type
    cfg.output_dir = f"{cfg.output_dir}_{cfg.lattice_type.lower()}"
    ensure_dir(cfg.output_dir)

    # Grid defaults
    if cfg.r_grid is None:
        cfg.r_grid = list(np.linspace(cfg.r_min, cfg.r_max, cfg.r_grid_points))
    if cfg.eps_grid is None:
        cfg.eps_grid = list(np.linspace(cfg.eps_bg_min, cfg.eps_bg_max, cfg.eps_grid_points))

    np.random.seed(cfg.seed)

    csv_path = os.path.join(cfg.output_dir, cfg.csv_name)
    write_csv_header(csv_path)

    run_meta: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "results": []
    }

    if not MEEP_AVAILABLE:
        print("✗ Meep / MPB not available. Install it to run this script.")
        return

    # Compute progress totals
    grid_total = (len(cfg.r_grid) * len(cfg.eps_grid)) if cfg.do_grid_sweep else 0
    opt_total = (cfg.opt_maxiter if cfg.do_optimizer and SCIPY_AVAILABLE else 0)
    total_steps = max(1, grid_total + opt_total)

    if RICH_AVAILABLE:
        console = Console()
        status_text = Text("Starting…")
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("• Elapsed:"), TimeElapsedColumn(),
            TextColumn("• ETA:"), TimeRemainingColumn(),
            expand=True,
        )
        from rich.console import Group
        group = Group(progress, status_text)
        with Live(group, console=console, refresh_per_second=8, transient=False):
            task_id = progress.add_task("Initializing", total=total_steps)

            # 1) Grid sweep on r/eps (path-based except oblique where we still do a path probe)
            if cfg.do_grid_sweep:
                progress.update(task_id, description="Grid sweep")
                for r in cfg.r_grid:
                    for eps in cfg.eps_grid:
                        r_val = float(clamp(r, cfg.r_min, cfg.r_max))
                        eps_val = float(clamp(eps, cfg.eps_bg_min, cfg.eps_bg_max))
                        status_text.plain = f"Grid: r/a={r_val:.4f}, eps_bg={eps_val:.3f}"

                        try:
                            # For oblique, still do a quick path probe in grid stage (mesh is heavy)
                            shape = None
                            result = run_mpb_path(cfg, r_val, eps_val, shape=shape)
                            cands = detect_candidates_path(cfg, result, r_val, eps_val)

                            if (len(cands) > 0) and cfg.save_plots:
                                png = os.path.join(cfg.output_dir, f"bands_r{r_val:.3f}_eps{eps_val:.2f}.png")
                                plot_bands_with_extrema(cfg, result, cands, png)
                            else:
                                png = ""

                            for c in cands:
                                c.plot_file = os.path.basename(png) if png else ""
                                append_candidate(csv_path, c)

                            run_meta["results"].append({
                                "mode": "grid",
                                "r": r_val, "eps_bg": eps_val,
                                "num_candidates": len(cands),
                                "plot": os.path.basename(png) if png else ""
                            })
                        except Exception as e:
                            status_text.plain = f"Grid FAILED r/a={r_val:.4f}, eps={eps_val:.3f}: {e}"
                            run_meta["results"].append({
                                "mode": "grid",
                                "r": r_val, "eps_bg": eps_val,
                                "error": str(e)
                            })
                        finally:
                            progress.advance(task_id, 1)

            # 2) Local optimization (with continuous shape parameters when needed)
            if cfg.do_optimizer and SCIPY_AVAILABLE:
                if opt_total > 0 and progress.tasks[0].total != grid_total + opt_total:
                    progress.update(task_id, total=grid_total + opt_total)

                lt = cfg.lattice_type.lower()
                x0 = np.array(list(cfg.opt_initial), dtype=float)
                
                # pad x0 if user supplied too few params
                if lt == "rectangular" and x0.size < 3:
                    x0 = np.concatenate([x0, [1.3]])              # default rho
                elif lt == "oblique" and x0.size < 4:
                    x0 = np.concatenate([x0, [1.2, 100.0]])       # default rho, gamma_deg
                
                status_text.plain = f"Opt: {lt} starting from x0={x0.tolist()} (maxiter={cfg.opt_maxiter})"

                it = {"n": 0}
                def cb(_xk): 
                    it["n"] += 1
                    progress.advance(task_id, 1)
                    status_text.plain = f"Opt iter {it['n']}/{cfg.opt_maxiter}"

                res = _minimize(lambda x: objective(cfg, x), x0,
                                method="Nelder-Mead",
                                callback=cb,
                                options=dict(maxiter=cfg.opt_maxiter, xatol=1e-3, fatol=1e-3, disp=False))

                xbest_clip, _ = _apply_bounds_and_penalties(cfg, res.x)
                status_text.plain = f"Opt best x*={xbest_clip.tolist()}, fun={res.fun:.4f}"

                # Final confirmation run & plots
                try:
                    shape = _shape_from_x(cfg, xbest_clip)
                    r_best = float(xbest_clip[0]); eps_best = float(xbest_clip[1])

                    if lt == "oblique" and cfg.oblique_mesh:
                        # Finer mesh for confirmation + plots
                        rho = float(xbest_clip[2]); gamma = float(xbest_clip[3])
                        result = run_mpb_mesh_oblique(cfg, r_best, eps_best, rho, gamma, N=cfg.oblique_mesh_n_final)
                        cands = detect_candidates_mesh(cfg, result, r_best, eps_best)
                        pngs = []
                        if cfg.save_plots:
                            out_prefix = os.path.join(cfg.output_dir, f"OBL_r{r_best:.3f}_eps{eps_best:.2f}_rho{rho:.3f}_g{gamma:.1f}")
                            pngs = plot_oblique_mesh_maps(cfg, result, cands, out_prefix)
                    else:
                        result = run_mpb_path(cfg, r_best, eps_best, shape=shape)
                        cands = detect_candidates_path(cfg, result, r_best, eps_best)
                        pngs = []
                        if cfg.save_plots:
                            png = os.path.join(cfg.output_dir, f"bands_OPT_r{r_best:.3f}_eps{eps_best:.2f}.png")
                            plot_bands_with_extrema(cfg, result, cands, png)
                            pngs = [png]

                    for c in cands:
                        # Record the *first* plot file (others are supplementary)
                        c.plot_file = os.path.basename(pngs[0]) if (pngs and len(pngs) > 0) else ""
                        append_candidate(csv_path, c)

                    run_meta["results"].append({
                        "mode": "opt",
                        "x0": list(cfg.opt_initial),
                        "best": xbest_clip.tolist(),
                        "fun": float(res.fun),
                        "success": bool(res.success),
                        "message": str(res.message),
                        "num_candidates": len(cands),
                        "plots": [os.path.basename(p) for p in pngs]
                    })
                except Exception as e:
                    status_text.plain = f"Opt confirm FAILED: {e}"
                    run_meta["results"].append({
                        "mode": "opt",
                        "error": str(e)
                    })

            elif cfg.do_optimizer and not SCIPY_AVAILABLE:
                status_text.plain = "SciPy not available — skipping optimizer."

            # Save run log
            log_path = os.path.join(cfg.output_dir, cfg.log_name)
            with open(log_path, "w") as f:
                json.dump(run_meta, f, indent=2)

            status_text.plain = (
                f"Done | CSV: {csv_path} | Log: {log_path}"
                + (f" | Plots: {cfg.output_dir}/" if cfg.save_plots else "")
            )
    else:
        # Fallback without Rich UI
        # 1) Grid sweep
        if cfg.do_grid_sweep:
            for r in cfg.r_grid:
                for eps in cfg.eps_grid:
                    r_val = float(clamp(r, cfg.r_min, cfg.r_max))
                    eps_val = float(clamp(eps, cfg.eps_bg_min, cfg.eps_bg_max))
                    print(f"[Grid] r/a={r_val:.4f}, eps_bg={eps_val:.3f}")
                    try:
                        result = run_mpb_path(cfg, r_val, eps_val, shape=None)
                        cands = detect_candidates_path(cfg, result, r_val, eps_val)
                        if (len(cands) > 0) and cfg.save_plots:
                            png = os.path.join(cfg.output_dir, f"bands_r{r_val:.3f}_eps{eps_val:.2f}.png")
                            plot_bands_with_extrema(cfg, result, cands, png)
                        else:
                            png = ""
                        for c in cands:
                            c.plot_file = os.path.basename(png) if png else ""
                            append_candidate(csv_path, c)
                    except Exception as e:
                        warnings.warn(f"Grid run failed @ (r={r_val:.4f}, eps={eps_val:.3f}): {e}")

        # 2) Optimizer
        if cfg.do_optimizer and SCIPY_AVAILABLE:
            lt = cfg.lattice_type.lower()
            x0 = np.array(list(cfg.opt_initial), dtype=float)
            
            # pad x0 if user supplied too few params
            if lt == "rectangular" and x0.size < 3:
                x0 = np.concatenate([x0, [1.3]])              # default rho
            elif lt == "oblique" and x0.size < 4:
                x0 = np.concatenate([x0, [1.2, 100.0]])       # default rho, gamma_deg
            
            print(f"[Opt] Starting local optimization from x0={x0} with Nelder-Mead (maxiter={cfg.opt_maxiter})")
            res = _minimize(lambda x: objective(cfg, x), x0, method="Nelder-Mead",
                            options=dict(maxiter=cfg.opt_maxiter, xatol=1e-3, fatol=1e-3, disp=False))
            xbest_clip, _ = _apply_bounds_and_penalties(cfg, res.x)
            print(f"[Opt] Best (approx): x*={xbest_clip} (fun={res.fun:.4f})")

            try:
                shape = _shape_from_x(cfg, xbest_clip)
                r_best = float(xbest_clip[0]); eps_best = float(xbest_clip[1])
                if cfg.lattice_type.lower() == "oblique" and cfg.oblique_mesh:
                    rho = float(xbest_clip[2]); gamma = float(xbest_clip[3])
                    result = run_mpb_mesh_oblique(cfg, r_best, eps_best, rho, gamma, N=cfg.oblique_mesh_n_final)
                    cands = detect_candidates_mesh(cfg, result, r_best, eps_best)
                    if cfg.save_plots:
                        out_prefix = os.path.join(cfg.output_dir, f"OBL_r{r_best:.3f}_eps{eps_best:.2f}_rho{rho:.3f}_g{gamma:.1f}")
                        plot_oblique_mesh_maps(cfg, result, cands, out_prefix)
                else:
                    result = run_mpb_path(cfg, r_best, eps_best, shape=shape)
                    cands = detect_candidates_path(cfg, result, r_best, eps_best)
                    if cfg.save_plots:
                        png = os.path.join(cfg.output_dir, f"bands_OPT_r{r_best:.3f}_eps{eps_best:.2f}.png")
                        plot_bands_with_extrema(cfg, result, cands, png)

                for c in cands:
                    append_candidate(csv_path, c)
            except Exception as e:
                warnings.warn(f"Opt confirm failed: {e}")

        print(f"Done. CSV: {csv_path}  (plots in {cfg.output_dir}/)")

if __name__ == "__main__":
    main()