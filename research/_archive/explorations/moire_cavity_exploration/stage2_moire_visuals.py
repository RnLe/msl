#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 — First moiré involvement (composition visuals + registry mini-cells + band diagrams)
=============================================================================================

Changes in this revision (based on feedback):
- Real-space panel now shows the COMPOSITION of both parent lattices (lattice_1 & lattice_2) from the
  moiré object, not just the moiré cell grid. Moiré vectors are taken from ml.PyMoire2D.as_lattice2d().
- Reciprocal-space panel overlays BOTH parent reciprocal lattices; extent is 2 translations in each
  direction (i.e., i,j in [-2,2]).
- Moiré reciprocal panel shows *only* registry dots (AA/AB/BA) without extra grid lines.
- Added three stacking panels (AA/AB/BA): each shows the unit cell of lattice_1 and a basis composed
  of all lattice_2 lattice sites that fall into that cell for the given registry shift. This yields
  a 2-atom-or-more basis view of the stacking micro-cell.
- Band diagrams for each registry now include cylinders for all basis sites from lattice_2 that are in
  the lattice_1 cell (fractional in [0,1)^2). This replaces the earlier single-shift proxy.

Interrupt behavior and logging are unchanged: Ctrl+C propagates cleanly.
"""

from __future__ import annotations

import os, sys, math, csv, signal, argparse, logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# NEW: parallel + tracing imports
import traceback
from multiprocessing import Pool, cpu_count

# Rich logging/progress
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.logging import RichHandler
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None
    class RichHandler(logging.StreamHandler):  # type: ignore
        def __init__(self, *a, **k): super().__init__(stream=sys.stdout)

handlers: List[logging.Handler] = []
if not os.path.isdir("stage2_logs"): os.makedirs("stage2_logs", exist_ok=True)
fh = logging.FileHandler(os.path.join("stage2_logs", "stage2.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG); handlers.append(fh)
ch = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True) if RICH else logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO); handlers.append(ch)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
log = logging.getLogger("stage2")

# Optional dependencies
MEEP_AVAILABLE = False
try:
    import meep as mp
    from meep import mpb
    MEEP_AVAILABLE = True
except Exception as e:
    log.warning("Meep/MPB not importable; band diagrams will be skipped. %s", e)

ML_AVAILABLE = False
try:
    import moire_lattice_py as ml
    ML_AVAILABLE = True
except Exception as e:
    log.info("moire_lattice_py not importable; using analytic moiré fallback. %s", e)

# ---------------------- Data structures -----------------------------------------------------------

@dataclass
class Candidate:
    lattice: str
    r: float
    eps_bg: float
    k_label: str
    band_index: int
    pol: str

# ---------------------- Helpers ------------------------------------------------------------------

def rotation_matrix(theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    return np.array([[math.cos(th), -math.sin(th)],
                     [math.sin(th),  math.cos(th)]], dtype=float)

def canonical_lattice_vectors(lattice: str) -> Tuple[np.ndarray, np.ndarray]:
    L = lattice.lower()
    if L in ("square",):
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if L in ("triangular", "hexagonal"):
        return np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3)/2.0])
    if L in ("rectangular",):
        return np.array([1.0, 0.0]), np.array([0.0, 1.4])
    if L in ("oblique",):
        return np.array([1.0, 0.0]), np.array([0.7, 0.9])
    raise ValueError(f"Unsupported lattice: {lattice}")

def reciprocal_from_direct(a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.column_stack([a1, a2])
    B = 2*math.pi * np.linalg.inv(A).T
    return B[:,0].copy(), B[:,1].copy()

def extract_vecs_from_py_lattice2d(py_lat) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract 2D direct and reciprocal basis from a PyLattice2D-like object.
    Tries friendly 2D accessors first, then falls back to 3D basis matrices.
    Returns (a1, a2, b1, b2) as 2D numpy arrays.
    """
    # If we were passed a bound/builtin method, call it to obtain the lattice object
    if callable(py_lat) and not hasattr(py_lat, "lattice_vectors") and not hasattr(py_lat, "direct_basis"):
        py_lat = py_lat()

    # Try simple 2D vector accessors if present
    a1 = a2 = b1 = b2 = None

    # Direct lattice vectors
    if hasattr(py_lat, "lattice_vectors") and callable(getattr(py_lat, "lattice_vectors")):
        (a1x, a1y), (a2x, a2y) = py_lat.lattice_vectors()
        a1 = np.array([a1x, a1y], dtype=float)
        a2 = np.array([a2x, a2y], dtype=float)
    elif hasattr(py_lat, "primitive_vectors") and callable(getattr(py_lat, "primitive_vectors")):
        v1, v2 = py_lat.primitive_vectors()
        a1 = np.array([v1[0], v1[1]], dtype=float)
        a2 = np.array([v2[0], v2[1]], dtype=float)
    elif hasattr(py_lat, "direct_basis") and callable(getattr(py_lat, "direct_basis")):
        c1, c2, _ = py_lat.direct_basis()
        a1 = np.array([c1[0], c1[1]], dtype=float)
        a2 = np.array([c2[0], c2[1]], dtype=float)
    else:
        raise AttributeError("Object does not expose direct lattice vectors")

    # Reciprocal lattice vectors
    if hasattr(py_lat, "reciprocal_vectors") and callable(getattr(py_lat, "reciprocal_vectors")):
        (b1x, b1y), (b2x, b2y) = py_lat.reciprocal_vectors()
        b1 = np.array([b1x, b1y], dtype=float)
        b2 = np.array([b2x, b2y], dtype=float)
    elif hasattr(py_lat, "reciprocal_basis") and callable(getattr(py_lat, "reciprocal_basis")):
        g1, g2, _ = py_lat.reciprocal_basis()
        b1 = np.array([g1[0], g1[1]], dtype=float)
        b2 = np.array([g2[0], g2[1]], dtype=float)
    else:
        b1, b2 = reciprocal_from_direct(a1, a2)

    return a1, a2, b1, b2

def build_moire_and_parents(lattice: str, angle_deg: float):
    """
    Build moiré and parent lattices using moire_lattice_py if available.
    Returns ((L1,L2),(B1m,B2m), (a1_1,a2_1,B1_1,B2_1), (a1_2,a2_2,B1_2,B2_2)).
    Falls back to analytic if bindings not available.
    """
    if ML_AVAILABLE:
        # Map lattice string to a creator in ml
        lat = lattice.lower()
        a = 1.0
        base = None
        if hasattr(ml, "create_square_lattice") and lat == "square":
            base = ml.create_square_lattice(a)
        elif hasattr(ml, "create_hexagonal_lattice") and lat in ("triangular","hexagonal"):
            base = ml.create_hexagonal_lattice(a)
        elif hasattr(ml, "create_rectangular_lattice") and lat == "rectangular":
            base = ml.create_rectangular_lattice(a, a*1.4)
        # Construct moiré object and extract parent lattices
        tb = None
        if hasattr(ml, "py_twisted_bilayer") and base is not None:
            tb = ml.py_twisted_bilayer(base, float(angle_deg))
        if tb is not None:
            # moiré as regular lattice
            lat_m = tb.as_lattice2d() if hasattr(tb, "as_lattice2d") and callable(getattr(tb, "as_lattice2d")) else tb
            L1, L2, B1m, B2m = extract_vecs_from_py_lattice2d(lat_m)
            # parents (call methods, don't take method objects)
            lat1 = None
            lat2 = None
            if hasattr(tb, "lattice_1") and callable(getattr(tb, "lattice_1")):
                lat1 = tb.lattice_1()
            elif hasattr(tb, "get_lattice_1") and callable(getattr(tb, "get_lattice_1")):
                lat1 = tb.get_lattice_1()
            if hasattr(tb, "lattice_2") and callable(getattr(tb, "lattice_2")):
                lat2 = tb.lattice_2()
            elif hasattr(tb, "get_lattice_2") and callable(getattr(tb, "get_lattice_2")):
                lat2 = tb.get_lattice_2()
            if lat1 is not None and lat2 is not None:
                a1_1,a2_1,B1_1,B2_1 = extract_vecs_from_py_lattice2d(lat1)
                a1_2,a2_2,B1_2,B2_2 = extract_vecs_from_py_lattice2d(lat2)
                return (L1,L2),(B1m,B2m),(a1_1,a2_1,B1_1,B2_1),(a1_2,a2_2,B1_2,B2_2)
            # Fallback: construct parents from base and rotated base
            a1,a2,B1,B2 = extract_vecs_from_py_lattice2d(base)
            R = rotation_matrix(angle_deg)
            a1_2, a2_2 = R @ a1, R @ a2
            B1_2, B2_2 = R @ B1, R @ B2
            return (L1,L2),(B1m,B2m),(a1,a2,B1,B2),(a1_2,a2_2,B1_2,B2_2)

    # Analytic fallback (no ml)
    a1,a2 = canonical_lattice_vectors(lattice)
    b1,b2 = reciprocal_from_direct(a1,a2)
    R = rotation_matrix(angle_deg)
    a1_2,a2_2 = R @ a1, R @ a2
    b1_2,b2_2 = R @ b1, R @ b2
    db1, db2 = (R @ b1 - b1), (R @ b2 - b2)
    D = np.column_stack([db1, db2])
    L = np.linalg.inv(D.T) * (2*math.pi)
    L1,L2 = L[:,0], L[:,1]
    Bm = 2*math.pi * np.linalg.inv(np.column_stack([L1,L2])).T
    B1m,B2m = Bm[:,0], Bm[:,1]
    return (L1,L2),(B1m,B2m),(a1,a2,b1,b2),(a1_2,a2_2,b1_2,b2_2)

def stacking_registers_frac(lattice: str) -> Dict[str, Tuple[float,float]]:
    L = lattice.lower()
    if L in ("square","rectangular"):
        return {"AA": (0.0, 0.0), "AB": (0.5, 0.0), "BA": (0.0, 0.5)}
    if L in ("triangular","hexagonal"):
        return {"AA": (0.0, 0.0), "AB": (1/3, 1/3), "BA": (2/3, 2/3)}
    return {"AA": (0.0, 0.0), "AB": (0.33, 0.17), "BA": (0.66, 0.34)}

# ---------------------- Plotting helpers ----------------------------------------------------------

def _dense_parent_points(a1: np.ndarray, a2: np.ndarray, L1: np.ndarray, L2: np.ndarray, n: int) -> Tuple[List[float], List[float]]:
    """Generate parent lattice points over translations of the moiré cell in [-n, n]^2.
    Uses a small stencil around each moiré-cell origin to provide a reasonably dense cloud.
    Returns two lists (xs, ys).
    """
    xs: List[float] = []
    ys: List[float] = []
    for I in range(-n, n+1):
        for J in range(-n, n+1):
            origin = I*L1 + J*L2
            for i in range(-1, 2):
                for j in range(-1, 2):
                    p = origin + i*a1 + j*a2
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
    return xs, ys

def draw_real_composition(ax, a1_1, a2_1, a1_2, a2_2, L1, L2, n: int = 2):

    """Draw both parent lattices densely over ±n moiré cells; outline a 3×3 moiré grid with central cell emphasized."""
    ax.set_aspect("equal","box")
    # Dense point clouds from parents over moiré bounding box
    xs1, ys1 = _dense_parent_points(a1_1, a2_1, L1, L2, n)
    xs2, ys2 = _dense_parent_points(a1_2, a2_2, L1, L2, n)
    # Draw 3×3 moiré grid
    base = np.array([[0,0],[1,0],[1,1],[0,1]], float)
    for I in range(-1, 2):
        for J in range(-1, 2):
            cell = (base[:,0])[:,None]*L1[None,:] + (base[:,1])[:,None]*L2[None,:] + (I*L1 + J*L2)[None,:]
            lw = 2.2 if (I==0 and J==0) else 0.8
            ax.add_patch(Polygon(cell, closed=True, fill=False, linewidth=lw, alpha=0.9))
    ax.scatter(xs1, ys1, s=8, alpha=0.9, label="lattice 1")
    ax.scatter(xs2, ys2, s=8, alpha=0.9, label="lattice 2")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Real space — composition (±{n} moiré cells)", fontweight="bold", fontsize=12)
    ax.set_xlabel("x"); ax.set_ylabel("y")


def draw_reciprocal_composition(ax, B1_1, B2_1, B1_2, B2_2, n: int = 2):
    """Overlay both parent reciprocal lattices as dots; 2 in each direction."""
    ax.set_aspect("equal","box")
    x1,y1,x2,y2 = [],[],[],[]
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            g1 = i*B1_1 + j*B2_1; x1.append(g1[0]); y1.append(g1[1])
            g2 = i*B1_2 + j*B2_2; x2.append(g2[0]); y2.append(g2[1])
    ax.scatter(x1,y1,s=10, alpha=0.9, label="G (lattice 1)")
    ax.scatter(x2,y2,s=10, alpha=0.9, label="G (lattice 2)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Reciprocal space — both parents (±2)")
    ax.set_xlabel("k_x"); ax.set_ylabel("k_y")

def draw_moire_registry_only(ax, B1m, B2m, regs: Dict[str, Tuple[float,float]]):
    """Moiré reciprocal panel: only registry dots, no extra grid."""
    ax.set_aspect("equal","box")
    for name,(fx,fy) in regs.items():
        k = fx*B1m + fy*B2m
        ax.scatter([k[0]],[k[1]], s=50, label=name, zorder=5)
    ax.legend(title="Registry", loc="upper right", fontsize=8)
    ax.set_title("Moiré reciprocal — registry dots only")
    ax.set_xlabel("k_x"); ax.set_ylabel("k_y")

def wrap01(u: float) -> float:
    w = u - math.floor(u)
    return 0.0 if abs(w-1.0) < 1e-12 else w

def unique_points(points: List[Tuple[float,float]], tol: float = 1e-9) -> List[Tuple[float,float]]:
    out: List[Tuple[float,float]] = []
    for (u,v) in points:
        if not any((abs(u-u2) < tol and abs(v-v2) < tol) for (u2,v2) in out):
            out.append((u,v))
    return out

def stacking_basis_in_cell(a1_1, a2_1, a1_2, a2_2, shift_frac: Tuple[float,float]) -> List[Tuple[float,float]]:
    """
    Compute basis points from lattice_2 that fall inside the lattice_1 unit cell.
    The cell is spanned by (a1_1, a2_1). Basis points are returned as fractional coords in [0,1)^2.
    The registry shift is interpreted as a *fractional shift in lattice_1 coordinates*.
    """
    A1 = np.column_stack([a1_1, a2_1])
    Ai = np.linalg.inv(A1)
    shift = shift_frac[0]*a1_1 + shift_frac[1]*a2_1
    pts: List[Tuple[float,float]] = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i*a1_2 + j*a2_2 + shift
            uv = Ai @ p
            u,v = uv[0], uv[1]
            if (-1e-9 <= u < 1.0+1e-9) and (-1e-9 <= v < 1.0+1e-9):
                pts.append((wrap01(u), wrap01(v)))
    return unique_points(pts)

def draw_stacking_cell(ax, a1_1, a2_1, basis_uv: List[Tuple[float,float]], title: str):
    """Draw lattice_1 unit cell and plot basis points from lattice_2 in that cell."""
    ax.set_aspect("equal","box")
    base = np.array([[0,0],[1,0],[1,1],[0,1]], float)
    cell = (base[:,0])[:,None]*a1_1[None,:] + (base[:,1])[:,None]*a2_1[None,:]
    ax.add_patch(Polygon(cell, closed=True, fill=False, linewidth=2.0))
    ax.scatter([0.0],[0.0], s=40, marker="s", label="lattice 1 site")
    if basis_uv:
        xs,ys = [],[]
        for (u,v) in basis_uv:
            p = u*a1_1 + v*a2_1
            xs.append(p[0]); ys.append(p[1])
        ax.scatter(xs, ys, s=40, label="lattice 2 basis")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")

# ---------------------- MPB band diagrams ---------------------------------------------------------

def hs_points(lattice: str) -> Dict[str, np.ndarray]:
    L = lattice.lower()
    if L == "square":
        return {"Γ": np.array([0.0, 0.0]), "X": np.array([0.5, 0.0]), "M": np.array([0.5, 0.5])}
    if L in ("triangular","hexagonal"):
        return {"Γ": np.array([0.0, 0.0]), "M": np.array([0.5, 0.0]), "K": np.array([1/3, 1/3])}
    if L == "rectangular":
        return {"Γ": np.array([0.0,0.0]), "X": np.array([0.5,0.0]), "Y": np.array([0.0,0.5]), "M": np.array([0.5,0.5])}
    return {"Γ": np.array([0.0,0.0]), "B1": np.array([0.5,0.0]), "B2": np.array([0.0,0.5])}

def default_k_path(lattice: str) -> List[str]:
    L = lattice.lower()
    if L == "square":       return ["Γ","X","M","Γ"]
    if L in ("triangular","hexagonal"): return ["Γ","M","K","Γ"]
    if L == "rectangular":  return ["Γ","X","M","Y","Γ"]
    return ["Γ","B1","B2","Γ"]

def mp_lattice_from_vectors(a1: np.ndarray, a2: np.ndarray):
    return mp.Lattice(size=mp.Vector3(1,1,0),
                      basis1=mp.Vector3(float(a1[0]), float(a1[1]), 0),
                      basis2=mp.Vector3(float(a2[0]), float(a2[1]), 0))

def build_kpath(P: Dict[str,np.ndarray], labels: List[str], pts_per_segment: int):
    nodes = [mp.Vector3(float(P[l][0]), float(P[l][1]), 0.0) for l in labels]
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

def run_band_diagram_registry(a1_1, a2_1, r: float, eps_bg: float,
                              basis_uv_2: List[Tuple[float,float]], pol: str,
                              num_bands: int, resolution: int, klabels: List[str], lattice_type: str):
    """MPB: lattice = lattice_1 cell; geometry = cylinder at (0,0) + cylinders at all lattice_2 basis points."""
    lat = mp_lattice_from_vectors(a1_1, a2_1)
    geom = [mp.Cylinder(radius=float(r), material=mp.air, center=mp.Vector3(0,0,0))]
    for (u,v) in basis_uv_2:
        p = u*a1_1 + v*a2_1
        geom.append(mp.Cylinder(radius=float(r), material=mp.air,
                                center=mp.Vector3(float(p[0]), float(p[1]), 0)))
    mat = mp.Medium(epsilon=float(eps_bg))

    P = hs_points(lattice_type)
    kpts, kidx = build_kpath(P, klabels, pts_per_segment=48)

    ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom, default_material=mat,
                        k_points=kpts, resolution=int(resolution),
                        num_bands=int(num_bands), dimensions=2)
    pol = pol.strip().lower()
    if pol == "tm":
        ms.run_tm()
    else:
        ms.run_te()
    freqs = np.array(ms.all_freqs, float)
    return freqs, kidx

# ---------------------- I/O and driver ------------------------------------------------------------

def read_seeds_csv(path: str, limit: Optional[int]) -> List[Candidate]:
    out: List[Candidate] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append(Candidate(
                    lattice=row.get("lattice","square"),
                    r=float(row["r"]),
                    eps_bg=float(row["eps_bg"]),
                    k_label=row.get("k_label","Γ"),
                    band_index=int(row.get("band_index","0")),
                    pol=row.get("pol","TM")
                ))
            except Exception as e:
                log.warning("Skipping malformed row: %s (%s)", row, e)
            if limit is not None and len(out) >= limit: break
    return out

# NEW: worker initialization to avoid thread oversubscription
_def_thread_env = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

def _init_pool_env():
    for k, v in _def_thread_env.items():
        os.environ.setdefault(k, v)

# NEW: worker function to render a single candidate

def _render_worker(task: Tuple[int, Candidate, float, int, int, str]) -> Tuple[int, str, bool, str]:
    try:
        _init_pool_env()
        i, cand, angle_deg, resolution, num_bands, outdir = task
        out_png = os.path.join(outdir, f"stage2_{i:03d}_{cand.lattice}_k{cand.k_label}_b{cand.band_index}.png")
        figure_for_candidate(cand, angle_deg=angle_deg, out_png=out_png,
                             resolution=resolution, num_bands=num_bands)
        return (i, out_png, True, "")
    except KeyboardInterrupt:
        # Let main handle
        raise
    except Exception as e:
        return (task[0], "", False, f"{e}\n{traceback.format_exc()}")


def figure_for_candidate(cand: Candidate, angle_deg: float, out_png: str,
                         resolution: int, num_bands: int):
    (L1,L2),(B1m,B2m),(a1_1,a2_1,B1_1,B2_1),(a1_2,a2_2,B1_2,B2_2) = build_moire_and_parents(cand.lattice, angle_deg)
    regs = stacking_registers_frac(cand.lattice)

    fig = plt.figure(figsize=(16.8, 11.0), constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1,1,1])

    # Row 0
    ax_r = fig.add_subplot(gs[0,0]); draw_real_composition(ax_r, a1_1,a2_1,a1_2,a2_2,L1,L2,n=2)
    ax_g = fig.add_subplot(gs[0,1]); draw_reciprocal_composition(ax_g, B1_1,B2_1,B1_2,B2_2,n=2)
    ax_m = fig.add_subplot(gs[0,2]); draw_moire_registry_only(ax_m, B1m,B2m, regs)

    # Row 1: stacking cells
    ax_sAA = fig.add_subplot(gs[1,0])
    ax_sAB = fig.add_subplot(gs[1,1])
    ax_sBA = fig.add_subplot(gs[1,2])

    basis_AA = stacking_basis_in_cell(a1_1,a2_1,a1_2,a2_2, regs["AA"])
    basis_AB = stacking_basis_in_cell(a1_1,a2_1,a1_2,a2_2, regs["AB"])
    basis_BA = stacking_basis_in_cell(a1_1,a2_1,a1_2,a2_2, regs["BA"])

    draw_stacking_cell(ax_sAA, a1_1,a2_1, basis_AA, "Stacking AA — unit cell (lattice 1) + basis (lattice 2)")
    draw_stacking_cell(ax_sAB, a1_1,a2_1, basis_AB, "Stacking AB — unit cell (lattice 1) + basis (lattice 2)")
    draw_stacking_cell(ax_sBA, a1_1,a2_1, basis_BA, "Stacking BA — unit cell (lattice 1) + basis (lattice 2)")

    # Row 2: band panels
    if MEEP_AVAILABLE:
        axs = [fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1]), fig.add_subplot(gs[2,2])]
        for ax, name, basis in zip(axs, ["AA","AB","BA"], [basis_AA,basis_AB,basis_BA]):
            klabels = default_k_path(cand.lattice)
            freqs, kidx = run_band_diagram_registry(a1_1,a2_1, cand.r, cand.eps_bg, basis, cand.pol,
                                                    num_bands, resolution, klabels, cand.lattice)
            Nk, Nb = freqs.shape
            xs = np.arange(Nk)
            for b in range(Nb):
                lw = 2.0 if b == cand.band_index else 0.9
                alpha = 1.0 if b == cand.band_index else 0.9
                ax.plot(xs, freqs[:,b], lw=lw, alpha=alpha)
            ticks = []; ticklabels = []
            for lbl in klabels:
                xi = kidx[lbl]; ax.axvline(xi, linewidth=0.8, alpha=0.6)
                ticks.append(xi); ticklabels.append(lbl)
            ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=8)
            ax.set_xlim(0, Nk-1)
            ax.set_ylabel("ω (a/λ)")
            if cand.k_label in kidx:
                xi = kidx[cand.k_label]; yi = freqs[xi, min(cand.band_index, Nb-1)]
                ax.scatter([xi],[yi], s=50, zorder=5)
            ax.set_title(f"{name} bands — {cand.pol} — band {cand.band_index}", fontsize=9)
    else:
        for i in range(3):
            ax = fig.add_subplot(gs[2,i])
            ax.text(0.5,0.5,"MPB unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    text = (f"Candidate — lattice={cand.lattice}, pol={cand.pol}, r={cand.r:.4f}, ε_bg={cand.eps_bg:.3f}, "
            f"k={cand.k_label}, band={cand.band_index}, twist={angle_deg:.2f}°")
    fig.suptitle(text, fontsize=13, fontweight='bold')
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.06, wspace=0.35, hspace=0.45)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Stage 2 — moiré composition visuals + stacking cells + band diagrams")
    ap.add_argument("--seeds", default="seeds_stage1_monolayer.csv", help="CSV created by Stage 1")
    ap.add_argument("--outdir", default="stage2_moire_outputs", help="Output directory for figures")
    ap.add_argument("--angle", type=float, default=5.0, help="Twist angle in degrees (default: 5°)")
    ap.add_argument("--max-cands", type=int, default=8, help="Max candidates to render")
    ap.add_argument("--resolution", type=int, default=48, help="MPB spatial resolution (pixels/a)")
    ap.add_argument("--bands", type=int, default=10, help="Number of bands for registry band diagrams")
    # NEW: parallelism control
    ap.add_argument("--jobs", type=int, default=0, help="Number of worker processes (0=auto, 1=serial)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if not os.path.isdir(args.outdir): os.makedirs(args.outdir, exist_ok=True)

    cands = read_seeds_csv(args.seeds, limit=args.max_cands)
    if not cands:
        log.error("No candidates found in %s", args.seeds); sys.exit(2)

    total = len(cands)
    # Resolve jobs
    jobs = args.jobs
    if jobs <= 0:
        try:
            ncpu = cpu_count()
        except Exception:
            ncpu = 2
        jobs = max(1, min(max(1, ncpu - 1), total))
    else:
        jobs = max(1, min(jobs, total))

    log.info("Rendering %d candidates with %d job(s)...", total, jobs)

    progress = None; task = None
    if RICH:
        progress = Progress(
            TextColumn("[bold cyan]Stage 2"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
            console=console,
        )
        task = progress.add_task("render", total=total)
        progress.start()

    try:
        if jobs == 1:
            # Serial path (original behavior)
            for i, cand in enumerate(cands, start=1):
                out_png = os.path.join(args.outdir, f"stage2_{i:03d}_{cand.lattice}_k{cand.k_label}_b{cand.band_index}.png")
                try:
                    figure_for_candidate(cand, angle_deg=args.angle, out_png=out_png,
                                         resolution=args.resolution, num_bands=args.bands)
                    log.info("Saved %s", out_png)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    log.warning("Rendering failed for candidate %d: %s", i, e)
                if progress and task is not None: progress.advance(task)
        else:
            # Parallel path: one process per candidate
            tasks = [(i, cand, float(args.angle), int(args.resolution), int(args.bands), args.outdir)
                     for i, cand in enumerate(cands, start=1)]
            _init_pool_env()
            with Pool(processes=jobs, initializer=_init_pool_env) as pool:
                for i, out_png, ok, err in pool.imap_unordered(_render_worker, tasks, chunksize=1):
                    if ok:
                        log.info("Saved %s", out_png)
                    else:
                        log.warning("Rendering failed for candidate %d: %s", i, err)
                    if progress and task is not None:
                        progress.advance(task)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(130)
    finally:
        if progress:
            try: progress.stop()
            except Exception: pass

if __name__ == "__main__":
    main()
