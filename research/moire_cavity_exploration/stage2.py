#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 — Envelope solve on the moiré cell + Q-proxies
======================================================

Implements "Stage B — Envelope stage" from instructions.md:
- Build V(r) on one moiré cell from AA/AB band-edge offsets (ω_edge - ω0).
- Use the Stage-1 Hessian H = ∂²ω/∂k_i∂k_j at k0 as M^{-1} (ħ=1 units).
- Solve:  [-1/2 ∇·(M^{-1})∇  + V(r)] F = Δω F
- Extract Δω_n, |F|², participation, mode volume surrogate, WKB leakage proxy.
- Assemble Stage-B objective J2 and export a ranked CSV.

Design is aligned with stage1_prefilter.py (Rich logging/progress, figures).
Comments are in English and avoid 2nd-person phrasing.

References:
- Pipeline definitions and envelope equation in instructions.md.  (cites)
- Stage-1 export columns and logging style mirrored.               (cites)
"""
from __future__ import annotations

import os, sys, math, csv, logging, argparse, contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from typing import Any as _Any  # local alias to help with optional types

# ============================== Output / Logging (Rich) ==========================================

OUT_DIR = os.path.join(os.path.dirname(__file__), "stage2_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "run.log")

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.logging import RichHandler
    RICH = True
except Exception:
    Console = None  # type: ignore
    Progress = None  # type: ignore
    BarColumn = None  # type: ignore
    TextColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    Table = None  # type: ignore
    RichHandler = None  # type: ignore
    RICH = False

handlers: List[logging.Handler] = []
fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.DEBUG); handlers.append(fh)
if RICH and RichHandler is not None:  # type: ignore[truthy-bool]
    ch = RichHandler(level=logging.INFO, rich_tracebacks=True, markup=True)  # type: ignore[operator]
else:
    ch = logging.StreamHandler(stream=sys.stdout); ch.setLevel(logging.INFO)
handlers.append(ch)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
log = logging.getLogger("stage2")
console = Console() if (RICH and Console is not None) else None  # type: ignore[call-overload]

# ============================== Optional libs (visuals / eigensolver) ============================

plt = None  # type: ignore
try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB = True
except Exception as e:
    MATPLOTLIB = False
    log.warning("Matplotlib not importable; plots will be skipped. %s", e)

SCIPY_OK = True
coo_matrix = None  # type: ignore
csr_matrix = None  # type: ignore
eigsh = None  # type: ignore
try:
    from scipy.sparse import coo_matrix, csr_matrix  # type: ignore
    from scipy.sparse.linalg import eigsh  # type: ignore
except Exception as e:
    SCIPY_OK = False
    log.error("SciPy sparse/eigsh required for envelope solve. %s", e)

ML_AVAILABLE = False
try:
    # Optional helper for moiré geometry (if present, used for registry coordinates)
    import moire_lattice_py as ml  # type: ignore
    ML_AVAILABLE = True
except Exception as e:
    log.info("moire_lattice_py not importable; using cosine model for V(r). %s", e)

# ============================== Physics/Stage-2 CONFIG ==========================================

CONFIG: Dict[str, Any] = dict(
    # Domain and grid (one moiré supercell; can repeat if desired)
    domain = dict(
        repeats_xy = (1, 1),     # integer repeats of one moiré cell (x,y)
        Nx = 128, Ny = 128,      # grid resolution per **total** domain
        bc = "dirichlet",       # boundary condition for envelope: "dirichlet" or "periodic"
    ),
    # Moiré potential model
    potential = dict(
        model = "cos2",          # "cos2" (square superposition) or "binary" (if moire_lattice_py available)
        smooth = True,           # if binary, Gaussian smoothing may be applied (σ as fraction of L)
        sigma_frac = 0.05,
    ),
    # Mass handling
    mass = dict(
        use_stage1_Hessian = True,  # uses Hxx, Hyy, Hxy as M^{-1}
        hbar = 1.0,                 # units scaling (Stage-1 curvatures assumed in ω-units with ħ=1)
        force_positive_kinetic = True,  # if H is negative-definite (band maximum), flip signs: H→-H, V→-V
    ),
    # Objective (Stage-B)
    objective = dict(
        w_dVe     = 1.0,  # local well depth around chosen minimum
        w_invV    = 0.8,  # 1 / mode-volume surrogate
        w_WKB     = 0.7,  # WKB in-plane leakage exponent
        w_light   = 0.0,  # placeholder (light-line margin not computed here)
        w_bic     = 0.0,  # placeholder (symmetry BIC score not computed here)
    ),
    # Visualization
    visuals = dict(
        plot_top = 4,            # number of top seeds to visualize
        cmap_potential = "viridis",
        cmap_mode = "magma",
    ),
)

# ============================== Data classes =====================================================

@dataclass
class Seed:
    # Geometry / lattice (kept for reference; stage-2 uses only twist and stage-1 metrics)
    a1x: float; a1y: float; a2x: float; a2y: float
    twist_deg: float
    # Stage-1 selection
    polarization: str; band_index: int; k_label: str
    # Stage-1 scalars
    omega0: float
    omega_AA: float; omega_AB: float
    Hxx: float; Hyy: float; Hxy: float
    J2_stage1: float

@dataclass
class EnvelopeResult:
    # Envelope spectrum and metrics
    domega0: float
    part_ratio: float
    mode_volume: float
    wkb_S: float
    # Objective
    J2: float
    # Book-keeping
    seed_idx: int
    status: str

# ============================== Utilities ========================================================

def read_seeds_csv(path: str, K: Optional[int]) -> List[Seed]:
    """Read seeds from Stage-1 CSV, sort by Stage-1 J2 descending, take top-K."""
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    # filter rows that have all required keys
    req = ["a1x","a1y","a2x","a2y","twist_deg","polarization","band_index","k_label",
           "omega0","omega_AA","omega_AB","Hxx","Hyy","Hxy","J2"]
    sel = []
    for r in rows:
        if any(k not in r or r[k] == "" for k in req): 
            continue
        try:
            # Base required scalars
            a1x = float(r['a1x']); a1y = float(r['a1y'])
            a2x = float(r['a2x']); a2y = float(r['a2y'])
            twist_deg = float(r.get('twist_deg', 1.10))
            omega0 = float(r['omega0'])
            omega_AA = float(r['omega_AA']); omega_AB = float(r['omega_AB'])
            J2 = float(r['J2'])
            # Early finite checks on base scalars
            if not all(np.isfinite(v) for v in (a1x,a1y,a2x,a2y,twist_deg,omega0,omega_AA,omega_AB,J2)):
                continue
            # Try Hessian directly
            Hxx = float(r.get('Hxx', 'nan'))
            Hyy = float(r.get('Hyy', 'nan'))
            Hxy = float(r.get('Hxy', 'nan'))
            if not (np.isfinite(Hxx) and np.isfinite(Hyy)):
                # Try reconstruct from directional curvatures if available
                kx = r.get('kappa_x'); ky = r.get('kappa_y'); kd = r.get('kappa_diag')
                kx = float(kx) if kx not in (None, "",) else math.nan
                ky = float(ky) if ky not in (None, "",) else math.nan
                kd = float(kd) if kd not in (None, "",) else math.nan
                if np.isfinite(kx) and np.isfinite(ky):
                    Hxx = kx; Hyy = ky
                    if np.isfinite(kd):
                        # kd = (Hxx + 2 Hxy + Hyy)/2  => Hxy = (2*kd - Hxx - Hyy)/2
                        Hxy = 0.5*(2.0*kd - Hxx - Hyy)
                    else:
                        Hxy = 0.0
                else:
                    # Fallback to isotropic using alpha if finite
                    alpha = r.get('alpha')
                    aval = float(alpha) if alpha not in (None, "",) else math.nan
                    if np.isfinite(aval):
                        Hxx = aval; Hyy = aval; Hxy = 0.0
                    else:
                        # Final conservative default
                        Hxx = 1.0e-2; Hyy = 1.0e-2; Hxy = 0.0
            # Ensure finite Hessian now
            if not all(np.isfinite(v) for v in (Hxx,Hyy,Hxy)):
                continue

            s = Seed(
                a1x=a1x, a1y=a1y,
                a2x=a2x, a2y=a2y,
                twist_deg=twist_deg,
                polarization=r['polarization'],
                band_index=int(r['band_index']),
                k_label=r['k_label'],
                omega0=omega0,
                omega_AA=omega_AA, omega_AB=omega_AB,
                Hxx=Hxx, Hyy=Hyy, Hxy=Hxy,
                J2_stage1=J2,
            )
            sel.append(s)
        except Exception:
            continue
    sel.sort(key=lambda s: s.J2_stage1, reverse=True)
    return sel[:K] if K is not None else sel

def moire_length_square(a: float, theta_deg: float) -> float:
    """Approximate moiré period for two identical square lattices (real-space a=|a1|) at twist θ."""
    theta = math.radians(theta_deg)
    s = max(1e-9, math.sin(0.5*theta))
    return a / (2.0 * s)

def build_potential_cos2(Lx: float, Ly: float, Nx: int, Ny: int, V_AA: float, V_AB: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cosine model on a square moiré cell:
        V(r) = V_avg + (Δ/4) [cos(Gx x) + cos(Gy y)]
    Ensures extremes at ±Δ/2 around V_avg match AA/AB offsets.
    """
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Vavg = 0.5*(V_AA + V_AB)
    dlt = (V_AB - V_AA)
    Gx = 2.0*math.pi / Lx
    Gy = 2.0*math.pi / Ly
    V = Vavg + 0.25*dlt*(np.cos(Gx*X) + np.cos(Gy*Y))
    return X, Y, V

def gaussian_blur_periodic(arr: np.ndarray, sigma_frac: float) -> np.ndarray:
    """Approximate Gaussian smoothing with FFT assuming periodic boundaries."""
    Ny, Nx = arr.shape
    sx = max(1e-9, sigma_frac) * Nx
    sy = max(1e-9, sigma_frac) * Ny
    A = np.fft.rfftn(arr)
    kx = np.fft.rfftfreq(Nx) * 2.0*math.pi
    ky = np.fft.fftfreq(Ny) * 2.0*math.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    # Gaussian in Fourier domain: exp(-0.5 * (σ_x^2 kx^2 + σ_y^2 ky^2))
    G = np.exp(-0.5*((KX*sx/Nx)**2 + (KY*sy/Ny)**2))
    out = np.fft.irfftn(A*G, s=arr.shape)
    return np.real(out)

def effective_mass_scalar_from_Minv(Minv: np.ndarray) -> float:
    """
    Construct an isotropic effective mass for WKB from tensor M^{-1}.
    For isotropic ∝ I with Minv = ħ²/m, m = ħ² / Minv. In 2D, take harmonic-like mean:
        m_eff ≈ 2 / trace(Minv)
    """
    tr = float(np.trace(Minv))
    return float(2.0 / max(tr, 1e-12))

# ============================== Discretization of the operator ===================================

def assemble_operator(Minv: np.ndarray, V: np.ndarray, Lx: float, Ly: float, bc: str = "periodic") -> _Any:
    """
    Assemble sparse matrix for operator:  H_en = -1/2 ∑_{ij} Minv_{ij} ∂_i∂_j + V
    bc: 'periodic' wraps indices; 'dirichlet' enforces zero outside the domain (no wrapping).
    Mixed derivatives use a standard 9-point stencil:
      ∂x∂y f ≈ (f_{i+1,j+1} - f_{i+1,j-1} - f_{i-1,j+1} + f_{i-1,j-1}) / (4 dx dy)
    """
    Ny, Nx = V.shape
    dx = Lx / Nx
    dy = Ly / Ny
    Mxx, Myy, Mxy = float(Minv[0,0]), float(Minv[1,1]), float(Minv[0,1])  # symmetric

    # Coefficients
    cx = -0.5 * Mxx / (dx*dx)
    cy = -0.5 * Myy / (dy*dy)
    cxy = -0.5 * Mxy / (4.0*dx*dy)  # applied to four diagonal neighbors with ± signs

    nnz_est = Nx*Ny*9
    I = np.empty(nnz_est, dtype=np.int64)
    J = np.empty(nnz_est, dtype=np.int64)
    Vals = np.empty(nnz_est, dtype=np.float64)
    ptr = 0

    def lin(i: int, j: int) -> int:
        return j*Nx + i  # row-major, x fastest

    def in_bounds(ii: int, jj: int) -> bool:
        return (0 <= ii < Nx) and (0 <= jj < Ny)

    for j in range(Ny):
        jm = j-1; jp = j+1
        for i in range(Nx):
            im = i-1; ip = i+1
            row = lin(i,j)

            # center
            diag_val = (-2.0*cx - 2.0*cy) + V[j, i]
            I[ptr] = row; J[ptr] = row; Vals[ptr] = diag_val; ptr += 1

            # x neighbors
            if bc == "periodic":
                I[ptr] = row; J[ptr] = lin((i-1) % Nx, j); Vals[ptr] = cx; ptr += 1
                I[ptr] = row; J[ptr] = lin((i+1) % Nx, j); Vals[ptr] = cx; ptr += 1
            else:  # dirichlet: only add if inside
                if in_bounds(im, j):
                    I[ptr] = row; J[ptr] = lin(im, j); Vals[ptr] = cx; ptr += 1
                if in_bounds(ip, j):
                    I[ptr] = row; J[ptr] = lin(ip, j); Vals[ptr] = cx; ptr += 1

            # y neighbors
            if bc == "periodic":
                I[ptr] = row; J[ptr] = lin(i, (j-1) % Ny); Vals[ptr] = cy; ptr += 1
                I[ptr] = row; J[ptr] = lin(i, (j+1) % Ny); Vals[ptr] = cy; ptr += 1
            else:
                if in_bounds(i, jm):
                    I[ptr] = row; J[ptr] = lin(i, jm); Vals[ptr] = cy; ptr += 1
                if in_bounds(i, jp):
                    I[ptr] = row; J[ptr] = lin(i, jp); Vals[ptr] = cy; ptr += 1

            # mixed neighbors
            if abs(Mxy) > 0.0:
                def add_mixed(ii, jj, coef):
                    nonlocal ptr
                    if bc == "periodic":
                        I[ptr] = row; J[ptr] = lin(ii % Nx, jj % Ny); Vals[ptr] = coef; ptr += 1
                    else:
                        if in_bounds(ii, jj):
                            I[ptr] = row; J[ptr] = lin(ii, jj); Vals[ptr] = coef; ptr += 1
                add_mixed(ip, jp, +cxy)
                add_mixed(ip, jm, -cxy)
                add_mixed(im, jp, -cxy)
                add_mixed(im, jm, +cxy)

    I = I[:ptr]; J = J[:ptr]; Vals = Vals[:ptr]
    H = coo_matrix((Vals, (I, J)), shape=(Nx*Ny, Nx*Ny)).tocsr()  # type: ignore
    return H

# ============================== Metrics ==========================================================

def normalize_mode(psi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    norm = math.sqrt(float(np.sum(np.abs(psi)**2) * dx * dy))
    return psi / max(norm, 1e-18)

def participation_ratio(psi: np.ndarray, dx: float, dy: float) -> float:
    num = float(np.sum(np.abs(psi)**2) * dx * dy)**2
    den = float(np.sum(np.abs(psi)**4) * dx * dy)
    return num / max(den, 1e-18)

def mode_volume_surrogate(psi: np.ndarray, dx: float, dy: float) -> float:
    """A simple surrogate (units of area): inverse of peak intensity times normalization area."""
    peak = float(np.max(np.abs(psi)**2))
    area = float(psi.size * dx * dy)
    return area / max(peak, 1e-18)

def wkb_leakage_exponent(V: np.ndarray, E: float, Minv: np.ndarray, Lx: float, Ly: float) -> float:
    """
    1D WKB along axis-aligned least-action cuts from the deepest minimum to the boundary.
    Uses isotropic m_eff derived from Minv for simplicity.
    S ≈ ∫ sqrt(2 m_eff (V(x)-E)) dx over classically forbidden region (V>E).
    """
    Ny, Nx = V.shape
    j0, i0 = np.unravel_index(np.argmin(V), V.shape)
    m_eff = effective_mass_scalar_from_Minv(Minv)

    def path_integral_along_x(j):
        vals = V[j, :]
        dx = Lx / Nx
        S = 0.0
        # forward
        in_forbidden = False
        for i in range(i0+1, Nx):
            if vals[i] > E:
                in_forbidden = True
                S += math.sqrt(max(0.0, 2.0*m_eff*(vals[i]-E))) * dx
            else:
                if in_forbidden:
                    break
        # backward
        in_forbidden = False
        for i in range(i0-1, -1, -1):
            if vals[i] > E:
                in_forbidden = True
                S += math.sqrt(max(0.0, 2.0*m_eff*(vals[i]-E))) * dx
            else:
                if in_forbidden:
                    break
        return S

    def path_integral_along_y(i):
        vals = V[:, i]
        dy = Ly / Ny
        S = 0.0
        in_forbidden = False
        for j in range(j0+1, Ny):
            if vals[j] > E:
                in_forbidden = True
                S += math.sqrt(max(0.0, 2.0*m_eff*(vals[j]-E))) * dy
            else:
                if in_forbidden:
                    break
        in_forbidden = False
        for j in range(j0-1, -1, -1):
            if vals[j] > E:
                in_forbidden = True
                S += math.sqrt(max(0.0, 2.0*m_eff*(vals[j]-E))) * dy
            else:
                if in_forbidden:
                    break
        return S

    Sx = path_integral_along_x(j0)
    Sy = path_integral_along_y(i0)
    return float(max(Sx, Sy))

# ============================== Plotting =========================================================

def plot_potential_and_mode(out_png: str, X: np.ndarray, Y: np.ndarray, V: np.ndarray, psi: np.ndarray):
    if not MATPLOTLIB:
        return
    Ny, Nx = V.shape
    fig = plt.figure(figsize=(9.5, 4.6))  # type: ignore
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.05, 1.0])
    ax0 = fig.add_subplot(gs[0,0]); ax1 = fig.add_subplot(gs[0,1])

    im0 = ax0.pcolormesh(X, Y, V, shading="auto", cmap=CONFIG["visuals"]["cmap_potential"])
    ax0.set_title("Moiré potential V(r)"); ax0.set_aspect("equal", "box")
    fig.colorbar(im0, ax=ax0, shrink=0.9)

    dens = np.abs(psi.reshape(Ny, Nx))**2
    im1 = ax1.pcolormesh(X, Y, dens, shading="auto", cmap=CONFIG["visuals"]["cmap_mode"])
    ax1.set_title("Envelope |F|² (ground state)"); ax1.set_aspect("equal", "box")
    fig.colorbar(im1, ax=ax1, shrink=0.9)

    for ax in (ax0, ax1):
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)  # type: ignore

# ============================== Stage-B objective ================================================

def stageB_objective(dVe: float, invV: float, S_wkb: float,
                     light_margin: float = 0.0, bic_score: float = 0.0) -> float:
    w = CONFIG["objective"]
    return (w["w_dVe"] * dVe + w["w_invV"] * invV +
            w["w_WKB"] * S_wkb + w["w_light"] * light_margin + w["w_bic"] * bic_score)

# ============================== Core per-seed computation ========================================

def process_seed(seed: Seed, idx: int, args) -> Tuple[Optional[EnvelopeResult], Optional[Dict[str, Any]]]:
    """
    Build V(r), assemble H_en, compute lowest mode and metrics for one seed.
    Returns metrics dict for CSV export in addition to EnvelopeResult for ranking.
    """
    if not SCIPY_OK:
        log.error("SciPy is required to run Stage 2. Skipping seed #%d", idx)
        return None, None
    # 1) Moiré cell size
    a_len = math.hypot(seed.a1x, seed.a1y)  # stage-1 used a1=(1,0) → a=1; kept general
    L = moire_length_square(a_len, seed.twist_deg)  # Stage-B Step 5.  (instructions.md)
    Rx, Ry = CONFIG["domain"]["repeats_xy"]
    Lx, Ly = L*Rx, L*Ry
    Nx, Ny = int(args.Nx or CONFIG["domain"]["Nx"]), int(args.Ny or CONFIG["domain"]["Ny"])

    # 2) Potential from Stage-1 edges (V = ω_edge - ω0)
    V_AA = float(seed.omega_AA - seed.omega0)
    V_AB = float(seed.omega_AB - seed.omega0)

    X, Y, V = build_potential_cos2(Lx, Ly, Nx, Ny, V_AA, V_AB)   # smoothed two-level per instructions.  (instructions.md)
    if CONFIG["potential"]["smooth"]:
        V = gaussian_blur_periodic(V, CONFIG["potential"]["sigma_frac"])

    # 3) Mass tensor M^{-1} from Stage-1 Hessian.  (instructions.md)
    Minv = np.array([[seed.Hxx, seed.Hxy],
                     [seed.Hxy, seed.Hyy]], dtype=float)

    # Handle band maximum (negative-definite Hessian): flip signs for positive kinetic form.
    lam = np.linalg.eigvalsh(Minv)
    flipped = False
    if CONFIG["mass"]["force_positive_kinetic"] and (lam[0] < 0 and lam[1] < 0):
        Minv = -Minv
        V = -V
        flipped = True

    # 4) Assemble operator and solve lowest eigenpair
    H = assemble_operator(Minv, V, Lx, Ly, bc=CONFIG["domain"].get("bc", "periodic"))
    # Use algebraically smallest eigenvalue (ground state). 'which=SA' targets smallest algebraic.
    vals, vecs = eigsh(H, k=1, which="SA", tol=1e-6, maxiter=2000)  # type: ignore[arg-type]
    domega = float(vals[0])  # this is Δω in the envelope equation
    psi = vecs[:, 0]
    dx, dy = Lx/Nx, Ly/Ny
    psi = normalize_mode(psi, dx, dy)

    # 5) Metrics
    part = participation_ratio(psi, dx, dy)
    mv = mode_volume_surrogate(psi, dx, dy)

    # Effective local well depth near minimum
    vmin = float(V.min()); dVe = float(max(0.0, (domega - vmin)))  # Δω - V_min
    # WKB leakage proxy (1D axis-aligned least-action)
    Swkb = wkb_leakage_exponent(V, domega, Minv, Lx, Ly)

    invV = 1.0 / max(mv, 1e-18)
    J2 = stageB_objective(dVe, invV, Swkb)

    # 6) Optional plots
    if args.plot and MATPLOTLIB:
        out_png = os.path.join(OUT_DIR, f"env_{idx:03d}_b{seed.band_index}_{seed.k_label}.png")
        plot_potential_and_mode(out_png, X, Y, V, psi)

    # Collect metrics row
    mrow = dict(
        idx=idx,
        polarization=seed.polarization, band_index=seed.band_index, k_label=seed.k_label,
        twist_deg=seed.twist_deg,
        omega0=seed.omega0, omega_AA=seed.omega_AA, omega_AB=seed.omega_AB,
        Hxx=seed.Hxx, Hyy=seed.Hyy, Hxy=seed.Hxy, flipped=int(flipped),
        L_moire=L, Nx=Nx, Ny=Ny,
        domega=domega, Vmin=vmin,
        participation=part, mode_volume=mv, wkb_S=Swkb,
        dVe=dVe, invV=invV, J2=J2,
        J2_stage1=seed.J2_stage1,
    )

    res = EnvelopeResult(domega, part, mv, Swkb, J2, idx, status="ok")
    return res, mrow

# ============================== CSV export =======================================================

def write_stage2_csv(path_csv: str, rows: List[Dict[str, Any]]):
    cols = ["idx","polarization","band_index","k_label","twist_deg",
            "omega0","omega_AA","omega_AB","Hxx","Hyy","Hxy","flipped",
            "L_moire","Nx","Ny","domega","Vmin","participation","mode_volume","wkb_S",
            "dVe","invV","J2","J2_stage1"]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

# ============================== CLI ===============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Stage 2 — Envelope solve and Q-proxies")
    default_seeds = os.path.join(os.path.dirname(__file__), "seeds_stage1_validated.csv")
    if (not os.path.exists(default_seeds)) or os.path.getsize(default_seeds) == 0:
        default_seeds = os.path.join(os.path.dirname(__file__), "seeds_stage1.csv")
    ap.add_argument("--seeds", default=default_seeds, help=f"Path to Stage-1 CSV (default: {default_seeds}).")
    ap.add_argument("--K", type=int, default=12, help="Top-K seeds from Stage-1 to process (by J2).")
    ap.add_argument("--Nx", type=int, help="Grid points in x (over total domain).")
    ap.add_argument("--Ny", type=int, help="Grid points in y (over total domain).")
    ap.add_argument("--plot-top", dest="plot", type=int, default=CONFIG["visuals"]["plot_top"], help="Number of top results to plot.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging to console.")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.verbose and RICH:
        log.setLevel(logging.DEBUG)

    # Load seeds
    seeds = read_seeds_csv(args.seeds, K=args.K)
    if not seeds:
        log.error("No valid Stage-1 seeds were read from %s", args.seeds); sys.exit(2)

    # Progress bar
    progress = None; task = None
    if RICH and Progress is not None and TextColumn is not None and BarColumn is not None and TimeElapsedColumn is not None and console is not None:
        progress = Progress(TextColumn("[bold green]Stage 2"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), transient=False, console=console)
        task = progress.add_task("envelope", total=len(seeds)); progress.start()

    results: List[EnvelopeResult] = []
    rows: List[Dict[str, Any]] = []

    for i, seed in enumerate(seeds, start=1):
        try:
            res, mrow = process_seed(seed, i, args)
            if res is not None and mrow is not None:
                results.append(res); rows.append(mrow)
                # Pretty table row
                if RICH and Table is not None and console is not None:
                    tbl = Table.grid()
                    tbl.add_row(
                        f"[bold]#{i}[/] {seed.polarization} b{seed.band_index} @{seed.k_label} θ={seed.twist_deg:.2f}°   "
                        f"Δω={mrow['domega']:.5f}  PR={mrow['participation']:.3f}  "
                        f"V≈{mrow['mode_volume']:.3e}  S_WKB={mrow['wkb_S']:.3f}  J2={mrow['J2']:.4f}"
                    )
                    console.print(tbl)
        except Exception as e:
            log.warning("Seed #%d failed: %s", i, e)
        if progress is not None and task is not None:
            progress.advance(task)

    if progress is not None:
        progress.stop()

    # Rank and write CSV
    rows.sort(key=lambda r: r["J2"], reverse=True)
    out_csv = os.path.join(OUT_DIR, "seeds_stage2.csv")
    write_stage2_csv(out_csv, rows)
    log.info("Wrote Stage-2 results to %s", out_csv)

    # Plot top-N results (already plotted inside process_seed if --plot-top>0)
    # Here ensure only top-N are plotted (process_seed plotted on-the-fly; this is a no-op by default).

if __name__ == "__main__":
    main()
