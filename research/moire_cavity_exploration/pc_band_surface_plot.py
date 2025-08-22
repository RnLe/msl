"""
3D band surface plot over the Brillouin zone
-------------------------------------------
Generates 3D surface plots of all bands within the first Brillouin zone
for a 2D photonic crystal consisting of air holes in a dielectric background.

- Lattice types: triangular | square | rectangular | oblique
- Parameters:
    * a: lattice constant
    * rectangular: rho = b/a
    * oblique: rho=b/a and gamma (degrees)
- Geometry: air cylinder holes (radius r*a) in background eps_bg.
- Polarization: tm | te | both (two figures when both)

Dependencies: meep/mpb, numpy, matplotlib, moire_lattice_py (for BZ polygon)

Usage:
  Modify the parameters in the CONFIGURATION SECTION within main() and run:
  python pc_band_surface_plot.py

Examples of configurations:
  # Triangular lattice
  lattice_type = "triangular"
  a = 1.0; r = 0.25; eps_bg = 12.0; polarization = "tm"
  
  # Rectangular lattice  
  lattice_type = "rectangular" 
  rho = 1.3; r = 0.22; eps_bg = 10.0; polarization = "both"
  
  # Oblique lattice
  lattice_type = "oblique"
  rho = 1.2; gamma = 100.0; r = 0.23; eps_bg = 12.5
"""

import os
import math
from typing import Any, Dict, Tuple, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
import matplotlib.tri as mtri

# MPB / Meep
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
ml: Any = None
_ml_ok = True
try:
    import moire_lattice_py as _ml  # BZ polygon + reciprocal basis
    ml = _ml
except Exception:
    _ml_ok = False
    ml = None


def lattice_basis_from_params(lattice_type: str, a: float, shape: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Return (a1, a2, mp.Lattice) using lattice_type + optional 'shape' params.
    shape keys:
      rectangular: {'rho': ...}
      oblique:     {'rho': ..., 'gamma_deg': ...}
    """
    lt = lattice_type.lower()
    if lt == "triangular":
        a1 = np.array([a, 0.0])
        a2 = np.array([0.5 * a, (math.sqrt(3)/2) * a])
    elif lt == "square":
        a1 = np.array([a, 0.0])
        a2 = np.array([0.0, a])
    elif lt == "rectangular":
        rho = 1.5 if shape is None else float(shape.get("rho", 1.5))
        a1 = np.array([a, 0.0])
        a2 = np.array([0.0, rho * a])
    elif lt == "oblique":
        rho = 1.2 if shape is None else float(shape.get("rho", 1.2))
        gamma_deg = 100.0 if shape is None else float(shape.get("gamma_deg", 100.0))
        gamma = math.radians(gamma_deg)
        a1 = np.array([a, 0.0])
        a2 = rho * a * np.array([math.cos(gamma), math.sin(gamma)])
    else:
        raise ValueError(f"Unknown lattice_type: {lattice_type}")

    lat = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(float(a1[0]), float(a1[1]), 0),
                     basis2=mp.Vector3(float(a2[0]), float(a2[1]), 0))
    return a1, a2, lat


def build_ml_lattice(lattice_type: str, a: float, shape: Optional[Dict[str, float]] = None):
    if not _ml_ok or ml is None:
        raise RuntimeError("moire_lattice_py not available; required for BZ mesh.")
    _mod = ml  # type: ignore[assignment]
    lt = lattice_type.lower()
    if lt == "triangular":
        return _mod.PyLattice2D("triangular", a=a)
    if lt == "square":
        return _mod.PyLattice2D("square", a=a)
    if lt == "rectangular":
        rho = 1.5 if shape is None else float(shape.get("rho", 1.5))
        return _mod.PyLattice2D("rectangular", a=a, b=rho * a)
    if lt == "oblique":
        rho = 1.2 if shape is None else float(shape.get("rho", 1.2))
        gamma_deg = 100.0 if shape is None else float(shape.get("gamma_deg", 100.0))
        return _mod.PyLattice2D("oblique", a=a, b=rho * a, angle=gamma_deg)
    raise ValueError(f"Unknown lattice_type: {lattice_type}")


def _bz_polygon_fractional(lattice: Any) -> np.ndarray:
    """Return BZ polygon in fractional reciprocal coordinates given ml.PyLattice2D lattice."""
    # Obtain reciprocal basis (2D) columns b1,b2 in physical coords
    if hasattr(lattice, "inner"):
        if hasattr(lattice.inner, "reciprocal"):
            B = np.array(lattice.inner.reciprocal)
        else:
            basis = lattice.inner.reciprocal_basis()
            B = np.array([list(basis[0]), list(basis[1]), list(basis[2])]).T
        if hasattr(lattice.inner, "brillouin_zone"):
            bz = lattice.inner.brillouin_zone
            bz_vertices = bz.vertices if hasattr(bz, "vertices") else bz.vertices()
        else:
            bz_vertices = lattice.inner.brillouin_zone().vertices()
    else:
        basis = lattice.reciprocal_basis()
        B = np.array([list(basis[0]), list(basis[1]), list(basis[2])]).T
        bz_vertices = lattice.brillouin_zone().vertices()

    B2 = B[:2, :2]
    verts: List[np.ndarray] = []
    for v in bz_vertices:
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
    from matplotlib.path import Path as MplPath
    mask = MplPath(poly_frac).contains_points(pts)
    pts_in = pts[mask]
    return pts_in, (gx, gy)


def run_mpb_mesh(lattice_type: str, a: float, r: float, eps_bg: float, shape: Optional[Dict[str, float]], N: int,
                 num_bands: int, resolution: int, polarization: str) -> Dict[str, Any]:
    if not MEEP_AVAILABLE:
        raise RuntimeError("Meep / MPB not available.")
    if not _ml_ok:
        raise RuntimeError("moire_lattice_py not available; required for BZ mesh.")

    # BZ from ml
    lat_ml = build_ml_lattice(lattice_type, a, shape)
    poly_frac = _bz_polygon_fractional(lat_ml)

    # mp lattice and geometry
    a1, a2, lat = lattice_basis_from_params(lattice_type, a, shape)
    geom = [mp.Cylinder(radius=r * a, material=mp.air, center=mp.Vector3(0, 0, 0))]

    # Grid inside polygon (fractional coords)
    pts_frac, _axes = _mesh_inside_polygon(poly_frac, N)
    kpts = [mp.Vector3(float(px), float(py), 0) for (px, py) in pts_frac]

    ms_kwargs = dict(
        geometry_lattice=lat,
        geometry=geom,
        default_material=mp.Medium(epsilon=eps_bg),
        k_points=kpts,
        resolution=resolution,
        num_bands=num_bands,
        dimensions=2,
    )
    ms = mpb.ModeSolver(**ms_kwargs)

    pol = polarization.lower()
    out: Dict[str, Any] = {}
    if pol == "tm":
        ms.run_tm(); out["freqs"] = np.array(ms.all_freqs)
    elif pol == "te":
        ms.run_te(); out["freqs"] = np.array(ms.all_freqs)
    elif pol == "both":
        ms.run_tm(); freqs_tm = np.array(ms.all_freqs)
        ms.reset_meep(); ms = mpb.ModeSolver(**ms_kwargs); ms.run_te(); freqs_te = np.array(ms.all_freqs)
        out["freqs_tm"] = freqs_tm; out["freqs_te"] = freqs_te
    else:
        raise ValueError("polarization must be 'tm', 'te' or 'both'")

    out.update({
        "k_frac": pts_frac,     # [nk,2]
        "a1": a1, "a2": a2,
        "poly_frac": poly_frac,
        "shape": shape or {},
    })
    return out


def plot_3d_surfaces(k_frac: np.ndarray, freqs: np.ndarray, title: str, outfile: str):
    """Plot all bands as 3D trisurfaces over fractional kx,ky."""
    kx = k_frac[:, 0]
    ky = k_frac[:, 1]
    triang = mtri.Triangulation(kx, ky)

    nb = freqs.shape[1]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.95, max(nb, 1)))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for b in range(nb):
        z = freqs[:, b]
        ax.plot_trisurf(triang, z, color=colors[b], alpha=0.65, linewidth=0.2, antialiased=True)

    ax.set_xlabel("k (fractional b1)")
    ax.set_ylabel("k (fractional b2)")
    ax.set_zlabel("Frequency (c/a)")
    ax.set_title(title)
    # a bit nicer view
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)
    plt.close(fig)


def main():
    # =============================================================================
    # CONFIGURATION SECTION - MODIFY THESE PARAMETERS
    # =============================================================================
    
    # Lattice type: "triangular", "square", "rectangular", "oblique"
    lattice_type = "square"
    
    # Basic parameters
    a = 1.0                    # lattice constant
    r = 0.43                   # hole radius in units of a
    eps_bg = 4.02              # background epsilon
    
    # Shape parameters (only needed for certain lattice types)
    rho = None                 # b/a ratio (for rectangular/oblique, e.g., 1.3)
    gamma = None               # angle in degrees (for oblique, e.g., 100.0)
    
    # MPB calculation parameters
    num_bands = 8              # number of bands to calculate
    resolution = 40            # MPB resolution (pixels/a)
    polarization = "tm"        # "tm", "te", or "both"
    
    # Mesh parameters
    N = 35                     # grid resolution across BZ box
    
    # Output
    output_dir = "pc_band_surfaces"
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================

    if not MEEP_AVAILABLE:
        raise SystemExit("Meep / MPB not available. Install meep and try again.")
    if not _ml_ok:
        raise SystemExit("moire_lattice_py not available (needed for BZ mesh). Build/install rust-python package.")

    shape: Optional[Dict[str, float]] = None
    if lattice_type == "rectangular":
        if rho is None:
            raise SystemExit("rho parameter required for rectangular lattice")
        shape = {"rho": float(rho)}
    elif lattice_type == "oblique":
        if rho is None or gamma is None:
            raise SystemExit("rho and gamma parameters required for oblique lattice")
        shape = {"rho": float(rho), "gamma_deg": float(gamma)}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running MPB mesh: {lattice_type} | a={a} r/a={r} eps={eps_bg} bands={num_bands} pol={polarization} N={N}")
    res = run_mpb_mesh(lattice_type, a, r, eps_bg, shape, N, num_bands, resolution, polarization)

    # Plot per polarization
    base = f"{lattice_type}_a{a:.3f}_r{r:.3f}_eps{eps_bg:.2f}"
    if shape:
        if "rho" in shape: base += f"_rho{shape['rho']:.3f}"
        if "gamma_deg" in shape: base += f"_g{shape['gamma_deg']:.1f}"

    if polarization == "both":
        for pol_name, freqs in [("TM", res["freqs_tm"]), ("TE", res["freqs_te"])]:
            title = f"{lattice_type} | {pol_name}"
            outfile = os.path.join(output_dir, f"{base}_{pol_name}_3D.png")
            plot_3d_surfaces(res["k_frac"], freqs, title, outfile)
            print(f"Saved {outfile}")
    else:
        pol_upper = polarization.upper()
        title = f"{lattice_type} | {pol_upper}"
        outfile = os.path.join(output_dir, f"{base}_{pol_upper}_3D.png")
        plot_3d_surfaces(res["k_frac"], res["freqs"], title, outfile)
        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
