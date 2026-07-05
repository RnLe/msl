#!/usr/bin/env python3
"""
Incremental data generation for twist-angle sweep studies.

Solves the envelope equation at the moiré Γ-point (k_s = 0) for each twist
angle, saving per-angle NPZ files. Supports single-band and multi-band
(e.g. Dirac doublet) configurations. Detects existing data and only
computes missing angles.

Usage:
    python generate.py --phase1 <path.h5> [--angles 5 2 1 0.5 ...] [--band 0]
    python generate.py --phase1 <path.h5> --bands 1 2 --n-modes 100

Output:
    data/<case_name>/theta_<angle>.npz   — per-angle solve results
    data/<case_name>/metadata.json       — parameters + list of computed angles
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Add parent paths for Phase 2 imports
sys.path.insert(0, str(Path(__file__).resolve().parent))  # vendored: phase2_blaze_v4 lives alongside
from phase2_blaze_v4 import (
    load_phase1_h5,
    interpolate_to_moire_grid,
    compute_moire_metadata,
    transform_mass_tensor,
    transform_velocity,
    born_huang_metric_factor,
    assemble_hamiltonian,
    compute_sigma,
    solve_envelope,
    eigenvalue_to_frequency,
    compute_mode_stats,
)

# ── Default angles ───────────────────────────────────────────────────────
DEFAULT_ANGLES = [
    5.0, 3.0, 2.0, 1.5, 1.0,
    0.8, 0.6, 0.5, 0.4, 0.3,
    0.2, 0.15, 0.1, 0.08, 0.05,
    0.03, 0.01, 0.005, 0.001,
]


# ── Core functions ───────────────────────────────────────────────────────

def extract_single_band(p1: dict, band: int, Ns: int) -> dict:
    """Extract single-band data from Phase 1 and interpolate to Ns×Ns."""
    n_reg = p1["n_reg"]

    def _interp(field):
        return interpolate_to_moire_grid(field, n_reg, Ns)

    eig_band = p1["eigenvalues"][..., band:band+1]
    eig_interp = _interp(eig_band)

    vx_band = p1["velocity_x"][..., band:band+1, band:band+1]
    vy_band = p1["velocity_y"][..., band:band+1, band:band+1]
    vx_interp = _interp(vx_band)
    vy_interp = _interp(vy_band)

    mxx = _interp(p1["mass_xx"][..., band:band+1, band:band+1])
    mxy = _interp(p1["mass_xy"][..., band:band+1, band:band+1])
    myx = _interp(p1["mass_yx"][..., band:band+1, band:band+1])
    myy = _interp(p1["mass_yy"][..., band:band+1, band:band+1])

    bx = _interp(p1["berry_x"][..., band:band+1, band:band+1]) if "berry_x" in p1 else None
    by = _interp(p1["berry_y"][..., band:band+1, band:band+1]) if "berry_y" in p1 else None
    bh = _interp(p1["born_huang"][..., band:band+1, band:band+1]) if "born_huang" in p1 else None
    sc = _interp(p1["slow_coefficient"][..., band:band+1, band:band+1]) if "slow_coefficient" in p1 else None

    return {
        "eigenvalues": eig_interp,
        "velocity_x": vx_interp, "velocity_y": vy_interp,
        "mass_xx": mxx, "mass_xy": mxy, "mass_yx": myx, "mass_yy": myy,
        "berry_x": bx, "berry_y": by,
        "born_huang": bh, "slow_coefficient": sc,
    }


def extract_multi_band(p1: dict, bands: list[int], Ns: int) -> dict:
    """Extract multi-band data from Phase 1 and interpolate to Ns×Ns.

    Args:
        bands: list of band indices, e.g. [1, 2] for the Dirac doublet
    """
    n_reg = p1["n_reg"]
    b0, b1 = min(bands), max(bands) + 1  # slice range

    def _interp(field):
        return interpolate_to_moire_grid(field, n_reg, Ns)

    eig_interp = _interp(p1["eigenvalues"][..., b0:b1])

    vx_interp = _interp(p1["velocity_x"][..., b0:b1, b0:b1])
    vy_interp = _interp(p1["velocity_y"][..., b0:b1, b0:b1])

    mxx = _interp(p1["mass_xx"][..., b0:b1, b0:b1])
    mxy = _interp(p1["mass_xy"][..., b0:b1, b0:b1])
    myx = _interp(p1["mass_yx"][..., b0:b1, b0:b1])
    myy = _interp(p1["mass_yy"][..., b0:b1, b0:b1])

    bx = _interp(p1["berry_x"][..., b0:b1, b0:b1]) if "berry_x" in p1 else None
    by = _interp(p1["berry_y"][..., b0:b1, b0:b1]) if "berry_y" in p1 else None
    bh = _interp(p1["born_huang"][..., b0:b1, b0:b1]) if "born_huang" in p1 else None
    sc = _interp(p1["slow_coefficient"][..., b0:b1, b0:b1]) if "slow_coefficient" in p1 else None

    return {
        "eigenvalues": eig_interp,
        "velocity_x": vx_interp, "velocity_y": vy_interp,
        "mass_xx": mxx, "mass_xy": mxy, "mass_yx": myx, "mass_yy": myy,
        "berry_x": bx, "berry_y": by,
        "born_huang": bh, "slow_coefficient": sc,
        "Nb": b1 - b0,
    }


def solve_multi_band_angle(
    band_data: dict, theta_deg: float, Ns: int, n_modes: int,
    lattice_type: str = "honeycomb",
) -> dict:
    """Solve the multi-band envelope equation for one twist angle."""
    Nb = band_data["Nb"]
    moire = compute_moire_metadata(lattice_type, 1.0, theta_deg)
    B_moire = moire["B_moire"]
    B_inv = np.linalg.inv(B_moire)
    eta = moire["eta"]

    eig = band_data["eigenvalues"]  # (Ns, Ns, Nb)
    lambda_ref = float(np.mean(eig))

    # Potential: Λ_mn = (λ_n - λ_ref) δ_mn
    Lambda = np.zeros((Ns, Ns, Nb, Nb), dtype=complex)
    for n in range(Nb):
        Lambda[..., n, n] = eig[..., n] - lambda_ref

    v1, v2 = transform_velocity(
        band_data["velocity_x"], band_data["velocity_y"], B_inv,
    )

    M11, M12, M21, M22 = transform_mass_tensor(
        band_data["mass_xx"], band_data["mass_xy"],
        band_data["mass_yx"], band_data["mass_yy"], B_inv,
    )

    A1 = band_data["berry_x"]
    A2 = band_data["berry_y"]

    bh_factor = born_huang_metric_factor(B_moire)
    bh = band_data["born_huang"] * bh_factor if band_data["born_huang"] is not None else None
    sc = band_data["slow_coefficient"] * bh_factor if band_data["slow_coefficient"] is not None else None

    H = assemble_hamiltonian(
        Lambda, v1, v2, M11, M12, M22, A1, A2, bh, sc,
        Ns, Nb,
        include_drift=True, include_kinetic=True,
        include_born_huang=True, include_slow_coeff=True,
        fd_order=4, k_s=(0.0, 0.0),
    )

    sigma, _ = compute_sigma(Lambda, M11, M22, target_idx=0)

    eigenvals, eigenvecs = solve_envelope(H, n_modes, sigma)

    mode_stats = []
    for i in range(n_modes):
        F = eigenvecs[:, i].reshape(Ns, Ns, Nb)
        stats = compute_mode_stats(F, eigenvals[i], lambda_ref)
        stats["mode_index"] = i
        mode_stats.append(stats)

    bandwidth_lambda = float(eigenvals.max() - eigenvals.min())
    freq_min = eigenvalue_to_frequency(eigenvals.min(), lambda_ref)
    freq_max = eigenvalue_to_frequency(eigenvals.max(), lambda_ref)
    bandwidth_freq = freq_max - freq_min

    # Store diagonal potential for all bands
    Lambda_diag = np.stack([Lambda[..., n, n].real for n in range(Nb)], axis=-1)

    return {
        "theta_deg": theta_deg,
        "eta": eta,
        "lambda_ref": lambda_ref,
        "eigenvalues": eigenvals,
        "eigenvectors": eigenvecs,
        "mode_stats": mode_stats,
        "bandwidth_lambda": bandwidth_lambda,
        "bandwidth_freq": bandwidth_freq,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "sigma": sigma,
        "Ns": Ns,
        "Nb": Nb,
        "Lambda": Lambda_diag,  # (Ns, Ns, Nb)
        "B_moire": moire["B_moire"],
    }


def solve_single_angle(
    band_data: dict, theta_deg: float, Ns: int, n_modes: int,
    lattice_type: str = "honeycomb",
) -> dict:
    """Solve the single-band envelope equation for one twist angle."""
    Nb = 1
    moire = compute_moire_metadata(lattice_type, 1.0, theta_deg)
    B_moire = moire["B_moire"]
    B_inv = np.linalg.inv(B_moire)
    eta = moire["eta"]

    eig = band_data["eigenvalues"]  # (Ns, Ns, 1)
    lambda_ref = float(np.mean(eig[..., 0]))

    # Potential: Λ = λ - λ_ref
    Lambda = np.zeros((Ns, Ns, Nb, Nb), dtype=complex)
    Lambda[..., 0, 0] = eig[..., 0] - lambda_ref

    # Velocity → fractional
    v1, v2 = transform_velocity(
        band_data["velocity_x"], band_data["velocity_y"], B_inv,
    )

    # Mass → fractional
    M11, M12, M21, M22 = transform_mass_tensor(
        band_data["mass_xx"], band_data["mass_xy"],
        band_data["mass_yx"], band_data["mass_yy"], B_inv,
    )

    # Berry connection (already in registry/fractional coords)
    A1 = band_data["berry_x"]
    A2 = band_data["berry_y"]

    # Born-Huang & slow: metric correction
    bh_factor = born_huang_metric_factor(B_moire)
    bh = band_data["born_huang"] * bh_factor if band_data["born_huang"] is not None else None
    sc = band_data["slow_coefficient"] * bh_factor if band_data["slow_coefficient"] is not None else None

    # Assemble
    H = assemble_hamiltonian(
        Lambda, v1, v2, M11, M12, M22, A1, A2, bh, sc,
        Ns, Nb,
        include_drift=True, include_kinetic=True,
        include_born_huang=True, include_slow_coeff=True,
        fd_order=4, k_s=(0.0, 0.0),
    )

    # Sigma: target the potential minimum
    sigma, _ = compute_sigma(Lambda, M11, M22, target_idx=0)

    # Solve
    eigenvals, eigenvecs = solve_envelope(H, n_modes, sigma)

    # Mode statistics
    mode_stats = []
    for i in range(n_modes):
        F = eigenvecs[:, i].reshape(Ns, Ns, Nb)
        stats = compute_mode_stats(F, eigenvals[i], lambda_ref)
        stats["mode_index"] = i
        mode_stats.append(stats)

    bandwidth_lambda = float(eigenvals.max() - eigenvals.min())
    freq_min = eigenvalue_to_frequency(eigenvals.min(), lambda_ref)
    freq_max = eigenvalue_to_frequency(eigenvals.max(), lambda_ref)
    bandwidth_freq = freq_max - freq_min

    return {
        "theta_deg": theta_deg,
        "eta": eta,
        "lambda_ref": lambda_ref,
        "eigenvalues": eigenvals,
        "eigenvectors": eigenvecs,
        "mode_stats": mode_stats,
        "bandwidth_lambda": bandwidth_lambda,
        "bandwidth_freq": bandwidth_freq,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "sigma": sigma,
        "Ns": Ns,
        "Nb": Nb,
        "Lambda": Lambda[..., 0, 0].real,  # (Ns, Ns) potential landscape
        "B_moire": moire["B_moire"],
    }


def save_angle_result(result: dict, outpath: Path):
    """Save per-angle result to NPZ."""
    np.savez_compressed(
        outpath,
        theta_deg=result["theta_deg"],
        eta=result["eta"],
        lambda_ref=result["lambda_ref"],
        eigenvalues=result["eigenvalues"],
        eigenvectors=result["eigenvectors"],
        bandwidth_lambda=result["bandwidth_lambda"],
        bandwidth_freq=result["bandwidth_freq"],
        freq_min=result["freq_min"],
        freq_max=result["freq_max"],
        sigma=result["sigma"],
        Ns=result["Ns"],
        Nb=result["Nb"],
        Lambda=result["Lambda"],       # (Ns, Ns) potential
        B_moire=result["B_moire"],     # (2, 2)
        # Mode stats stored as parallel arrays
        mode_frequencies=np.array([s["frequency"] for s in result["mode_stats"]]),
        mode_iprs=np.array([s["ipr"] for s in result["mode_stats"]]),
        mode_spreads=np.array([s["spread"] for s in result["mode_stats"]]),
    )


def load_angle_result(path: Path) -> dict:
    """Load a previously saved angle result from NPZ."""
    d = np.load(path, allow_pickle=False)
    n_modes = len(d["eigenvalues"])
    mode_stats = []
    for i in range(n_modes):
        mode_stats.append({
            "mode_index": i,
            "frequency": float(d["mode_frequencies"][i]),
            "ipr": float(d["mode_iprs"][i]),
            "spread": float(d["mode_spreads"][i]),
        })
    return {
        "theta_deg": float(d["theta_deg"]),
        "eta": float(d["eta"]),
        "lambda_ref": float(d["lambda_ref"]),
        "eigenvalues": d["eigenvalues"],
        "eigenvectors": d["eigenvectors"],
        "bandwidth_lambda": float(d["bandwidth_lambda"]),
        "bandwidth_freq": float(d["bandwidth_freq"]),
        "freq_min": float(d["freq_min"]),
        "freq_max": float(d["freq_max"]),
        "sigma": float(d["sigma"]),
        "Ns": int(d["Ns"]),
        "Nb": int(d["Nb"]),
        "Lambda": d["Lambda"],
        "B_moire": d["B_moire"],
        "mode_stats": mode_stats,
    }


def angle_to_filename(theta_deg: float) -> str:
    """Consistent filename for an angle: theta_5.000.npz."""
    return f"theta_{theta_deg:.3f}.npz"


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Twist-angle sweep data generation")
    parser.add_argument("--phase1", type=Path, required=True, help="Path to Phase 1 HDF5")
    parser.add_argument("--angles", type=float, nargs="*", default=None,
                        help="Twist angles in degrees (default: built-in set)")
    parser.add_argument("--add-angles", type=float, nargs="*", default=None,
                        help="Additional angles to add to existing data")
    parser.add_argument("--band", type=int, default=None, help="Single band index")
    parser.add_argument("--bands", type=int, nargs="*", default=None,
                        help="Multiple band indices (e.g. --bands 1 2 for Dirac doublet)")
    parser.add_argument("--interp-factor", type=int, default=4,
                        help="Interpolation factor for registry grid (default: 4)")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of envelope modes")
    parser.add_argument("--lattice", type=str, default="honeycomb",
                        help="Lattice type (default: honeycomb)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: data/<case_name>/)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if data exists")
    args = parser.parse_args()

    # Determine band configuration
    if args.bands is not None:
        multi_band = True
        bands = sorted(args.bands)
        band_label = f"bands {bands}"
    else:
        multi_band = False
        args.band = args.band if args.band is not None else 0
        bands = [args.band]
        band_label = f"band {args.band}"

    # Determine angles
    angles = sorted(set(args.angles or DEFAULT_ANGLES), reverse=True)
    if args.add_angles:
        angles = sorted(set(angles) | set(args.add_angles), reverse=True)

    # Determine case name from Phase 1 filename
    case_name = args.phase1.stem.replace("_phase1", "")

    # Output directory
    data_dir = args.output_dir or (Path(__file__).resolve().parent / "data" / case_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"ANGLE SWEEP: Data Generation")
    print(f"  Phase 1:    {args.phase1}")
    print(f"  Bands:      {band_label} ({'multi-band' if multi_band else 'single-band'})")
    print(f"  Angles:     {len(angles)} values, {max(angles):.3f}° → {min(angles):.4f}°")
    print(f"  Modes:      {args.n_modes}")
    print(f"  Output:     {data_dir}")
    print("=" * 72)

    # Check which angles already exist
    existing = set()
    for npz in data_dir.glob("theta_*.npz"):
        try:
            d = np.load(npz, allow_pickle=False)
            existing.add(float(d["theta_deg"]))
        except Exception:
            pass

    if not args.force:
        to_compute = [a for a in angles if a not in existing]
        skipped = [a for a in angles if a in existing]
        if skipped:
            print(f"\nSkipping {len(skipped)} existing angles: {skipped}")
    else:
        to_compute = angles

    if not to_compute:
        print("\nAll angles already computed. Use --force to recompute.")
        _update_metadata(data_dir, angles, args, multi_band, bands)
        return

    print(f"\nComputing {len(to_compute)} angles: {to_compute}")

    # Load Phase 1
    print(f"\nLoading Phase 1: {args.phase1}")
    p1 = load_phase1_h5(args.phase1)
    n_reg = p1["n_reg"]
    Ns = n_reg * args.interp_factor
    print(f"  Registry: {n_reg}×{n_reg} → interpolated {Ns}×{Ns}")

    # Extract band data
    if multi_band:
        print(f"Extracting bands {bands} and interpolating...")
        t0 = time.time()
        band_data = extract_multi_band(p1, bands, Ns)
        print(f"  Done in {time.time() - t0:.1f}s")
        Nb = band_data["Nb"]
        for bi, b in enumerate(bands):
            eig = band_data["eigenvalues"][..., bi]
            lam = float(np.mean(eig))
            f_b = math.sqrt(max(lam, 0)) / (2 * math.pi)
            print(f"  Band {b}: λ_ref = {lam:.6f}, f_ref = {f_b:.6f} c/a")
        print(f"  Total Nb = {Nb}, matrix size = {Ns*Ns*Nb}×{Ns*Ns*Nb}")
    else:
        print(f"Extracting band {args.band} and interpolating...")
        t0 = time.time()
        band_data = extract_single_band(p1, args.band, Ns)
        print(f"  Done in {time.time() - t0:.1f}s")
        eig = band_data["eigenvalues"][..., 0]
        lambda_ref = float(np.mean(eig))
        f_ref = math.sqrt(max(lambda_ref, 0)) / (2 * math.pi)
        print(f"  λ_ref = {lambda_ref:.6f}, f_ref = {f_ref:.6f} c/a")

    # Solve each angle
    for i, theta in enumerate(to_compute):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(to_compute)}] θ = {theta}°")
        print(f"{'─'*60}")
        t0 = time.time()
        if multi_band:
            result = solve_multi_band_angle(
                band_data, theta, Ns, args.n_modes, args.lattice,
            )
        else:
            result = solve_single_angle(
                band_data, theta, Ns, args.n_modes, args.lattice,
            )
        elapsed = time.time() - t0

        print(f"  η = {result['eta']:.6f}, L_m = {1.0/result['eta']:.1f} a")
        print(f"  Bandwidth: Δλ = {result['bandwidth_lambda']:.6e}")
        print(f"  Freq range: [{result['freq_min']:.6f}, {result['freq_max']:.6f}]")
        print(f"  Time: {elapsed:.1f}s")

        # Save
        outpath = data_dir / angle_to_filename(theta)
        save_angle_result(result, outpath)
        print(f"  Saved: {outpath.name}")

    # Update metadata
    _update_metadata(data_dir, angles, args, multi_band, bands)

    print(f"\n{'='*72}")
    print(f"Done. Data saved to: {data_dir}")


def _update_metadata(data_dir: Path, angles: list[float], args,
                     multi_band: bool = False, bands: list[int] | None = None):
    """Write/update metadata.json with all computed angles."""
    # Scan what's actually on disk
    computed = []
    for npz in sorted(data_dir.glob("theta_*.npz")):
        try:
            d = np.load(npz, allow_pickle=False)
            computed.append({
                "theta_deg": float(d["theta_deg"]),
                "eta": float(d["eta"]),
                "bandwidth_lambda": float(d["bandwidth_lambda"]),
                "bandwidth_freq": float(d["bandwidth_freq"]),
                "n_modes": len(d["eigenvalues"]),
                "Ns": int(d["Ns"]),
                "file": npz.name,
            })
        except Exception:
            pass

    computed.sort(key=lambda x: x["theta_deg"], reverse=True)

    meta = {
        "phase1_path": str(args.phase1),
        "interp_factor": args.interp_factor,
        "n_modes": args.n_modes,
        "lattice_type": args.lattice,
        "requested_angles": sorted(angles, reverse=True),
        "computed": computed,
        "n_computed": len(computed),
    }

    if multi_band and bands is not None:
        meta["band_indices"] = bands
        meta["multi_band"] = True
    else:
        meta["band_index"] = args.band
        meta["multi_band"] = False

    meta_path = data_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated metadata: {meta_path}")


if __name__ == "__main__":
    main()
