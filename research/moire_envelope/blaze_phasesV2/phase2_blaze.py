"""
Phase 2 (BLAZE): Prepare Envelope-Approximation Data — V2 Pipeline

This is the BLAZE-pipeline version of Phase 2 for the V2 coordinate system.
It reads V2 Phase 1 data (fractional coordinates) and prepares the fields
for BLAZE Phase 3's EA solver.

KEY V2 CHANGES from legacy BLAZE Phase 2:
1. Input is in fractional coordinates (s1, s2) ∈ [0,1)² — s_grid is primary
2. Transformed mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ for fractional Laplacian
3. Grid spacing: ds1 = 1/Ns1, ds2 = 1/Ns2 (uniform on unit square)
4. Stores pipeline_version="V2", coordinate_system="fractional"

Unlike the MPB Phase 2 which assembles a sparse operator for scipy eigsh,
this version prepares the raw fields (V, M̃⁻¹, s_grid) that BLAZE's EA solver
will use to construct its own internal operator.

Note: The BLAZE EA solver uses the convention H = V + (η²/2) ∇·M̃⁻¹∇ (positive kinetic),
which differs from the legacy convention H = V - (η²/2) ∇·M̃⁻¹∇ (negative kinetic).

Based on README_V2.md and phasesV2/phase2_ea_operator.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, cast

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_json, load_yaml, save_json
from common.plotting import plot_phase2_fields


def log(msg: str):
    """Simple logging helper."""
    print(msg, flush=True)


# =============================================================================
# Mass Tensor Transformation (V2 Key Feature)
# =============================================================================

def transform_mass_tensor_to_fractional(
    M_inv: np.ndarray, 
    B_moire: np.ndarray
) -> np.ndarray:
    """
    Transform mass tensor from Cartesian to fractional coordinates.
    
    M̃⁻¹(s) = B⁻¹ M⁻¹(s) B⁻ᵀ
    
    This is the key V2 change: the Laplacian ∇_R · M⁻¹ ∇_R becomes
    ∇_s · M̃⁻¹ ∇_s when R = B·s.
    
    Args:
        M_inv: [Ns1, Ns2, 2, 2] inverse mass tensor in Cartesian coords
        B_moire: [2, 2] moiré basis matrix (columns = A1, A2)
    
    Returns:
        M_inv_tilde: [Ns1, Ns2, 2, 2] transformed mass tensor for fractional Laplacian
    """
    B_inv = np.linalg.inv(B_moire)
    # M̃⁻¹[i,j] = B_inv @ M⁻¹[i,j] @ B_inv.T
    # Using einsum: 'ia,...ab,jb->...ij' contracts properly
    return np.einsum('ia,...ab,jb->...ij', B_inv, M_inv, B_moire.T @ np.linalg.inv(B_moire @ B_moire.T) @ B_inv.T)
    # Simpler direct version:
    # return np.einsum('ia,...ab,jb->...ij', B_inv, M_inv, B_inv)


def transform_mass_tensor_simple(
    M_inv: np.ndarray, 
    B_moire: np.ndarray,
    B_mono: np.ndarray | None = None,
    k_units: str = "physical",
) -> np.ndarray:
    """
    Transform mass tensor from Cartesian k to fractional real-space coordinates.
    
    The full transformation depends on the input k-units:
    
    If k_units == "physical" (M_inv computed with physical k in 1/length):
        M̃⁻¹ = B_moire⁻¹ M⁻¹ B_moire⁻ᵀ
    
    If k_units == "fractional" (M_inv computed with fractional k as in BLAZE):
        M⁻¹_phys = G⁻ᵀ M⁻¹_frac G⁻¹  (where G = 2π B_mono⁻ᵀ is reciprocal basis)
        M̃⁻¹ = B_moire⁻¹ M⁻¹_phys B_moire⁻ᵀ
        
        Combined: M̃⁻¹ = (1/(2π)²) B_moire⁻¹ B_mono M⁻¹_frac B_monoᵀ B_moire⁻ᵀ
    
    CRITICAL: BLAZE computes ∂²ω/∂k_i∂k_j where k is in fractional reciprocal units
    (normalized to the BZ). This requires the k-unit conversion factor!
    
    Args:
        M_inv: [Ns1, Ns2, 2, 2] inverse mass tensor from Phase 1
        B_moire: [2, 2] moiré basis matrix (columns = A1, A2)
        B_mono: [2, 2] monolayer basis matrix (required if k_units="fractional")
        k_units: "physical" (1/length) or "fractional" (normalized to BZ)
    
    Returns:
        M_inv_tilde: [Ns1, Ns2, 2, 2] transformed mass tensor for fractional Laplacian
    """
    Ns1, Ns2 = M_inv.shape[:2]
    M_inv_tilde = np.zeros_like(M_inv)
    
    if k_units == "fractional":
        if B_mono is None:
            raise ValueError("B_mono is required when k_units='fractional'")
        
        # Convert from fractional-k to physical-k units:
        # G = 2π (B_mono⁻¹)ᵀ  (reciprocal basis matrix)
        # M⁻¹_phys = G⁻ᵀ M⁻¹_frac G⁻¹ = (1/2π)² B_mono M⁻¹_frac B_monoᵀ
        #
        # Then transform to fractional real-space:
        # M̃⁻¹ = B_moire⁻¹ M⁻¹_phys B_moire⁻ᵀ
        #
        # Combined: M̃⁻¹ = (1/(2π)²) B_moire⁻¹ B_mono M⁻¹_frac B_monoᵀ B_moire⁻ᵀ
        
        two_pi_sq_inv = 1.0 / (2.0 * np.pi) ** 2
        B_moire_inv = np.linalg.inv(B_moire)
        
        # Pre-compute the combined transformation matrices
        # Left: (1/2π) B_moire⁻¹ B_mono
        # Right: (1/2π) B_monoᵀ B_moire⁻ᵀ
        left_transform = B_moire_inv @ B_mono
        right_transform = B_mono.T @ B_moire_inv.T
        
        for i in range(Ns1):
            for j in range(Ns2):
                M_inv_tilde[i, j] = two_pi_sq_inv * left_transform @ M_inv[i, j] @ right_transform
        
        log(f"    Applied k-unit conversion: fractional → physical (×1/(2π)² ≈ {two_pi_sq_inv:.4e})")
    else:
        # Physical k-units: just the real-space coordinate transform
        B_inv = np.linalg.inv(B_moire)
        B_inv_T = B_inv.T
        for i in range(Ns1):
            for j in range(Ns2):
                M_inv_tilde[i, j] = B_inv @ M_inv[i, j] @ B_inv_T
    
    return M_inv_tilde


def _regularize_mass_tensor(M_inv: np.ndarray, min_eig: float | None) -> np.ndarray:
    """Clamp tiny curvature eigenvalues to keep the operator elliptic."""
    if min_eig is None or min_eig <= 0:
        return M_inv
    tensors = M_inv.reshape(-1, 2, 2)
    adjusted = False
    for idx, tensor in enumerate(tensors):
        eigvals, eigvecs = np.linalg.eigh(tensor)
        mask = np.abs(eigvals) < min_eig
        if not np.any(mask):
            continue
        adjusted = True
        eigvals[mask] = np.sign(eigvals[mask]) * min_eig
        # Handle zero eigenvalues
        eigvals = np.where(eigvals == 0, min_eig, eigvals)
        tensors[idx] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return tensors.reshape(M_inv.shape) if adjusted else M_inv


def transform_vg_to_fractional(vg: np.ndarray, B_moire: np.ndarray) -> np.ndarray:
    """
    Transform group velocity from Cartesian to fractional coordinates.
    
    v_g^tilde = B^{-1} v_g
    """
    B_inv = np.linalg.inv(B_moire)
    return np.einsum('ij,...j->...i', B_inv, vg)


# =============================================================================
# Envelope Width Estimation
# =============================================================================

def estimate_envelope_width(
    V: np.ndarray,
    M_inv_tilde: np.ndarray,
    eta: float,
    s_grid: np.ndarray,
    B_moire: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate expected envelope mode width based on harmonic approximation.
    
    For a 1D harmonic oscillator with potential V(x) = V₀ + ½K(x-x₀)²
    and kinetic term T = -η²/(2M) d²/dx², the ground state width is:
    
        L_env ~ (η² / (M⁻¹ K))^{1/4}
    
    This function:
    1. Finds the minimum of V (the cavity center)
    2. Computes the Hessian of V at that point (stiffness K)
    3. Gets the local M̃⁻¹ at that point
    4. Estimates L_env in fractional and physical units
    
    This helps validate that the envelope theory predictions match observations.
    
    Args:
        V: [Ns1, Ns2] potential field in fractional coordinates
        M_inv_tilde: [Ns1, Ns2, 2, 2] transformed mass tensor
        eta: physics small parameter (a/L_m)
        s_grid: [Ns1, Ns2, 2] fractional coordinate grid
        B_moire: [2, 2] moiré basis matrix
    
    Returns:
        Dict with envelope width estimates and diagnostics
    """
    Ns1, Ns2 = V.shape
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    
    # Find V minimum location
    min_idx = np.unravel_index(np.argmin(V), V.shape)
    i_min, j_min = min_idx
    V_min = V[i_min, j_min]
    s_min = s_grid[i_min, j_min]
    
    # Compute V Hessian at minimum (second derivative = stiffness K)
    # Using central differences with periodic wrapping
    def wrap(i, N):
        return i % N
    
    # K_11 = ∂²V/∂s₁²
    V_pp = V[wrap(i_min + 1, Ns1), j_min]
    V_mm = V[wrap(i_min - 1, Ns1), j_min]
    K_11 = (V_pp - 2 * V_min + V_mm) / (ds1 ** 2)
    
    # K_22 = ∂²V/∂s₂²
    V_pp = V[i_min, wrap(j_min + 1, Ns2)]
    V_mm = V[i_min, wrap(j_min - 1, Ns2)]
    K_22 = (V_pp - 2 * V_min + V_mm) / (ds2 ** 2)
    
    # K_12 = ∂²V/∂s₁∂s₂
    V_pp = V[wrap(i_min + 1, Ns1), wrap(j_min + 1, Ns2)]
    V_pm = V[wrap(i_min + 1, Ns1), wrap(j_min - 1, Ns2)]
    V_mp = V[wrap(i_min - 1, Ns1), wrap(j_min + 1, Ns2)]
    V_mm = V[wrap(i_min - 1, Ns1), wrap(j_min - 1, Ns2)]
    K_12 = (V_pp - V_pm - V_mp + V_mm) / (4 * ds1 * ds2)
    
    K = np.array([[K_11, K_12], [K_12, K_22]])
    K_eigvals = np.linalg.eigvalsh(K)
    K_eff = np.sqrt(np.abs(K_eigvals[0] * K_eigvals[1]))  # Geometric mean
    K_max = np.max(np.abs(K_eigvals))
    K_min = np.min(np.abs(K_eigvals))
    
    # Get local M̃⁻¹ at minimum
    M_tilde_local = M_inv_tilde[i_min, j_min]
    M_eigvals = np.linalg.eigvalsh(M_tilde_local)
    M_eff = np.sqrt(np.abs(M_eigvals[0] * M_eigvals[1]))  # Geometric mean
    M_max = np.max(np.abs(M_eigvals))
    M_min = np.min(np.abs(M_eigvals))
    
    # Estimate envelope width in fractional coordinates
    # L_env ~ (η² / (M⁻¹_eff × K_eff))^{1/4}
    eta_sq = eta ** 2
    if M_eff > 0 and K_eff > 0:
        L_env_frac = (eta_sq / (M_eff * K_eff)) ** 0.25
    else:
        L_env_frac = np.nan
    
    # Convert to physical length
    # In fractional coords, L_env_frac is in units of the unit square [0,1)
    # Physical length: L_env_phys = L_env_frac × L_m (approx)
    L_m = np.sqrt(np.abs(np.linalg.det(B_moire)))  # Approximate moiré length
    L_env_phys = L_env_frac * L_m
    
    # Expected participation ratio (area occupied / total area)
    # PR ~ L_env_frac² for a 2D Gaussian-like mode
    expected_PR_frac = L_env_frac ** 2 if not np.isnan(L_env_frac) else np.nan
    
    # Sanity check: if L_env_frac > 1, mode extends beyond one unit cell
    # If L_env_frac < ds1, mode is sub-grid (problematic!)
    is_well_resolved = (L_env_frac > 2 * max(ds1, ds2)) if not np.isnan(L_env_frac) else False
    is_localized = (L_env_frac < 0.5) if not np.isnan(L_env_frac) else False
    
    return {
        "V_min": float(V_min),
        "V_min_location_s1": float(s_min[0]),
        "V_min_location_s2": float(s_min[1]),
        "K_eff": float(K_eff),
        "K_max": float(K_max),
        "K_min": float(K_min),
        "M_inv_tilde_eff": float(M_eff),
        "M_inv_tilde_max": float(M_max),
        "M_inv_tilde_min": float(M_min),
        "eta": float(eta),
        "eta_squared": float(eta_sq),
        "L_env_fractional": float(L_env_frac),
        "L_env_physical": float(L_env_phys),
        "L_m_estimate": float(L_m),
        "expected_PR_fraction": float(expected_PR_frac),
        "is_well_resolved": bool(is_well_resolved),
        "is_localized": bool(is_localized),
        "grid_resolution_ds1": float(ds1),
        "grid_resolution_ds2": float(ds2),
    }


# =============================================================================
# Grid Metrics (V2: Fractional Coordinates)
# =============================================================================

def _grid_metrics_fractional(s_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics for fractional coordinate system (unit square)."""
    Ns1, Ns2, _ = s_grid.shape
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    return {
        "Ns1": Ns1,
        "Ns2": Ns2,
        "ds1": ds1,
        "ds2": ds2,
        "s1_min": 0.0,
        "s1_max": 1.0 - ds1,
        "s2_min": 0.0,
        "s2_max": 1.0 - ds2,
        "cell_area": 1.0,  # Unit square in fractional coords
    }


def _grid_metrics_cartesian(R_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics for Cartesian coordinates (for compatibility)."""
    Nx, Ny, _ = R_grid.shape
    dx = np.linalg.norm(R_grid[1, 0] - R_grid[0, 0]) if Nx > 1 else 1.0
    dy = np.linalg.norm(R_grid[0, 1] - R_grid[0, 0]) if Ny > 1 else 1.0
    return {
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
    }


# =============================================================================
# Supercell Tiling (V2: Fractional Coordinates)
# =============================================================================

def _tile_fields_fractional(
    s_grid: np.ndarray,
    V: np.ndarray,
    M_inv_tilde: np.ndarray,
    vg_tilde: np.ndarray | None,
    tiles: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Tile fields for supercell calculation in fractional coordinates.
    
    For (Tx, Ty) tiling, the new unit square covers Tx×Ty moiré cells.
    The fractional coordinates are rescaled: s' = s/T for each dimension.
    """
    Tx, Ty = tiles
    if Tx == 1 and Ty == 1:
        return s_grid, V, M_inv_tilde, vg_tilde
    
    Ns1, Ns2 = s_grid.shape[:2]
    Ns1_new = Ns1 * Tx
    Ns2_new = Ns2 * Ty
    
    # Tile the fields (values are periodic)
    V_tiled = np.tile(V, (Tx, Ty))
    M_inv_tiled = np.tile(M_inv_tilde, (Tx, Ty, 1, 1))
    vg_tiled = np.tile(vg_tilde, (Tx, Ty, 1)) if vg_tilde is not None else None
    
    # Build new s_grid for the supercell
    s1_new = np.arange(Ns1_new) / Ns1_new
    s2_new = np.arange(Ns2_new) / Ns2_new
    S1, S2 = np.meshgrid(s1_new, s2_new, indexing='ij')
    s_grid_new = np.stack([S1, S2], axis=-1)
    
    return s_grid_new, V_tiled, M_inv_tiled, vg_tiled


# =============================================================================
# Run Directory Resolution
# =============================================================================

def resolve_blaze_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete BLAZE Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        # V2: look in runsV2 directory for BLAZE runs only
        base = Path(config.get("output_dir", "runsV2"))
        blaze_runs = sorted(base.glob("phase0_blaze_*"))
        if not blaze_runs:
            raise FileNotFoundError(
                f"No BLAZE phase0 run directories found in {base}\n"
                f"  (Looking for phase0_blaze_*)\n"
                f"  Found MPB runs? Use explicit path instead."
            )
        run_dir = blaze_runs[-1]
        log(f"Auto-selected latest BLAZE Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runsV2")) / run_dir
        if candidate.exists():
            run_dir = candidate
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


def _discover_blaze_phase1_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that contain BLAZE Phase 1 V2 data."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # Check for Phase 1 band data
        h5_path = cdir / "phase1_band_data.h5"
        if h5_path.exists():
            # Verify it's V2 data
            try:
                with h5py.File(h5_path, 'r') as hf:
                    version = hf.attrs.get('pipeline_version', 'V1')
                    if version == 'V2':
                        discovered.append((cid, cdir))
                    else:
                        log(f"  Skipping {cdir.name}: not V2 pipeline data (version={version})")
            except Exception as e:
                log(f"  Warning: Could not read {h5_path}: {e}")
    return discovered


def _load_blaze_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    """Load per-candidate metadata JSON, falling back to CSV if necessary."""
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception as exc:
            log(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return {"candidate_id": candidate_id}


# =============================================================================
# Candidate Processing (V2)
# =============================================================================

def process_candidate_v2(
    row: Dict,
    config: Dict,
    run_dir: Path,
    plot_fields: bool,
):
    """
    Process a single candidate through BLAZE Phase 2 V2.
    
    Key V2 changes:
    - Reads s_grid (fractional) as primary grid
    - Transforms M_inv to M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ
    - Stores V, M̃⁻¹ on unit square for BLAZE EA solver
    
    Note: BLAZE constructs its own operator internally in Phase 3.
    """
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data for candidate {cid}: {h5_path}")

    log(f"  Candidate {cid}: loading BLAZE Phase 1 V2 data from {h5_path}")
    
    # Load V2 Phase 1 data
    with h5py.File(h5_path, "r") as hf:
        # V2: s_grid is primary
        s_grid = np.asarray(cast(h5py.Dataset, hf["s_grid"])[:])
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])  # For visualization
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        vg_field = np.asarray(cast(h5py.Dataset, hf["vg"])[:]) if "vg" in hf else None
        
        # V2 attributes
        eta = float(hf.attrs.get("eta", 1.0))
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))
        B_moire = np.asarray(hf.attrs.get("B_moire")) if "B_moire" in hf.attrs else None
        B_mono = np.asarray(hf.attrs.get("B_mono")) if "B_mono" in hf.attrs else None
        theta_rad = float(hf.attrs.get("theta_rad", 0.0))
        pipeline_version = str(hf.attrs.get("pipeline_version", "V1"))

    if pipeline_version != "V2":
        log(f"    WARNING: Phase 1 data is not V2 (version={pipeline_version})")
    
    log(f"    Using η = {eta:.6f}")
    log(f"    Grid shape: {s_grid.shape[:2]}")
    
    # Get B_moire for mass tensor transformation
    if B_moire is None:
        # Fall back: try to reconstruct from metadata
        log("    WARNING: B_moire not found in Phase 1 attrs; using identity")
        B_moire = np.eye(2)
    
    # Get B_mono for k-unit conversion (CRITICAL for BLAZE!)
    if B_mono is None:
        log("    WARNING: B_mono not found in Phase 1 attrs; using identity")
        B_mono = np.eye(2)
    
    log(f"    B_moire = [[{B_moire[0,0]:.4f}, {B_moire[0,1]:.4f}], [{B_moire[1,0]:.4f}, {B_moire[1,1]:.4f}]]")
    log(f"    B_mono = [[{B_mono[0,0]:.4f}, {B_mono[0,1]:.4f}], [{B_mono[1,0]:.4f}, {B_mono[1,1]:.4f}]]")

    # === Transform mass tensor to fractional coordinates ===
    # CRITICAL: BLAZE computes M_inv in fractional k-units (normalized to BZ)
    # We must convert: M̃⁻¹ = (1/(2π)²) B_moire⁻¹ B_mono M⁻¹_frac B_monoᵀ B_moire⁻ᵀ
    M_inv_tilde = transform_mass_tensor_simple(
        M_inv, B_moire, B_mono=B_mono, k_units="fractional"
    )
    log(f"    Transformed M_inv to M̃⁻¹ with k-unit correction for fractional Laplacian")

    # Transform v_g if present
    vg_tilde = transform_vg_to_fractional(vg_field, B_moire) if vg_field is not None else None

    # Regularize transformed mass tensor
    min_mass_eig = config.get("phase2_min_mass_eig")
    M_inv_tilde = _regularize_mass_tensor(M_inv_tilde, min_mass_eig)

    # Handle supercell tiling
    tiles_cfg = config.get("phase2_supercell_tiles", [1, 1])
    if isinstance(tiles_cfg, int):
        tiles = (int(tiles_cfg), int(tiles_cfg))
    elif isinstance(tiles_cfg, (list, tuple)) and len(tiles_cfg) == 2:
        tiles = (int(tiles_cfg[0]), int(tiles_cfg[1]))
    else:
        raise ValueError("phase2_supercell_tiles must be an int or [Tx, Ty] list")
    if tiles[0] < 1 or tiles[1] < 1:
        raise ValueError("phase2_supercell_tiles entries must be ≥ 1")

    if tiles != (1, 1):
        log(f"    Tiling Phase 1 fields into a {tiles[0]}×{tiles[1]} supercell")
        s_grid, V, M_inv_tilde, vg_tilde = _tile_fields_fractional(
            s_grid, V, M_inv_tilde, vg_tilde, tiles
        )
        # Scale B_moire for supercell
        B_moire_scaled = B_moire * np.array([[tiles[0], 0], [0, tiles[1]]])
    else:
        B_moire_scaled = B_moire

    grid_info = _grid_metrics_fractional(s_grid)
    log(f"    Prepared V2 data for BLAZE EA: grid {int(grid_info['Ns1'])}×{int(grid_info['Ns2'])}")

    # === Save prepared data for BLAZE Phase 3 ===
    h5_out = cdir / "phase2_blaze_data.h5"
    with h5py.File(h5_out, "w") as hf:
        # V2: s_grid is primary
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")  # For visualization
        hf.create_dataset("V", data=V, compression="gzip")
        hf.create_dataset("M_inv_tilde", data=M_inv_tilde, compression="gzip")  # Transformed!
        hf.create_dataset("M_inv", data=M_inv, compression="gzip")  # Original for reference
        # NOTE: vg is not passed to BLAZE EA solver - commenting out to avoid confusion
        # if vg_tilde is not None:
        #     hf.create_dataset("vg_tilde", data=vg_tilde, compression="gzip")
        #     hf.create_dataset("vg", data=vg_field, compression="gzip")  # Original
        
        # V2 attributes
        hf.attrs["eta"] = eta
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["tiles_x"] = tiles[0]
        hf.attrs["tiles_y"] = tiles[1]
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_moire_scaled"] = B_moire_scaled
        if B_mono is not None:
            hf.attrs["B_mono"] = B_mono
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["Ns1"] = int(grid_info['Ns1'])
        hf.attrs["Ns2"] = int(grid_info['Ns2'])
        hf.attrs["ds1"] = grid_info['ds1']
        hf.attrs["ds2"] = grid_info['ds2']
        hf.attrs["pipeline_version"] = "V2"
        hf.attrs["coordinate_system"] = "fractional"
        hf.attrs["mass_tensor_transform"] = "M_tilde = B_inv @ M @ B_inv.T"
        
    log(f"    Saved V2 prepared data to {h5_out}")

    # Compute field statistics for V2
    eigvals_M = np.linalg.eigvalsh(M_inv.reshape(-1, 2, 2))
    eigvals_M_tilde = np.linalg.eigvalsh(M_inv_tilde.reshape(-1, 2, 2))
    
    # === Operator diagnostics (estimate what BLAZE will build) ===
    Ns1, Ns2 = int(grid_info["Ns1"]), int(grid_info["Ns2"])
    matrix_size = Ns1 * Ns2
    
    # For a 5-point stencil Laplacian on a periodic 2D grid:
    # Each row has ~13 entries (center + 4 neighbors for 2nd deriv + 4 for mixed + 4 for 1st deriv corrections)
    # For finite difference: typically 5-9 entries per row depending on order
    estimated_nnz_per_row = 13  # Conservative estimate for 2nd-order FD with position-dependent mass
    estimated_nnz = matrix_size * estimated_nnz_per_row
    estimated_density = 100.0 * estimated_nnz / (matrix_size ** 2)
    
    # Check mass tensor symmetry at each grid point (should be symmetric for Hermitian operator)
    M_tilde_flat = M_inv_tilde.reshape(-1, 2, 2)
    symmetry_defect = M_tilde_flat - np.transpose(M_tilde_flat, (0, 2, 1))
    max_symmetry_defect = float(np.abs(symmetry_defect).max())
    mean_symmetry_defect = float(np.abs(symmetry_defect).mean())
    
    # Estimate diagonal range: V + (η²/2) * Tr(M̃⁻¹) * (diagonal Laplacian term)
    # For periodic FD: diagonal contribution from ∇²  is roughly -2*(1/ds1² + 1/ds2²)
    ds1, ds2 = grid_info["ds1"], grid_info["ds2"]
    laplacian_diag_scale = 2.0 * (1.0/ds1**2 + 1.0/ds2**2)
    
    # Approximate diagonal: V(s) + (η²/2) * trace(M̃⁻¹) * laplacian_diag_scale
    M_tilde_trace = np.trace(M_inv_tilde, axis1=-2, axis2=-1)  # (Ns1, Ns2)
    diag_kinetic_contrib = (eta**2 / 2.0) * M_tilde_trace * laplacian_diag_scale
    diag_estimate = V + diag_kinetic_contrib
    
    # Positive-definiteness check for mass tensor
    min_mass_eig = float(eigvals_M_tilde.min())
    is_positive_definite = min_mass_eig > 0
    
    # === Envelope Width Estimation ===
    # This helps validate that the k-unit correction is working correctly
    envelope_est = estimate_envelope_width(V, M_inv_tilde, eta, s_grid, B_moire)
    log(f"    Envelope width estimate: L_env ≈ {envelope_est['L_env_fractional']:.3f} (fractional)")
    log(f"    Envelope width estimate: L_env ≈ {envelope_est['L_env_physical']:.2f} (physical)")
    log(f"    Expected PR fraction: {envelope_est['expected_PR_fraction']:.2%}")
    if not envelope_est['is_well_resolved']:
        log(f"    WARNING: Mode may not be well-resolved (L_env < 2×grid spacing)")
    
    stats = {
        "candidate_id": cid,
        "Ns1": Ns1,
        "Ns2": Ns2,
        "ds1": grid_info["ds1"],
        "ds2": grid_info["ds2"],
        "eta": eta,
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "V_mean": float(V.mean()),
        "V_std": float(V.std()),
        # Original M_inv stats
        "M_inv_min_eig": float(eigvals_M.min()),
        "M_inv_max_eig": float(eigvals_M.max()),
        "M_inv_mean_eig": float(eigvals_M.mean()),
        # Transformed M̃⁻¹ stats
        "M_inv_tilde_min_eig": float(eigvals_M_tilde.min()),
        "M_inv_tilde_max_eig": float(eigvals_M_tilde.max()),
        "M_inv_tilde_mean_eig": float(eigvals_M_tilde.mean()),
        # Envelope width estimates
        "L_env_fractional": envelope_est['L_env_fractional'],
        "L_env_physical": envelope_est['L_env_physical'],
        "expected_PR_fraction": envelope_est['expected_PR_fraction'],
        "K_eff": envelope_est['K_eff'],
        "M_inv_tilde_local_eff": envelope_est['M_inv_tilde_eff'],
        "is_well_resolved": envelope_est['is_well_resolved'],
        # Basis info
        "B_moire_det": float(np.linalg.det(B_moire)),
        "pipeline_version": "V2",
        # Operator diagnostics
        "matrix_size": matrix_size,
        "estimated_nnz": estimated_nnz,
        "estimated_density_pct": estimated_density,
        "mass_tensor_symmetric": max_symmetry_defect < 1e-12,
        "mass_tensor_symmetry_defect_max": max_symmetry_defect,
        "mass_tensor_symmetry_defect_mean": mean_symmetry_defect,
        "mass_tensor_positive_definite": is_positive_definite,
        "diag_estimate_min": float(diag_estimate.min()),
        "diag_estimate_max": float(diag_estimate.max()),
        "diag_estimate_std": float(diag_estimate.std()),
    }
    if vg_tilde is not None:
        vg_norm = np.linalg.norm(vg_field[..., :2], axis=-1)
        vg_tilde_norm = np.linalg.norm(vg_tilde[..., :2], axis=-1)
        stats.update({
            "vg_norm_min": float(vg_norm.min()),
            "vg_norm_max": float(vg_norm.max()),
            "vg_norm_mean": float(vg_norm.mean()),
            "vg_tilde_norm_min": float(vg_tilde_norm.min()),
            "vg_tilde_norm_max": float(vg_tilde_norm.max()),
            "vg_tilde_norm_mean": float(vg_tilde_norm.mean()),
        })

    info_path = cdir / "phase2_operator_info.csv"
    pd.DataFrame([stats]).to_csv(info_path, index=False)
    log(f"    Wrote stats to {info_path}")

    # Save envelope width estimates separately for easy access
    save_json(envelope_est, cdir / "phase2_envelope_estimate.json")
    log(f"    Wrote envelope estimates to phase2_envelope_estimate.json")

    # Save metadata
    meta = {
        "tiles": {"x": tiles[0], "y": tiles[1]},
        "pipeline": "blaze",
        "pipeline_version": "V2",
        "coordinate_system": "fractional",
        "operator_convention": "positive_kinetic",  # H = V + T
        "mass_tensor_transform": "M_tilde = (1/(2pi)^2) * B_moire_inv @ B_mono @ M @ B_mono.T @ B_moire_inv.T",
        "k_unit_correction": "Applied 1/(2pi)^2 factor to convert from fractional k to physical k",
        "note": "Data prepared for BLAZE EA solver V2 (no prebuilt operator)",
    }
    save_json(meta, cdir / "phase2_operator_meta.json")

    # Write V2 report
    _write_phase2_blaze_report_v2(cdir, cid, grid_info, stats, eta, B_moire, tiles, plot_fields)

    # Optional visualization (use R_grid for Cartesian plotting)
    if plot_fields:
        try:
            plot_phase2_fields(cdir, R_grid, V, M_inv_tilde, vg_tilde)
            log("    Rendered phase2_fields_visualization.png")
        except Exception as e:
            log(f"    Warning: Could not plot fields: {e}")
    else:
        log("    Skipped field visualization (set phase2_plot_fields: true to enable).")

    return {
        "success": True,
        "candidate_id": cid,
        "grid_size": (int(grid_info["Ns1"]), int(grid_info["Ns2"])),
        "pipeline_version": "V2",
    }


def _write_phase2_blaze_report_v2(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    field_stats: Dict,
    eta: float,
    B_moire: np.ndarray,
    tiles: tuple,
    include_plot: bool,
):
    """Emit a human-readable summary of BLAZE Phase 2 V2 results."""
    
    # Extract operator diagnostics
    matrix_size = field_stats.get("matrix_size", 0)
    estimated_nnz = field_stats.get("estimated_nnz", 0)
    estimated_density = field_stats.get("estimated_density_pct", 0.0)
    mass_symmetric = field_stats.get("mass_tensor_symmetric", False)
    symmetry_defect_max = field_stats.get("mass_tensor_symmetry_defect_max", 0.0)
    symmetry_defect_mean = field_stats.get("mass_tensor_symmetry_defect_mean", 0.0)
    mass_pos_def = field_stats.get("mass_tensor_positive_definite", False)
    diag_min = field_stats.get("diag_estimate_min", 0.0)
    diag_max = field_stats.get("diag_estimate_max", 0.0)
    diag_std = field_stats.get("diag_estimate_std", 0.0)
    
    # Envelope width estimates
    L_env_frac = field_stats.get("L_env_fractional", np.nan)
    L_env_phys = field_stats.get("L_env_physical", np.nan)
    expected_PR = field_stats.get("expected_PR_fraction", np.nan)
    K_eff = field_stats.get("K_eff", np.nan)
    M_local_eff = field_stats.get("M_inv_tilde_local_eff", np.nan)
    is_well_resolved = field_stats.get("is_well_resolved", False)
    
    lines = [
        "# Phase 2 BLAZE V2 Data Preparation Report",
        "",
        f"**Candidate**: {candidate_id:04d}",
        "",
        "## Discretization",
        f"- Grid: {int(grid_info['Ns1'])} × {int(grid_info['Ns2'])} points",
        f"- Δs₁ = {grid_info['ds1']:.6f}, Δs₂ = {grid_info['ds2']:.6f}",
        f"- Supercell tiling: {tiles[0]} × {tiles[1]}",
        f"- η (moiré scale ratio) = {eta:.6f}",
        "",
        "## Moiré Basis",
        f"- B_moire:",
        f"  - A₁ = [{B_moire[0, 0]:.4f}, {B_moire[1, 0]:.4f}]",
        f"  - A₂ = [{B_moire[0, 1]:.4f}, {B_moire[1, 1]:.4f}]",
        f"- |det(B_moire)| = {abs(np.linalg.det(B_moire)):.4f} (moiré cell area)",
        "",
        "## Envelope Width Estimate (Harmonic Approximation)",
        f"- L_env (fractional): {L_env_frac:.4f}",
        f"- L_env (physical): {L_env_phys:.2f}",
        f"- Expected PR fraction: {expected_PR:.2%}",
        f"- Potential stiffness K_eff: {K_eff:.4f}",
        f"- Local mass M̃⁻¹_eff: {M_local_eff:.4e}",
        f"- Well-resolved: {'Yes' if is_well_resolved else 'No (may need finer grid)'}",
        "",
        "**Interpretation**: For healthy cavities, L_env should be 0.1-0.5 (fractional),",
        "corresponding to PR ~ 1-25% of the moiré cell. If L_env << grid spacing,",
        "the mode is under-resolved and results may be unreliable.",
        "",
        "## Input Field Diagnostics",
        f"- Potential V(s): min {field_stats['V_min']:.6f}, max {field_stats['V_max']:.6f}, mean {field_stats['V_mean']:.6f}",
        f"- M⁻¹ eigenvalues (original): min {field_stats['M_inv_min_eig']:.4f}, max {field_stats['M_inv_max_eig']:.4f}",
        f"- M̃⁻¹ eigenvalues (transformed): min {field_stats['M_inv_tilde_min_eig']:.6e}, max {field_stats['M_inv_tilde_max_eig']:.6e}",
        "",
        "## K-Unit Correction Applied",
        "- BLAZE computes derivatives in fractional k-units (normalized to BZ)",
        "- Correction factor: 1/(2π)² ≈ 2.53×10⁻²",
        "- This scales down M̃⁻¹ by ~40×, leading to ~2.5× larger envelope widths",
        "",
        "## Operator Diagnostics",
        f"- Matrix size: {matrix_size} × {matrix_size}",
        f"- Estimated non-zeros: {estimated_nnz} ({estimated_density:.3f}% density)",
        f"- Diagonal range (estimate): [{diag_min:.6f}, {diag_max:.6f}]",
        f"- Diagonal std dev: {diag_std:.6f}",
        "",
        "## Hermitian Checks",
        f"- Mass tensor M̃⁻¹ symmetric: {'pass' if mass_symmetric else 'FAIL'}",
        f"  - max |M̃⁻¹ - M̃⁻ᵀ| = {symmetry_defect_max:.3e}",
        f"  - mean |M̃⁻¹ - M̃⁻ᵀ| = {symmetry_defect_mean:.3e}",
        f"- Mass tensor M̃⁻¹ positive definite: {'pass' if mass_pos_def else 'FAIL'}",
        f"  - min eigenvalue = {field_stats['M_inv_tilde_min_eig']:.6e}",
        "",
        "**Hermiticity Note**: For real-valued V(s) and symmetric M̃⁻¹(s), the operator",
        "H = V + (η²/2) ∇·M̃⁻¹∇ is self-adjoint (Hermitian) with real eigenvalues.",
        "",
        "## BLAZE Pipeline Notes",
        "- Coordinate system: Fractional (unit square [0,1)²)",
        "- Transformed mass tensor: M̃⁻¹ = (1/(2π)²) B⁻¹ B_mono M⁻¹ B_monoᵀ B⁻ᵀ",
        "- Convention: H = V + (η²/2) ∇·M̃⁻¹∇ (positive kinetic term)",
        "- No sparse operator is prebuilt; BLAZE constructs it internally",
        "",
    ]

    if include_plot:
        lines.append("- Field visualization saved alongside this report.")
    else:
        lines.append("- Field visualization skipped (set phase2_plot_fields: true to enable).")

    report_path = Path(cdir) / "phase2_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log(f"    Wrote {report_path}")


# =============================================================================
# Main Runner
# =============================================================================

def run_phase2_blaze_v2(run_dir: str | Path, config: Dict):
    """
    Run BLAZE Phase 2 V2 on all candidates in a run directory.
    
    Args:
        run_dir: Path to the BLAZE run directory (or 'auto'/'latest')
        config: Configuration dictionary
    """
    log("\n" + "=" * 70)
    log("PHASE 2 (BLAZE V2): Data Preparation for EA Solver")
    log("=" * 70)
    log("Coordinate system: Fractional (unit square [0,1)²)")
    log("Mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ")
    log("")

    plot_fields = bool(config.get("phase2_plot_fields", False))

    run_dir = resolve_blaze_run_dir(run_dir, config)
    log(f"Using run directory: {run_dir}")

    # Load candidate list if available
    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    else:
        log(f"WARNING: {candidates_path} not found; relying solely on per-candidate metadata.")

    # Discover candidates with Phase 1 V2 data
    discovered = _discover_blaze_phase1_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with BLAZE Phase 1 V2 data found in {run_dir}. "
            "Run Phase 1 (BLAZE V2) before Phase 2."
        )

    # Optionally limit number of candidates
    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        discovered = discovered[:K]
        log(f"Processing {len(discovered)} candidate(s) (limited to K={K}).")
    else:
        log(f"Processing {len(discovered)} candidate(s).")

    results = []
    for cid, _ in discovered:
        row = _load_blaze_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            result = process_candidate_v2(
                row,
                config,
                run_dir,
                plot_fields,
            )
            results.append(result)
        except FileNotFoundError as exc:
            log(f"  Skipping candidate {cid}: {exc}")
        except Exception as exc:
            log(f"  ERROR processing candidate {cid}: {exc}")
            import traceback
            traceback.print_exc()

    log(f"\nPhase 2 (BLAZE V2) completed: {len(results)}/{len(discovered)} candidates processed.\n")
    return results


def run_phase2(run_dir: str | Path, config_path: str | Path):
    """Entry point for command-line usage."""
    config = load_yaml(config_path)
    return run_phase2_blaze_v2(run_dir, config)


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 2 V2 config."""
    return PROJECT_ROOT / "configsV2" / "phase2_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest BLAZE run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2("auto", default_config)
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2(sys.argv[1], default_config)
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase2(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python blaze_phasesV2/phase2_blaze.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest BLAZE run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
