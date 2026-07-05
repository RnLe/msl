#!/usr/bin/env python3
"""
S1: Single-R-point MPB Audit
==============================
Pick representative R-points and verify:
  S1.1  ω matches stored values
  S1.2  Bloch function orthonormality (before and after SVQB)
  S1.3  Bloch function spatial symmetry at C4-symmetric registry
  S1.4  Band ordering consistency (do bands swap?)
  S1.5  k-stencil FD vs analytic derivatives

We load existing data and run targeted checks — NO re-running MPB here,
just inspecting the stored data for internal consistency.
"""

import sys, json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "plots"
OUTDIR.mkdir(exist_ok=True)

BASE  = Path(__file__).resolve().parent.parent
CAND  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
P1    = CAND / "phase1_multiband_data.h5"
P2    = CAND / "phase2_multiband_data.h5"

SUBSPACE = [5, 6, 7, 8, 9]
TARGET   = 2  # index in subspace (band 7)


def load_data():
    """Load Phase 1 + 2 data."""
    d = {}
    with h5py.File(P1, 'r') as f:
        d['omega']        = f['omega'][:]           # (128,128,5)
        d['vg']           = f['vg'][:]              # (128,128,5,2)
        d['M_inv']        = f['M_inv'][:]           # (128,128,5,2,2)
        d['bloch_fields'] = f['bloch_fields'][:]    # (64,64,18,32,32,3) complex64
        d['epsilon']      = f['epsilon'][:]         # (64,64,32,32)
        d['s_grid']       = f['s_grid'][:]          # (128,128,2)
        d['Ns1']          = int(f.attrs['Ns1'])
        d['Ns2']          = int(f.attrs['Ns2'])
        d['L_m']          = float(f.attrs['moire_length'])
        d['omega_ref']    = float(f.attrs['omega_ref'])

    with h5py.File(P2, 'r') as f:
        d['Lambda']  = f['Lambda'][:]     # (128,128,5,5)
        d['A_berry'] = f['A_berry'][:]    # (128,128,5,5,2) complex128
        d['M_inv_2'] = f['M_inv'][:]      # (128,128,5,5,2,2)
        d['v_drift'] = f['v_drift'][:]    # (128,128,5,5,2)
        d['Phi_BH']  = f['Phi_BH'][:]    # (128,128,5,5)
    return d


# ══════════════════════════════════════════════════════════════════════════
# S1.1: Check ω values make sense
# ══════════════════════════════════════════════════════════════════════════

def check_omega(d):
    """S1.1: ω consistency checks."""
    print("="*70)
    print("S1.1: Frequency (ω) consistency")
    print("="*70)

    omega = d['omega']  # (128,128,5)
    Ns = d['Ns1']

    for n in range(5):
        w = omega[:, :, n]
        print(f"  Band {SUBSPACE[n]} (subspace idx {n}): "
              f"min={w.min():.6f}, max={w.max():.6f}, range={w.max()-w.min():.6f}, "
              f"mean={w.mean():.6f}")

    # Check relative to omega_ref
    print(f"\n  omega_ref = {d['omega_ref']:.6f}")
    for n in range(5):
        V = omega[:, :, n] - d['omega_ref']
        print(f"  V_{n} = ω_{n} - ω_ref: min={V.min():.6f}, max={V.max():.6f}")

    # Check Lambda consistency
    for n in range(5):
        lam_diag = d['Lambda'][:, :, n, n]
        V_direct = omega[:, :, n] - d['omega_ref']
        diff = np.max(np.abs(lam_diag - V_direct))
        print(f"  |Lambda_{n}{n} - (omega_{n} - omega_ref)|_max = {diff:.2e}")

    # Check bands never cross (ordering preserved)
    ordering_ok = True
    for i in range(4):
        crossings = np.sum(omega[:, :, i] > omega[:, :, i+1])
        if crossings > 0:
            ordering_ok = False
            frac = crossings / (Ns * Ns)
            print(f"  ⚠ BAND CROSSING: ω_{SUBSPACE[i]} > ω_{SUBSPACE[i+1]} at {crossings} points ({frac:.1%})")
    if ordering_ok:
        print("  ✓ Band ordering preserved (no crossings)")

    print()


# ══════════════════════════════════════════════════════════════════════════
# S1.2: Bloch function orthonormality
# ══════════════════════════════════════════════════════════════════════════

def check_orthonormality(d):
    """S1.2: ⟨u_m|ε|u_n⟩ = δ_mn check at representative points."""
    print("="*70)
    print("S1.2: Bloch function orthonormality")
    print("="*70)

    bf = d['bloch_fields']   # (64,64,18,32,32,3)
    eps = d['epsilon']       # (64,64,32,32)
    Nr = 64  # registry grid
    Nx, Ny = 32, 32
    N_sub = 5

    # Check at 4 R-points: center, corner, two edges
    check_points = [
        ("center",    (32, 32)),
        ("corner",    (0,  0)),
        ("edge_s1",   (16, 0)),
        ("edge_s2",   (0, 16)),
    ]

    all_max_offdiag = []
    all_diag_min = []
    all_diag_max = []

    for label, (ix, iy) in check_points:
        # Extract subspace bands from bloch_fields
        u = bf[ix, iy, SUBSPACE, :, :, :]  # (5, 32, 32, 3)
        e = eps[ix, iy]                     # (32, 32)

        # Compute Gram matrix G_mn = ⟨u_m|ε|u_n⟩
        G_eps = np.zeros((N_sub, N_sub), dtype=complex)
        G_flat = np.zeros((N_sub, N_sub), dtype=complex)
        for m in range(N_sub):
            for n in range(N_sub):
                # ε-weighted: ∑_{x,y,c} ε(x,y) * u_m*(x,y,c) * u_n(x,y,c) / (Nx*Ny)
                um = u[m]  # (32,32,3)
                un = u[n]  # (32,32,3)
                integrand_eps = e[:, :, None] * np.conj(um) * un
                G_eps[m, n] = np.sum(integrand_eps) / (Nx * Ny)
                # flat: ∑ u_m* u_n / (Nx*Ny)
                G_flat[m, n] = np.sum(np.conj(um) * un) / (Nx * Ny)

        # Report
        diag_eps = np.abs(np.diag(G_eps))
        offdiag_eps = np.abs(G_eps - np.diag(np.diag(G_eps)))
        max_offdiag = offdiag_eps.max()
        all_max_offdiag.append(max_offdiag)
        all_diag_min.append(diag_eps.min())
        all_diag_max.append(diag_eps.max())

        print(f"\n  [{label}] R=({ix},{iy})")
        print(f"    ε-weighted Gram diag: {np.diag(G_eps).real}")
        print(f"    ε-weighted max |off-diag|: {max_offdiag:.6e}")
        diag_flat = np.abs(np.diag(G_flat))
        print(f"    flat-norm diag:  {diag_flat}")
        print(f"    flat max |off-diag|: {np.abs(G_flat - np.diag(np.diag(G_flat))).max():.6e}")

    print(f"\n  Summary across checked points:")
    print(f"    ε-Gram diag range: [{min(all_diag_min):.6f}, {max(all_diag_max):.6f}]")
    print(f"    ε-Gram max |off-diag|: {max(all_max_offdiag):.6e}")
    if max(all_max_offdiag) < 1e-6:
        print(f"    ✓ Orthonormality excellent (off-diag < 1e-6)")
    elif max(all_max_offdiag) < 1e-3:
        print(f"    ⚠ Orthonormality OK but imperfect (off-diag ~ {max(all_max_offdiag):.2e})")
    else:
        print(f"    ❌ Orthonormality BROKEN (off-diag = {max(all_max_offdiag):.2e})")

    print()


# ══════════════════════════════════════════════════════════════════════════
# S1.3: Bloch function C4 symmetry at high-symmetry R
# ══════════════════════════════════════════════════════════════════════════

def check_bloch_symmetry(d):
    """S1.3: At C4-symmetric registry, do u_n have spatial C4 structure?"""
    print("="*70)
    print("S1.3: Bloch function spatial symmetry (C4 test)")
    print("="*70)

    bf = d['bloch_fields']   # (64,64,18,32,32,3)
    Nr = 64
    Nx = 32

    # C4 rotation of a field on a square grid: (x,y) → (y, Nx-1-x)
    def rotate_90(field):
        """Rotate field 90° CCW on (Nx,Ny) grid. field: (Nx,Ny,3)."""
        # Spatial rotation: (x,y) → (-y,x) on grid means index transform
        rotated = np.zeros_like(field)
        for ix in range(Nx):
            for iy in range(Nx):
                # New position after 90° CCW: (ix,iy) → (iy, Nx-1-ix)
                jx, jy = iy, Nx - 1 - ix
                # Also rotate vector components: (Ex,Ey,Ez) → (-Ey,Ex,Ez)
                rotated[jx, jy, 0] = -field[ix, iy, 1]
                rotated[jx, jy, 1] =  field[ix, iy, 0]
                rotated[jx, jy, 2] =  field[ix, iy, 2]
        return rotated

    # At δ=(0,0): the two cylinders coincide → single rod → full C4v
    # At δ=(0.5,0.5): also C4v (center of cell)
    symm_points = [
        ("δ=(0,0)",     (0, 0)),
        ("δ=(0.5,0.5)", (32, 32)),
    ]

    for label, (ix, iy) in symm_points:
        print(f"\n  [{label}] registry=({ix},{iy})")
        for n_sub, n_band in enumerate(SUBSPACE):
            u = bf[ix, iy, n_band]  # (32,32,3)
            u_rot = rotate_90(u)

            # C4 test: u_rot should be proportional to u (up to phase)
            # Compute overlap |⟨u|R₄u⟩| / (||u|| ||R₄u||)
            ov = np.sum(np.conj(u) * u_rot) / (Nx * Nx)
            norm_u = np.sqrt(np.sum(np.abs(u)**2) / (Nx * Nx))
            norm_r = np.sqrt(np.sum(np.abs(u_rot)**2) / (Nx * Nx))
            fidelity = np.abs(ov) / (norm_u * norm_r + 1e-30)
            phase = np.angle(ov)

            # Also check intensity C4: |u(r)|² = |u(R₄r)|²
            I_orig = np.sum(np.abs(u)**2, axis=2)  # (32,32)
            I_rot  = np.sum(np.abs(u_rot)**2, axis=2)
            I_diff = np.max(np.abs(I_orig - I_rot)) / (I_orig.max() + 1e-30)

            status = "✓" if fidelity > 0.99 else ("⚠" if fidelity > 0.9 else "❌")
            print(f"    Band {n_band}: |⟨u|C4u⟩|/(||u||||C4u||) = {fidelity:.6f} "
                  f"(phase={phase/np.pi:.3f}π), intensity_diff={I_diff:.4e} {status}")

    print()


# ══════════════════════════════════════════════════════════════════════════
# S1.4: Band crossing detection
# ══════════════════════════════════════════════════════════════════════════

def check_band_crossings(d):
    """S1.4: Detailed band crossing analysis."""
    print("="*70)
    print("S1.4: Band crossing / near-degeneracy analysis")
    print("="*70)

    omega = d['omega']  # (128,128,5)
    Ns = d['Ns1']

    # Gaps between consecutive bands at each R
    for i in range(4):
        gap = omega[:, :, i+1] - omega[:, :, i]
        min_gap = gap.min()
        pct_small = np.sum(gap < 0.001) / (Ns * Ns)
        print(f"  Gap(band {SUBSPACE[i]}→{SUBSPACE[i+1]}): "
              f"min={min_gap:.6f}, mean={gap.mean():.6f}, "
              f"max={gap.max():.6f}, %<0.001={pct_small:.1%}")

    # Gap below subspace (band 4 → band 5) — from stencil data if available
    # Gap above subspace (band 9 → band 10)
    print("\n  (Gaps to bands outside subspace not available in Phase 1 scalar data)")
    print("  (Would need raw stencil/registry_omega_all for full band structure)")

    # Plot: min inter-band gap as heatmap
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    s1 = np.linspace(0, 1, Ns, endpoint=False)
    s2 = np.linspace(0, 1, Ns, endpoint=False)
    for i in range(4):
        gap = omega[:, :, i+1] - omega[:, :, i]
        im = axes[i].pcolormesh(s1, s2, gap.T, cmap='RdYlGn', shading='auto')
        axes[i].set_aspect('equal')
        axes[i].set_title(f"Gap {SUBSPACE[i]}→{SUBSPACE[i+1]}", fontsize=10)
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    fig.suptitle("Inter-band gaps across moiré cell", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S1_band_gaps.png", dpi=150)
    print(f"  Saved S1_band_gaps.png")
    plt.close(fig)

    print()


# ══════════════════════════════════════════════════════════════════════════
# S1.5: k-stencil verification (vg, M_inv consistency)
# ══════════════════════════════════════════════════════════════════════════

def check_k_stencil(d):
    """S1.5: Verify group velocity and mass consistency."""
    print("="*70)
    print("S1.5: k-stencil derived quantities")
    print("="*70)

    vg = d['vg']        # (128,128,5,2)
    M_inv = d['M_inv']  # (128,128,5,2,2)

    # At Γ-point, ALL group velocities should vanish (by time-reversal + inversion)
    for n in range(5):
        vg_n = vg[:, :, n, :]
        max_vg = np.max(np.abs(vg_n))
        mean_vg = np.mean(np.abs(vg_n))
        print(f"  Band {SUBSPACE[n]}: max|vg| = {max_vg:.6e}, mean|vg| = {mean_vg:.6e}")

    if np.max(np.abs(vg)) < 1e-3:
        print("  ✓ All group velocities near zero (consistent with Γ-point)")
    else:
        print("  ⚠ Non-zero group velocities — unexpected at Γ!")

    print(f"\n  M_inv statistics (per band):")
    for n in range(5):
        M = M_inv[:, :, n]  # (128,128,2,2)
        tr = M[:, :, 0, 0] + M[:, :, 1, 1]
        det = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]
        aniso = np.abs(M[:, :, 0, 0] - M[:, :, 1, 1]) / (np.abs(tr) + 1e-30)
        Mxy_rel = np.abs(M[:, :, 0, 1]) / (np.abs(tr) / 2 + 1e-30)
        print(f"  Band {SUBSPACE[n]}: Tr range=[{tr.min():.4f},{tr.max():.4f}], "
              f"mean(Tr)={tr.mean():.4f}")
        print(f"    Anisotropy |Mxx-Myy|/|Tr|: mean={aniso.mean():.4f}, max={aniso.max():.4f}")
        print(f"    Off-diag |Mxy|/(|Tr|/2):   mean={Mxy_rel.mean():.4f}, max={Mxy_rel.max():.4f}")

    # Check: at C4-symmetric points, Mxx=Myy and Mxy=0
    print(f"\n  C4 check at s=(0.5,0.5) — center:")
    ix, iy = 64, 64  # center of 128×128 grid
    for n in range(5):
        M = M_inv[ix, iy, n]
        print(f"    Band {SUBSPACE[n]}: Mxx={M[0,0]:.6f}, Myy={M[1,1]:.6f}, "
              f"Mxy={M[0,1]:.6f}  "
              f"{'✓ C4' if abs(M[0,0]-M[1,1])<0.01*abs(M[0,0]+M[1,1]) and abs(M[0,1])<0.01*abs(M[0,0]) else '⚠ NOT C4'}")

    print()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("S1: Single-R-point MPB Audit")
    print("="*70 + "\n")

    d = load_data()
    print(f"Data loaded: Ns={d['Ns1']}×{d['Ns2']}, L_m={d['L_m']:.2f}a, "
          f"ω_ref={d['omega_ref']:.6f}")
    print(f"Bloch fields: {d['bloch_fields'].shape}, ε: {d['epsilon'].shape}")
    print()

    check_omega(d)
    check_orthonormality(d)
    check_bloch_symmetry(d)
    check_band_crossings(d)
    check_k_stencil(d)

    print("="*70)
    print("S1 complete. Check plots/ for figures.")
    print("="*70)


if __name__ == '__main__':
    main()
