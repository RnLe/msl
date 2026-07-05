#!/usr/bin/env python3
"""
S2: R-dependent field symmetry and subspace validity scan
==========================================================
Maps across the FULL moiré registry grid:
  S2.1  Subspace closure metric (do bands [5-9] form a closed C4 subspace?)
  S2.2  Band gaps to outside bands (need raw MPB data for bands 4 and 10)
  S2.3  C4 fidelity per band and per pair
  S2.4  Potential V(R) = Λ(R) heatmaps with C4 test
  S2.5  M_inv field maps + divergence locations
  S2.6  Berry connection smoothness + C4 covariance
  S2.7  SVQB Gram matrix condition check at every R
"""

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

def rotate_90_origin(field, Nx):
    """C4 rotation (90° CCW) about origin on periodic grid."""
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated


def load_data():
    d = {}
    with h5py.File(P1, 'r') as f:
        d['omega']        = f['omega'][:]
        d['vg']           = f['vg'][:]
        d['M_inv']        = f['M_inv'][:]
        d['bloch_fields'] = f['bloch_fields'][:]
        d['epsilon']      = f['epsilon'][:]
        d['Ns1']          = int(f.attrs['Ns1'])
        d['omega_ref']    = float(f.attrs['omega_ref'])
    with h5py.File(P2, 'r') as f:
        d['Lambda']  = f['Lambda'][:]
        d['A_berry'] = f['A_berry'][:]
        d['M_inv_2'] = f['M_inv'][:]
        d['v_drift'] = f['v_drift'][:]
        d['Phi_BH']  = f['Phi_BH'][:]
    return d


# ══════════════════════════════════════════════════════════════════════════
# S2.1: Subspace closure under C4 at every R
# ══════════════════════════════════════════════════════════════════════════

def scan_subspace_closure(d):
    """Compute C4 subspace closure metric at every R on the 64×64 bloch grid."""
    print("="*70)
    print("S2.1: Subspace C4 closure scan (64×64 registry grid)")
    print("="*70)

    bf = d['bloch_fields']  # (64,64,18,32,32,3)
    Nr = 64
    Nx = 32

    # For each R, compute 5×5 overlap matrix M_mn = ⟨u_m|C4 u_n⟩/norms
    # Subspace closure ≡ |det(M)| ≈ 1
    # Also compute min singular value (0 → not closed)
    closure_det = np.zeros((Nr, Nr))
    closure_minsv = np.zeros((Nr, Nr))
    c4_fidelity_map = np.zeros((Nr, Nr, 5))

    for ix in range(Nr):
        for iy in range(Nr):
            states = [bf[ix, iy, b] for b in SUBSPACE]
            norms = np.array([np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx)) for u in states])
            rotated = [rotate_90_origin(u, Nx) for u in states]

            M = np.zeros((5, 5), dtype=complex)
            for m in range(5):
                for n in range(5):
                    M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx*Nx)

            M_norm = M / np.outer(norms, norms)
            closure_det[ix, iy] = np.abs(np.linalg.det(M_norm))
            sv = np.linalg.svd(M_norm, compute_uv=False)
            closure_minsv[ix, iy] = sv.min()

            # Per-band fidelity (diagonal)
            for n in range(5):
                c4_fidelity_map[ix, iy, n] = np.abs(M_norm[n, n])

    # Report
    print(f"  |det(M5)| across registry:")
    print(f"    min={closure_det.min():.6f}, max={closure_det.max():.6f}, "
          f"mean={closure_det.mean():.6f}")
    print(f"    fraction <0.5: {np.sum(closure_det < 0.5)/(Nr*Nr):.1%}")
    print(f"    fraction <0.9: {np.sum(closure_det < 0.9)/(Nr*Nr):.1%}")
    print(f"    fraction >0.99: {np.sum(closure_det > 0.99)/(Nr*Nr):.1%}")

    print(f"\n  min_sv(M5) across registry:")
    print(f"    min={closure_minsv.min():.6f}, max={closure_minsv.max():.6f}")
    print(f"    fraction <0.5: {np.sum(closure_minsv < 0.5)/(Nr*Nr):.1%}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    s = np.linspace(0, 1, Nr, endpoint=False)

    im0 = axes[0].pcolormesh(s, s, closure_det.T, cmap='RdYlGn', shading='auto',
                             vmin=0, vmax=1)
    axes[0].set_title('|det(M₅)| — subspace closure')
    axes[0].set_xlabel('s₁'); axes[0].set_ylabel('s₂')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(s, s, closure_minsv.T, cmap='RdYlGn', shading='auto',
                             vmin=0, vmax=1)
    axes[1].set_title('min σ(M₅) — worst direction')
    axes[1].set_xlabel('s₁')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(s, s, c4_fidelity_map[:,:,2].T,
                             cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    axes[2].set_title('Band 7 C4 self-fidelity')
    axes[2].set_xlabel('s₁')
    axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle('S2.1: C4 subspace closure across moiré cell', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_subspace_closure.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S2_subspace_closure.png\n")

    return closure_det, closure_minsv


# ══════════════════════════════════════════════════════════════════════════
# S2.2: Band gap analysis (within subspace + to outside)
# ══════════════════════════════════════════════════════════════════════════

def scan_band_gaps(d):
    """Analyze inter-band gaps across the moiré cell."""
    print("="*70)
    print("S2.2: Band gap analysis (128×128 moiré grid)")
    print("="*70)

    omega = d['omega']  # (128,128,5)
    Ns = d['Ns1']

    # Internal gaps
    gaps = np.zeros((Ns, Ns, 4))
    for i in range(4):
        gaps[:, :, i] = omega[:, :, i+1] - omega[:, :, i]

    min_internal_gap = gaps.min(axis=2)  # (128,128) min gap within subspace

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    s = np.linspace(0, 1, Ns, endpoint=False)
    for i in range(4):
        im = axes[i].pcolormesh(s, s, gaps[:,:,i].T, cmap='RdYlGn',
                                shading='auto', vmin=0, vmax=0.05)
        axes[i].set_title(f'Gap {SUBSPACE[i]}→{SUBSPACE[i+1]}')
        axes[i].set_aspect('equal')
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    im = axes[4].pcolormesh(s, s, min_internal_gap.T, cmap='RdYlGn',
                            shading='auto', vmin=0, vmax=0.05)
    axes[4].set_title('Min internal gap')
    axes[4].set_aspect('equal')
    plt.colorbar(im, ax=axes[4], shrink=0.8)

    fig.suptitle('S2.2: Inter-band gaps across moiré cell', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_band_gaps_map.png", dpi=150)
    plt.close(fig)

    # Near-degeneracy locations
    for i in range(4):
        ndeg = np.sum(gaps[:,:,i] < 0.001)
        print(f"  Gap {SUBSPACE[i]}→{SUBSPACE[i+1]}: "
              f"min={gaps[:,:,i].min():.6f}, <0.001 at {ndeg} points ({ndeg/(Ns*Ns):.1%})")

    print(f"  Saved S2_band_gaps_map.png\n")


# ══════════════════════════════════════════════════════════════════════════
# S2.3: ε-orthonormality scan across R (before SVQB)
# ══════════════════════════════════════════════════════════════════════════

def scan_orthonormality(d):
    """Check ⟨u_m|ε|u_n⟩ = δ_mn at every R on the 64×64 grid."""
    print("="*70)
    print("S2.3: ε-orthonormality scan (pre-SVQB, 64×64 bloch grid)")
    print("="*70)

    bf = d['bloch_fields']  # (64,64,18,32,32,3)
    eps = d['epsilon']      # (64,64,32,32)
    Nr = 64
    Nx = 32
    Nb = 5

    max_offdiag  = np.zeros((Nr, Nr))
    diag_spread  = np.zeros((Nr, Nr))  # max|diag - 1|
    condition_G  = np.zeros((Nr, Nr))  # condition number of Gram

    for ix in range(Nr):
        for iy in range(Nr):
            e = eps[ix, iy]  # (32,32)
            G = np.zeros((Nb, Nb), dtype=complex)
            for m in range(Nb):
                for n in range(Nb):
                    um = bf[ix, iy, SUBSPACE[m]]
                    un = bf[ix, iy, SUBSPACE[n]]
                    G[m, n] = np.sum(e[:,:,None] * np.conj(um) * un) / (Nx*Nx)

            max_offdiag[ix, iy] = np.abs(G - np.diag(np.diag(G))).max()
            diag_spread[ix, iy] = np.max(np.abs(np.diag(G).real - 1.0))
            sv = np.linalg.svd(G, compute_uv=False)
            condition_G[ix, iy] = sv.max() / (sv.min() + 1e-30)

    print(f"  Max |off-diag|: min={max_offdiag.min():.6e}, max={max_offdiag.max():.6e}, "
          f"mean={max_offdiag.mean():.6e}")
    print(f"  Max |diag-1|:   min={diag_spread.min():.6e}, max={diag_spread.max():.6e}, "
          f"mean={diag_spread.mean():.6e}")
    print(f"  Condition(G):   min={condition_G.min():.2f}, max={condition_G.max():.2f}, "
          f"mean={condition_G.mean():.2f}")
    print(f"  Fraction with off-diag > 0.01: {np.sum(max_offdiag > 0.01)/(Nr*Nr):.1%}")
    print(f"  Fraction with cond(G) > 2:     {np.sum(condition_G > 2)/(Nr*Nr):.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    im0 = axes[0].pcolormesh(sr, sr, np.log10(max_offdiag.T + 1e-15),
                             cmap='hot_r', shading='auto')
    axes[0].set_title('log₁₀(max |off-diag|)')
    axes[0].set_aspect('equal'); plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(sr, sr, diag_spread.T, cmap='hot_r', shading='auto')
    axes[1].set_title('max |diag - 1|')
    axes[1].set_aspect('equal'); plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(sr, sr, np.log10(condition_G.T), cmap='hot_r', shading='auto')
    axes[2].set_title('log₁₀(cond(G))')
    axes[2].set_aspect('equal'); plt.colorbar(im2, ax=axes[2])

    fig.suptitle('S2.3: ε-Gram quality before SVQB', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_gram_quality.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S2_gram_quality.png\n")


# ══════════════════════════════════════════════════════════════════════════
# S2.4: Potential V(R) = Λ(R) heatmaps + C4 test
# ══════════════════════════════════════════════════════════════════════════

def scan_potential(d):
    """Map the potential landscape and test C4."""
    print("="*70)
    print("S2.4: Potential landscape Λ(R)")
    print("="*70)

    Lambda = d['Lambda']  # (128,128,5,5)
    Ns = d['Ns1']

    # Diagonal V_n(R) = Λ_nn(R)
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    s = np.linspace(0, 1, Ns, endpoint=False)
    for n in range(5):
        V = Lambda[:, :, n, n]
        im = axes[n].pcolormesh(s, s, V.T, cmap='RdBu_r', shading='auto',
                                vmin=-0.15, vmax=0.15)
        axes[n].set_title(f'Λ_{{{SUBSPACE[n]},{SUBSPACE[n]}}}')
        axes[n].set_aspect('equal')
        plt.colorbar(im, ax=axes[n], shrink=0.8)
    fig.suptitle('S2.4: Diagonal potential Λ_nn(R)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_potential_diagonal.png", dpi=150)
    plt.close(fig)

    # Off-diagonal |Λ_mn| to check coupling
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    count = 0
    for m in range(5):
        for n in range(m+1, 5):
            if count >= 10:
                break
            r = count // 5
            c = count % 5
            V_off = np.abs(Lambda[:, :, m, n])
            im = axes[r, c].pcolormesh(s, s, V_off.T, cmap='hot', shading='auto',
                                        vmin=0, vmax=0.02)
            axes[r, c].set_title(f'|Λ_{{{SUBSPACE[m]},{SUBSPACE[n]}}}|', fontsize=9)
            axes[r, c].set_aspect('equal')
            plt.colorbar(im, ax=axes[r, c], shrink=0.6)
            count += 1
    fig.suptitle('S2.4: Off-diagonal coupling |Λ_mn(R)|', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_potential_offdiag.png", dpi=150)
    plt.close(fig)

    # C4 test: Λ(R) should satisfy Λ(C4·R) = Λ(R) for diagonal
    # On 128×128 grid, C4 maps (ix,iy) → ((Ns-iy)%Ns, ix)
    c4_err = np.zeros(5)
    for n in range(5):
        V = Lambda[:, :, n, n]
        V_rot = np.zeros_like(V)
        for ix in range(Ns):
            for iy in range(Ns):
                jx = (Ns - iy) % Ns
                jy = ix
                V_rot[jx, jy] = V[ix, iy]
        c4_err[n] = np.max(np.abs(V - V_rot)) / (np.max(np.abs(V)) + 1e-30)
        print(f"  Band {SUBSPACE[n]}: max|V - C4·V|/max|V| = {c4_err[n]:.6e}")

    if np.max(c4_err) < 1e-6:
        print("  ✓ Potential is C4-symmetric")
    elif np.max(c4_err) < 0.01:
        print("  ⚠ Potential approximately C4 (within 1%)")
    else:
        print(f"  ❌ Potential breaks C4! max error = {np.max(c4_err):.2e}")

    print(f"  Saved S2_potential_diagonal.png, S2_potential_offdiag.png\n")


# ══════════════════════════════════════════════════════════════════════════
# S2.5: M_inv field maps + divergence locations
# ══════════════════════════════════════════════════════════════════════════

def scan_M_inv(d):
    """Map M_inv components and locate divergence regions."""
    print("="*70)
    print("S2.5: Inverse effective mass M⁻¹(R)")
    print("="*70)

    M_inv = d['M_inv']  # (128,128,5,2,2) — Phase 1 single-band
    M_inv2 = d['M_inv_2']  # (128,128,5,5,2,2) — Phase 2 multi-band
    Ns = d['Ns1']

    # Phase 1: Single-band M_inv
    fig, axes = plt.subplots(3, 5, figsize=(24, 12))
    s = np.linspace(0, 1, Ns, endpoint=False)
    for n in range(5):
        for comp_idx, (ci, cj, comp_name) in enumerate([(0,0,'xx'), (1,1,'yy'), (0,1,'xy')]):
            v = M_inv[:, :, n, ci, cj].real
            vmax = min(np.percentile(np.abs(v), 95), 20)
            im = axes[comp_idx, n].pcolormesh(s, s, v.T, cmap='RdBu_r',
                                               shading='auto', vmin=-vmax, vmax=vmax)
            axes[comp_idx, n].set_title(f'M⁻¹_{comp_name} band {SUBSPACE[n]}', fontsize=9)
            axes[comp_idx, n].set_aspect('equal')
            plt.colorbar(im, ax=axes[comp_idx, n], shrink=0.6)

    fig.suptitle('S2.5: Single-band M⁻¹(R) — Phase 1', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_Minv_singleband.png", dpi=150)
    plt.close(fig)

    # Divergence map: points where |Tr(M)| > threshold
    thresholds = [5, 10, 20, 40]
    print("  Divergence fraction (|Tr(M⁻¹)| > threshold):")
    for th in thresholds:
        for n in range(5):
            tr = np.abs(M_inv[:, :, n, 0, 0] + M_inv[:, :, n, 1, 1])
            frac = np.sum(tr > th) / (Ns * Ns)
            print(f"    Band {SUBSPACE[n]}, |Tr|>{th}: {frac:.1%}")
        print()

    # C4 test on M_inv: at C4 symmetric R, M should be isotropic
    ix_c, iy_c = Ns // 2, Ns // 2
    print(f"  C4 check at center (ix={ix_c}, iy={iy_c}):")
    for n in range(5):
        M = M_inv[ix_c, iy_c, n]
        iso_err = abs(M[0,0] - M[1,1]) / (abs(M[0,0] + M[1,1])/2 + 1e-30)
        xy_err  = abs(M[0,1]) / (abs(M[0,0] + M[1,1])/2 + 1e-30)
        status = "✓" if iso_err < 0.02 and xy_err < 0.02 else "❌"
        print(f"    Band {SUBSPACE[n]}: Mxx={M[0,0]:.4f}, Myy={M[1,1]:.4f}, "
              f"Mxy={M[0,1]:.4f} → aniso={iso_err:.4f}, |xy|_rel={xy_err:.4f} {status}")

    print(f"  Saved S2_Minv_singleband.png\n")


# ══════════════════════════════════════════════════════════════════════════
# S2.6: Berry connection smoothness
# ══════════════════════════════════════════════════════════════════════════

def scan_berry_connection(d):
    """Analyze Berry connection smoothness and C4 covariance."""
    print("="*70)
    print("S2.6: Berry connection A(R)")
    print("="*70)

    A = d['A_berry']  # (128,128,5,5,2) complex
    Ns = d['Ns1']

    # Diagonal A_nn = diagonal Berry connection
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    s = np.linspace(0, 1, Ns, endpoint=False)
    for n in range(5):
        # s1 component
        A_s1 = A[:, :, n, n, 0].real
        A_s2 = A[:, :, n, n, 1].real
        vmax1 = np.percentile(np.abs(A_s1), 98)
        vmax2 = np.percentile(np.abs(A_s2), 98)

        im0 = axes[0, n].pcolormesh(s, s, A_s1.T, cmap='RdBu_r', shading='auto',
                                     vmin=-vmax1, vmax=vmax1)
        axes[0, n].set_title(f'A₁(band {SUBSPACE[n]})', fontsize=9)
        axes[0, n].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, n], shrink=0.6)

        im1 = axes[1, n].pcolormesh(s, s, A_s2.T, cmap='RdBu_r', shading='auto',
                                     vmin=-vmax2, vmax=vmax2)
        axes[1, n].set_title(f'A₂(band {SUBSPACE[n]})', fontsize=9)
        axes[1, n].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, n], shrink=0.6)

    fig.suptitle('S2.6: Diagonal Berry connection A_nn(R)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_berry_diagonal.png", dpi=150)
    plt.close(fig)

    # Statistics
    for n in range(5):
        for d_idx, d_name in [(0, 's1'), (1, 's2')]:
            A_nd = A[:, :, n, n, d_idx]
            print(f"  A_{d_name}(band {SUBSPACE[n]}): "
                  f"mean={A_nd.real.mean():.4f}±{A_nd.real.std():.4f}, "
                  f"|imag|_max={np.abs(A_nd.imag).max():.4e}")

    # Smoothness: gradient of A
    print(f"\n  Berry connection gradients (roughness):")
    for n in range(5):
        for d_idx, d_name in [(0, 's1'), (1, 's2')]:
            A_nd = A[:, :, n, n, d_idx].real
            grad_s1 = np.diff(A_nd, axis=0)
            grad_s2 = np.diff(A_nd, axis=1)
            max_grad = max(np.abs(grad_s1).max(), np.abs(grad_s2).max())
            rms_grad = np.sqrt(np.mean(grad_s1**2) + np.mean(grad_s2**2))
            print(f"    ∇A_{d_name}(band {SUBSPACE[n]}): max_jump={max_grad:.4f}, "
                  f"rms={rms_grad:.4f}")

    # Off-diagonal Berry connection
    print(f"\n  Off-diagonal |A_mn| statistics:")
    for m in range(5):
        for n in range(m+1, 5):
            A_mn = A[:, :, m, n, :]
            maxval = np.abs(A_mn).max()
            meanval = np.abs(A_mn).mean()
            print(f"    |A({SUBSPACE[m]},{SUBSPACE[n]})|: mean={meanval:.4e}, max={maxval:.4e}")

    print(f"  Saved S2_berry_diagonal.png\n")


# ══════════════════════════════════════════════════════════════════════════
# S2.7: Born-Huang + v_drift
# ══════════════════════════════════════════════════════════════════════════

def scan_born_huang_and_drift(d):
    """Analyze Born-Huang potential and drift velocity."""
    print("="*70)
    print("S2.7: Born-Huang Φ_BH and drift velocity v_drift")
    print("="*70)

    Phi = d['Phi_BH']   # (128,128,5,5)
    vd  = d['v_drift']  # (128,128,5,5,2)
    Ns = d['Ns1']

    # Diagonal Φ_BH
    for n in range(5):
        phi_nn = Phi[:, :, n, n]
        print(f"  Φ_BH({SUBSPACE[n]},{SUBSPACE[n]}): "
              f"mean={phi_nn.mean():.6e}, std={phi_nn.std():.6e}, "
              f"range=[{phi_nn.min():.6e}, {phi_nn.max():.6e}]")

    # Off-diagonal Φ_BH
    print(f"\n  Off-diagonal |Φ_BH_mn|:")
    for m in range(5):
        for n in range(m+1, 5):
            phi_mn = np.abs(Phi[:, :, m, n])
            print(f"    |Φ_BH({SUBSPACE[m]},{SUBSPACE[n]})|: mean={phi_mn.mean():.6e}, max={phi_mn.max():.6e}")

    # Drift velocity
    print(f"\n  Diagonal drift velocity v_drift_nn:")
    for n in range(5):
        for d_idx, d_name in [(0, 's1'), (1, 's2')]:
            v = vd[:, :, n, n, d_idx]
            print(f"    v_{d_name}({SUBSPACE[n]},{SUBSPACE[n]}): "
                  f"mean={v.mean():.6e}, std={v.std():.6e}")

    # Off-diagonal v_drift
    max_offdiag_v = 0
    for m in range(5):
        for n in range(5):
            if m != n:
                v_mn = np.abs(vd[:, :, m, n, :])
                max_offdiag_v = max(max_offdiag_v, v_mn.max())
    print(f"\n  Max |off-diagonal v_drift|: {max_offdiag_v:.6e}")
    if max_offdiag_v < 1e-10:
        print(f"  ⚠ Off-diagonal v_drift is ZERO — inter-band coupling missing!")

    # Phi_BH heatmaps
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    s = np.linspace(0, 1, Ns, endpoint=False)
    for n in range(5):
        phi_nn = Phi[:, :, n, n]
        im = axes[n].pcolormesh(s, s, phi_nn.T, cmap='RdBu_r', shading='auto')
        axes[n].set_title(f'Φ_BH({SUBSPACE[n]},{SUBSPACE[n]})', fontsize=9)
        axes[n].set_aspect('equal')
        plt.colorbar(im, ax=axes[n], shrink=0.8)
    fig.suptitle('S2.7: Born-Huang potential Φ_BH(R)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2_born_huang.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S2_born_huang.png\n")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("S2: R-dependent field symmetry & subspace validity scan")
    print("="*70 + "\n")

    d = load_data()
    print(f"Data loaded: bloch_fields {d['bloch_fields'].shape}, "
          f"Ns={d['Ns1']}, omega_ref={d['omega_ref']:.6f}\n")

    scan_subspace_closure(d)
    scan_band_gaps(d)
    scan_orthonormality(d)
    scan_potential(d)
    scan_M_inv(d)
    scan_berry_connection(d)
    scan_born_huang_and_drift(d)

    print("="*70)
    print("S2 complete. Check plots/ for figures.")
    print("="*70)


if __name__ == '__main__':
    main()
