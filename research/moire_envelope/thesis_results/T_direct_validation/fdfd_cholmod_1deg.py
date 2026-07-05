"""
Full comparison: Envelope approximation vs FDFD (CHOLMOD) at θ ≈ 1.1°
for the honeycomb TM bilayer at the Dirac K' point.

Commensurate angle: (m,n) = (30,29), θ = 1.1213°, N_cells = 2611

Steps:
  1. Run envelope approximation at θ=1.1213° (reuses Phase 1/2 data)
  2. Run FDFD with CHOLMOD-accelerated shift-invert on the (30,29) supercell
  3. Compare spectra, bandwidth, gaps, and density
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import json
import time
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle

# ════════════════════════════════════════════════════════════════
# PARAMETERS
# ════════════════════════════════════════════════════════════════
M, N = 30, 29
EPS_BG, EPS_ROD, R_OVER_A = 1.0, 11.56, 0.2
RES = 16  # pts per unit cell side — balance of speed and accuracy
N_FDFD_MODES = 100  # request many modes to cover the narrow envelope window

# Envelope parameters (from pipeline)
OMEGA_REF = 0.23394866216575594  # theta-independent, from Phase 1

N_cells = M*M + M*N + N*N
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N))
Nx = int(round(np.sqrt(N_cells) * RES))

print(f"{'='*70}")
print(f"ENVELOPE vs FDFD COMPARISON AT θ ≈ 1.1°")
print(f"Honeycomb TM bilayer, Dirac K' point")
print(f"{'='*70}")
print(f"(m,n) = ({M},{N}), θ = {theta_deg:.4f}°, N_cells = {N_cells}")
print(f"FDFD: res={RES}, Nx={Nx}, DOF={Nx*Nx:,}")
print()


# ════════════════════════════════════════════════════════════════
# STEP 1: ENVELOPE APPROXIMATION
# ════════════════════════════════════════════════════════════════
print("═" * 70)
print("STEP 1: ENVELOPE APPROXIMATION")
print("═" * 70)

# Run the envelope pipeline at this specific angle
env_results = None
env_sweep_dir = None

try:
    # Import envelope pipeline
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'phasesV3'))
    sys.path.insert(0, PROJECT_ROOT)

    from thesis_utils import find_thesis_run_dir
    import eta_sweep
    import phase4_field_reconstruction as p4_direct
    from pathlib import Path

    # Explicitly use TM run directory (not TE which has _TE_ suffix)
    run_dir = Path('/home/renlephy/msl/research/moire_envelope/runsV3/thesis_honeycomb_K_b1_20260307_171424')
    # Verify this is the TM directory, not TE
    meta_file = run_dir / "candidate_0000" / "phase0_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"Run dir: {run_dir}")
        print(f"Polarization: {meta.get('polarization', 'unknown')}")
    else:
        # Check for phase1 data
        print(f"Run dir: {run_dir}")

    # Check if we already have a sweep at this angle
    existing_sweeps = sorted(run_dir.glob("eta_sweep_*"))
    env_data = None
    for sweep_dir in existing_sweeps:
        sf = sweep_dir / "sweep_results.json"
        if sf.exists():
            with open(sf) as f:
                data = json.load(f)
            for item in data:
                if abs(item['theta_deg'] - theta_deg) < 0.01:
                    env_data = item
                    env_sweep_dir = sweep_dir
                    print(f"Found existing envelope data at θ={item['theta_deg']:.4f}° in {sweep_dir.name}")
                    break
            if env_data:
                break

    if env_data is None:
        print(f"No existing data at θ={theta_deg:.4f}°. Running envelope pipeline...")

        # Monkey-patch to use correct run directory
        original_find = p4_direct.find_latest_run_dir
        p4_direct.find_latest_run_dir = lambda base_name=None: run_dir

        config_overrides = {
            'include_born_huang': False,
            'include_drift_term': True,
            'include_offdiag_A': True,
            'use_parallel_transport_gauge': True,
            'n_extra_bands': 4,
            'mpb_fd_order': 4,
        }

        try:
            results, sweep_dir = eta_sweep.run_eta_sweep(
                candidate_id=0,
                theta_list=[theta_deg],
                n_modes=50,
                config_overrides=config_overrides,
            )
            env_data = results[0]
            env_sweep_dir = sweep_dir
        finally:
            p4_direct.find_latest_run_dir = original_find

    # Extract envelope data
    env_evals = np.array(env_data['eigenvalues'])
    env_omega_ref = env_data['omega_ref']
    env_freqs = env_omega_ref + env_evals
    env_theta = env_data['theta_deg']
    env_eta = env_data['eta']
    env_bw = env_data.get('bandwidth_50', env_freqs.max() - env_freqs.min())

    print(f"\nEnvelope results:")
    print(f"  θ = {env_theta:.4f}°, η = {env_eta:.6f}")
    print(f"  ω_ref = {env_omega_ref:.6f}")
    print(f"  {len(env_evals)} modes")
    print(f"  Eigenvalue range: [{env_evals.min():.6f}, {env_evals.max():.6f}]")
    print(f"  Physical freq range: [{env_freqs.min():.6f}, {env_freqs.max():.6f}]")
    print(f"  Bandwidth: {env_freqs.max() - env_freqs.min():.6f}")

except Exception as e:
    print(f"Envelope pipeline error: {e}")
    import traceback
    traceback.print_exc()
    print("\nFalling back to convergence data at θ=1.1°...")
    # Use convergence test data as fallback
    conv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'T_convergence', 'convergence_results_theta1.1.json')
    with open(conv_file) as f:
        conv = json.load(f)
    hc_data = conv['results']['honeycomb']
    # Use k=50 eigenvalues
    for nm in hc_data['nmodes']:
        if nm['k'] == 50:
            env_evals = np.array(nm['eigenvalues'])
            break
    env_omega_ref = OMEGA_REF
    env_freqs = env_omega_ref + env_evals
    env_theta = conv['theta_deg']
    env_eta = hc_data['eta']
    env_bw = env_freqs.max() - env_freqs.min()

    print(f"  θ = {env_theta}° (convergence data), η = {env_eta:.6f}")
    print(f"  {len(env_evals)} modes")
    print(f"  Physical freq range: [{env_freqs.min():.6f}, {env_freqs.max():.6f}]")
    print(f"  Bandwidth: {env_bw:.6f}")

print()

# ════════════════════════════════════════════════════════════════
# STEP 2: FDFD WITH CHOLMOD
# ════════════════════════════════════════════════════════════════
print("═" * 70)
print("STEP 2: FDFD SUPERCELL SOLVE (CHOLMOD)")
print("═" * 70)

# Build supercell epsilon
t0 = time.time()
eps, info = build_supercell_eps(
    'honeycomb', m=M, n=N, a=1.0,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=Nx, Ny=Nx,
)
t_eps = time.time() - t0
print(f"Supercell eps built: {t_eps:.1f}s, fill={np.mean(eps > 1.5):.3f}")

# Target sigma near the envelope center
sigma = (2 * np.pi * OMEGA_REF) ** 2
print(f"Target sigma = {sigma:.4f} (ω = {OMEGA_REF:.6f})")
print(f"Requesting {N_FDFD_MODES} modes")

# Build FDFD operator at Gamma (q=0)
t0 = time.time()
L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
t_build = time.time() - t0
print(f"Operator built: {t_build:.1f}s, DOF={L.shape[0]:,}, nnz={L.nnz:,}")

# Solve with CHOLMOD
from sksparse.cholmod import cholesky

N_dof = L.shape[0]
L_shifted = L - sigma * sp.eye(N_dof, format='csc')
L_shifted_csc = L_shifted.tocsc()

print("Factorizing with CHOLMOD (LDL^T)...")
t0 = time.time()
factor = cholesky(L_shifted_csc, beta=0, mode='simplicial')
t_factor = time.time() - t0
print(f"  Factorization: {t_factor:.1f}s")

def solve_shifted(b):
    return factor(b)

OPinv = LinearOperator((N_dof, N_dof), matvec=solve_shifted, dtype=L.dtype)

print("Running eigsh (Lanczos)...")
t0 = time.time()
evals, evecs = eigsh(L, k=N_FDFD_MODES, sigma=sigma, which='LM',
                     OPinv=OPinv, maxiter=5000, tol=1e-8)
t_solve = time.time() - t0
print(f"  Eigensolver: {t_solve:.1f}s for {N_FDFD_MODES} modes")
print(f"  Total FDFD time: {t_factor + t_solve:.1f}s")

idx = np.argsort(evals)
evals = evals[idx]
fdfd_freqs_all = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

print(f"\nFDFD frequency range: [{fdfd_freqs_all.min():.6f}, {fdfd_freqs_all.max():.6f}]")

# Save raw FDFD results
out_dir = os.path.dirname(os.path.abspath(__file__))
fdfd_outfile = os.path.join(out_dir, f"fdfd_dirac_m{M}_n{N}_res{RES}.npz")
np.savez(fdfd_outfile,
    freqs=fdfd_freqs_all, evals=evals,
    m=M, n=N, N_cells=N_cells,
    res=RES, Nx=Nx, n_modes=N_FDFD_MODES,
    omega_ref=OMEGA_REF, theta_deg=theta_deg,
    t_factor=t_factor, t_solve=t_solve,
)
print(f"Saved: {fdfd_outfile}")

# ════════════════════════════════════════════════════════════════
# STEP 3: COMPARISON
# ════════════════════════════════════════════════════════════════
print()
print("═" * 70)
print("STEP 3: COMPARISON")
print("═" * 70)

# Focus on envelope frequency window
env_min, env_max = env_freqs.min(), env_freqs.max()
window_margin = 0.002
window_lo = env_min - window_margin
window_hi = env_max + window_margin

fdfd_mask = (fdfd_freqs_all > window_lo) & (fdfd_freqs_all < window_hi)
fdfd_freqs = fdfd_freqs_all[fdfd_mask]

print(f"\nEnvelope: {len(env_freqs)} modes in [{env_min:.6f}, {env_max:.6f}]")
print(f"FDFD: {len(fdfd_freqs)} modes in window [{window_lo:.6f}, {window_hi:.6f}]")
print(f"  (from {len(fdfd_freqs_all)} total FDFD modes)")

if len(fdfd_freqs) == 0:
    print("\nWARNING: No FDFD modes found in envelope window!")
    print(f"FDFD range: [{fdfd_freqs_all.min():.6f}, {fdfd_freqs_all.max():.6f}]")
    print(f"Envelope range: [{env_min:.6f}, {env_max:.6f}]")
    print("Check that sigma is targeting the right frequency range.")
    sys.exit(1)

# ── Bandwidth ──
bw_env = env_max - env_min
bw_fdfd = fdfd_freqs.max() - fdfd_freqs.min()
print(f"\nBandwidth:")
print(f"  Envelope: {bw_env:.6f}")
print(f"  FDFD (in window): {bw_fdfd:.6f}")
print(f"  Ratio: {bw_env / bw_fdfd:.4f}" if bw_fdfd > 0 else "  FDFD bandwidth is zero")

# ── Spectral moments ──
env_mean = np.mean(env_freqs)
fdfd_mean = np.mean(fdfd_freqs)
env_std = np.std(env_freqs)
fdfd_std = np.std(fdfd_freqs)
print(f"\nSpectral moments:")
print(f"  Mean:  Env={env_mean:.6f}, FDFD={fdfd_mean:.6f}, Δ={abs(env_mean-fdfd_mean):.6f}")
print(f"  Std:   Env={env_std:.6f}, FDFD={fdfd_std:.6f}, ratio={env_std/fdfd_std:.4f}" if fdfd_std > 0 else "")

# ── Gap structure ──
def find_gaps(freqs, min_gap_ratio=2.0):
    sorted_f = np.sort(freqs)
    spacings = np.diff(sorted_f)
    if len(spacings) == 0:
        return []
    median_sp = np.median(spacings)
    if median_sp == 0:
        return []
    gaps = []
    for i, sp_val in enumerate(spacings):
        if sp_val > min_gap_ratio * median_sp:
            gaps.append({
                'below': sorted_f[i], 'above': sorted_f[i+1],
                'gap': sp_val, 'center': 0.5*(sorted_f[i]+sorted_f[i+1]),
                'index': i,
            })
    return gaps

env_gaps = find_gaps(env_freqs, 2.0)
fdfd_gaps = find_gaps(fdfd_freqs, 2.0)

print(f"\nGap structure:")
print(f"  Envelope: {len(env_gaps)} significant gaps")
for g in env_gaps[:5]:
    print(f"    Δω={g['gap']:.6f} at ω={g['center']:.6f} (modes {g['index']}-{g['index']+1})")
if len(env_gaps) > 5:
    print(f"    ... ({len(env_gaps)-5} more)")
print(f"  FDFD: {len(fdfd_gaps)} significant gaps")
for g in fdfd_gaps[:5]:
    print(f"    Δω={g['gap']:.6f} at ω={g['center']:.6f} (modes {g['index']}-{g['index']+1})")
if len(fdfd_gaps) > 5:
    print(f"    ... ({len(fdfd_gaps)-5} more)")

# ── Spectral density correlation ──
bins = np.linspace(window_lo, window_hi, 40)
env_hist, _ = np.histogram(env_freqs, bins=bins)
fdfd_hist, _ = np.histogram(fdfd_freqs, bins=bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
env_hist_norm = env_hist / max(len(env_freqs), 1)
fdfd_hist_norm = fdfd_hist / max(len(fdfd_freqs), 1)
if np.any(env_hist_norm > 0) and np.any(fdfd_hist_norm > 0):
    corr = np.corrcoef(env_hist_norm, fdfd_hist_norm)[0, 1]
else:
    corr = 0.0
print(f"\nSpectral density correlation: {corr:.4f}")

# ── CDF / KS statistic ──
env_sorted = np.sort(env_freqs)
fdfd_sorted = np.sort(fdfd_freqs)
env_cdf = np.arange(1, len(env_sorted)+1) / len(env_sorted)
fdfd_cdf = np.arange(1, len(fdfd_sorted)+1) / len(fdfd_sorted)
f_grid = np.linspace(window_lo, window_hi, 500)
env_cdf_interp = np.interp(f_grid, env_sorted, env_cdf, left=0, right=1)
fdfd_cdf_interp = np.interp(f_grid, fdfd_sorted, fdfd_cdf, left=0, right=1)
ks_stat = np.max(np.abs(env_cdf_interp - fdfd_cdf_interp))
print(f"CDF KS statistic: {ks_stat:.4f}")

# ── Nearest-neighbor matching ──
matched_fdfd = []
for ef in env_freqs:
    idx_match = np.argmin(np.abs(fdfd_freqs - ef))
    matched_fdfd.append(fdfd_freqs[idx_match])
matched_fdfd = np.array(matched_fdfd)
residuals = env_freqs - matched_fdfd

print(f"\nNearest-neighbor matching:")
print(f"  Mean |residual|: {np.mean(np.abs(residuals)):.6f}")
print(f"  Max  |residual|: {np.max(np.abs(residuals)):.6f}")
print(f"  RMS  residual:   {np.sqrt(np.mean(residuals**2)):.6f}")
print(f"  Relative (to BW): mean={np.mean(np.abs(residuals))/bw_env:.4f}, max={np.max(np.abs(residuals))/bw_env:.4f}")

# ════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Figure 1: Overview ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

ax = axes[0]
ax.plot(env_freqs, np.arange(len(env_freqs)), 'ro-', ms=3, lw=0.8, label=f'Envelope ({len(env_freqs)})', alpha=0.8)
ax.plot(fdfd_freqs, np.arange(len(fdfd_freqs)), 'bs-', ms=2, lw=0.8, label=f'FDFD ({len(fdfd_freqs)})', alpha=0.8)
ax.axvline(OMEGA_REF, color='gray', ls='--', lw=0.8, label=f'ω_ref={OMEGA_REF:.4f}')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Mode index')
ax.set_title('(a) Ordered eigenvalue spectrum')
ax.legend(fontsize=8)
ax.set_xlim(window_lo, window_hi)

ax = axes[1]
ax.step(bin_centers, env_hist, where='mid', color='red', lw=1.5, label=f'Envelope')
ax.step(bin_centers, fdfd_hist, where='mid', color='blue', lw=1.5, label=f'FDFD')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Count per bin')
ax.set_title(f'(b) Spectral density (corr={corr:.3f})')
ax.legend(fontsize=8)

ax = axes[2]
ax.step(env_sorted, env_cdf, where='post', color='red', lw=1.5, label='Envelope')
ax.step(fdfd_sorted, fdfd_cdf, where='post', color='blue', lw=1.5, label='FDFD')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('CDF')
ax.set_title(f'(c) CDF (KS={ks_stat:.3f})')
ax.legend(fontsize=8)

fig.suptitle(f'Envelope vs FDFD: Honeycomb TM, θ={theta_deg:.2f}°, (m,n)=({M},{N}), Γ_m',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, f'comparison_1deg_overview.png'), dpi=150)
print(f"\nSaved comparison_1deg_overview.png")

# ── Figure 2: Level diagram ──
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

for i, f in enumerate(env_freqs):
    ax2.plot([0.0, 0.4], [f, f], 'r-', lw=0.8, alpha=0.7)
for i, f in enumerate(fdfd_freqs):
    ax2.plot([0.6, 1.0], [f, f], 'b-', lw=0.5, alpha=0.5)

for g in env_gaps:
    ax2.axhspan(g['below'], g['above'], xmin=0, xmax=0.47, color='red', alpha=0.1)
for g in fdfd_gaps:
    ax2.axhspan(g['below'], g['above'], xmin=0.53, xmax=1, color='blue', alpha=0.1)

ax2.axhline(OMEGA_REF, color='green', ls='--', lw=1, alpha=0.6, label=f'ω_ref={OMEGA_REF:.4f}')
ax2.set_xlim(-0.1, 1.1)
ax2.set_xticks([0.2, 0.8])
ax2.set_xticklabels(['Envelope', 'FDFD'], fontsize=12)
ax2.set_ylabel('ω (c/a)')
ax2.set_title(f'Level diagram: θ={theta_deg:.2f}°, (m,n)=({M},{N})')
ax2.legend(loc='upper right')
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, f'comparison_1deg_levels.png'), dpi=150)
print(f"Saved comparison_1deg_levels.png")

# ── Figure 3: Nearest-neighbor matching ──
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

ax = axes3[0]
ax.scatter(env_freqs, matched_fdfd, s=15, c='purple', alpha=0.7)
lims = [window_lo, window_hi]
ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('Envelope ω')
ax.set_ylabel('Nearest FDFD ω')
ax.set_title('(a) Nearest-neighbor matching')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax = axes3[1]
ax.plot(env_freqs, residuals * 1000, 'o', ms=4, alpha=0.7)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Envelope ω')
ax.set_ylabel('Residual × 10³')
ax.set_title(f'(b) Residuals: mean={np.mean(np.abs(residuals))*1000:.2f}×10⁻³, '
             f'max={np.max(np.abs(residuals))*1000:.2f}×10⁻³')

fig3.suptitle(f'Matching: θ={theta_deg:.2f}°', fontsize=12)
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, f'comparison_1deg_matching.png'), dpi=150)
print(f"Saved comparison_1deg_matching.png")

plt.close('all')

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY: θ = {theta_deg:.4f}°, (m,n)=({M},{N}), N_cells={N_cells}")
print(f"{'='*70}")
print(f"Envelope: {len(env_freqs)} modes, BW={bw_env:.6f}")
print(f"FDFD:     {len(fdfd_freqs)} modes in window, BW={bw_fdfd:.6f}")
print(f"BW ratio: {bw_env/bw_fdfd:.4f}" if bw_fdfd > 0 else "")
print(f"Mean Δω:  {abs(env_mean - fdfd_mean):.6f}")
print(f"Spectral density corr: {corr:.4f}")
print(f"KS statistic: {ks_stat:.4f}")
print(f"NN matching: mean|Δ|={np.mean(np.abs(residuals)):.6f}, "
      f"max|Δ|={np.max(np.abs(residuals)):.6f}")
print(f"FDFD time: factor={t_factor:.1f}s, solve={t_solve:.1f}s, total={t_factor+t_solve:.1f}s")
