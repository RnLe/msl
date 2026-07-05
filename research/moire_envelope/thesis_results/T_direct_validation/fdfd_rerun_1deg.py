"""
FDFD re-run with corrected sigma targeting the envelope window center.
Loads envelope data from the completed sweep, then runs FDFD and compares.
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
M, N = 30, 29
EPS_BG, EPS_ROD, R_OVER_A = 1.0, 11.56, 0.2
RES = 16
N_FDFD_MODES = 300  # Need many modes — high mode density near Dirac

N_cells = M*M + M*N + N*N
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N))
Nx = int(round(np.sqrt(N_cells) * RES))

print(f"{'='*70}")
print(f"FDFD RERUN: σ centered on envelope window")
print(f"(m,n)=({M},{N}), θ={theta_deg:.4f}°, N_cells={N_cells}, res={RES}, Nx={Nx}")
print(f"{'='*70}")

# ── Load envelope data ──
sweep_file = "/home/renlephy/msl/research/moire_envelope/runsV3/thesis_honeycomb_K_b1_20260307_171424/eta_sweep_20260310_181650/sweep_results.json"
with open(sweep_file) as f:
    sweep_data = json.load(f)
env_data = sweep_data[0]

env_evals = np.array(env_data['eigenvalues'])
env_omega_ref = env_data['omega_ref']
env_freqs = env_omega_ref + env_evals
env_min, env_max = env_freqs.min(), env_freqs.max()
env_center = 0.5 * (env_min + env_max)
env_bw = env_max - env_min

print(f"\nEnvelope: {len(env_freqs)} modes in [{env_min:.6f}, {env_max:.6f}]")
print(f"  Center: {env_center:.6f}, BW: {env_bw:.6f}")
print(f"  ω_ref = {env_omega_ref:.6f}")

# ── FDFD with sigma at envelope center ──
omega_target = env_center
sigma = (2 * np.pi * omega_target) ** 2
print(f"\nFDFD sigma target: ω = {omega_target:.6f} → σ = {sigma:.4f}")
print(f"Requesting {N_FDFD_MODES} modes")

t0 = time.time()
eps, info = build_supercell_eps(
    'honeycomb', m=M, n=N, a=1.0,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=Nx, Ny=Nx,
)
print(f"Eps built: {time.time()-t0:.1f}s")

t0 = time.time()
L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
print(f"Operator built: {time.time()-t0:.1f}s, DOF={L.shape[0]:,}")

from sksparse.cholmod import cholesky

N_dof = L.shape[0]
L_shifted = L - sigma * sp.eye(N_dof, format='csc')
L_shifted_csc = L_shifted.tocsc()

print("CHOLMOD factorization...")
t0 = time.time()
factor = cholesky(L_shifted_csc, beta=0, mode='simplicial')
t_factor = time.time() - t0
print(f"  Factorization: {t_factor:.1f}s")

OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L.dtype)

print("eigsh...")
t0 = time.time()
evals, evecs = eigsh(L, k=N_FDFD_MODES, sigma=sigma, which='LM',
                     OPinv=OPinv, maxiter=5000, tol=1e-8)
t_solve = time.time() - t0
print(f"  Eigensolver: {t_solve:.1f}s")
print(f"  Total: {t_factor + t_solve:.1f}s")

idx = np.argsort(evals)
evals = evals[idx]
fdfd_freqs_all = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

print(f"\nFDFD full range: [{fdfd_freqs_all.min():.6f}, {fdfd_freqs_all.max():.6f}]")

# Save
out_dir = os.path.dirname(os.path.abspath(__file__))
np.savez(os.path.join(out_dir, f"fdfd_dirac_m{M}_n{N}_res{RES}_v2.npz"),
    freqs=fdfd_freqs_all, evals=evals,
    m=M, n=N, N_cells=N_cells,
    res=RES, Nx=Nx, n_modes=N_FDFD_MODES,
    omega_target=omega_target, theta_deg=theta_deg,
    t_factor=t_factor, t_solve=t_solve,
)

# ── Window and compare ──
window_margin = 0.5 * env_bw  # generous margin
window_lo = env_min - window_margin
window_hi = env_max + window_margin

fdfd_mask = (fdfd_freqs_all >= window_lo) & (fdfd_freqs_all <= window_hi)
fdfd_freqs = fdfd_freqs_all[fdfd_mask]

print(f"\nEnvelope window: [{env_min:.6f}, {env_max:.6f}]")
print(f"FDFD in extended window [{window_lo:.6f}, {window_hi:.6f}]: {len(fdfd_freqs)} modes")
print(f"  (from {len(fdfd_freqs_all)} total)")

if len(fdfd_freqs) == 0:
    print("\nFATAL: No FDFD modes in envelope window!")
    print(f"FDFD spans [{fdfd_freqs_all.min():.6f}, {fdfd_freqs_all.max():.6f}]")
    print(f"Envelope spans [{env_min:.6f}, {env_max:.6f}]")
    # Print histogram of FDFD modes
    bins = np.linspace(fdfd_freqs_all.min(), fdfd_freqs_all.max(), 20)
    hist, edges = np.histogram(fdfd_freqs_all, bins)
    for i, (h, lo, hi) in enumerate(zip(hist, edges[:-1], edges[1:])):
        print(f"  [{lo:.5f}, {hi:.5f}]: {h:4d} modes")
    sys.exit(1)

# Restrict FDFD to exactly the envelope band (not the margin)
fdfd_in_env = fdfd_freqs_all[(fdfd_freqs_all >= env_min) & (fdfd_freqs_all <= env_max)]

bw_fdfd = fdfd_freqs.max() - fdfd_freqs.min()
bw_fdfd_exact = fdfd_in_env.max() - fdfd_in_env.min() if len(fdfd_in_env) > 1 else 0

print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"Envelope: {len(env_freqs)} modes, BW = {env_bw:.6f}")
print(f"FDFD in envelope window: {len(fdfd_in_env)} modes")
print(f"FDFD in extended window: {len(fdfd_freqs)} modes, BW = {bw_fdfd:.6f}")

# ── Nearest-neighbor matching ──
matched_fdfd = []
for ef in env_freqs:
    idx_match = np.argmin(np.abs(fdfd_freqs - ef))
    matched_fdfd.append(fdfd_freqs[idx_match])
matched_fdfd = np.array(matched_fdfd)
residuals = env_freqs - matched_fdfd

print(f"\nNearest-neighbor matching:")
print(f"  Mean |Δ|: {np.mean(np.abs(residuals)):.6f}")
print(f"  Max  |Δ|: {np.max(np.abs(residuals)):.6f}")
print(f"  RMS  Δ:   {np.sqrt(np.mean(residuals**2)):.6f}")
print(f"  Relative (÷ env BW): mean={np.mean(np.abs(residuals))/env_bw:.4f}")

# ── Spectral moments ──
env_mean = np.mean(env_freqs)
fdfd_mean = np.mean(fdfd_freqs)
print(f"\nSpectral moments:")
print(f"  Env  mean={env_mean:.6f}, std={np.std(env_freqs):.6f}")
print(f"  FDFD mean={fdfd_mean:.6f}, std={np.std(fdfd_freqs):.6f}")

# ── Spectral density ──
bins = np.linspace(window_lo, window_hi, 50)
env_hist, _ = np.histogram(env_freqs, bins=bins)
fdfd_hist, _ = np.histogram(fdfd_freqs, bins=bins)
env_n = env_hist / max(len(env_freqs), 1)
fdfd_n = fdfd_hist / max(len(fdfd_freqs), 1)
corr = np.corrcoef(env_n, fdfd_n)[0, 1] if np.any(env_n > 0) and np.any(fdfd_n > 0) else 0
print(f"  Density correlation: {corr:.4f}")

# ── CDF / KS ──
env_sorted = np.sort(env_freqs)
fdfd_sorted = np.sort(fdfd_freqs)
f_grid = np.linspace(window_lo, window_hi, 500)
env_cdf_i = np.interp(f_grid, env_sorted, np.arange(1,len(env_sorted)+1)/len(env_sorted), left=0, right=1)
fdfd_cdf_i = np.interp(f_grid, fdfd_sorted, np.arange(1,len(fdfd_sorted)+1)/len(fdfd_sorted), left=0, right=1)
ks = np.max(np.abs(env_cdf_i - fdfd_cdf_i))
print(f"  KS statistic: {ks:.4f}")

# ── Gap structure ──
def find_gaps(freqs, min_ratio=2.0):
    s = np.sort(freqs)
    sp_vals = np.diff(s)
    med = np.median(sp_vals)
    if med == 0: return []
    return [{'below': s[i], 'above': s[i+1], 'gap': sp_vals[i], 'center': 0.5*(s[i]+s[i+1]), 'idx': i}
            for i, sp_val in enumerate(sp_vals) if sp_val > min_ratio * med]

env_gaps = find_gaps(env_freqs)
fdfd_gaps = find_gaps(fdfd_freqs)
print(f"\nGaps: envelope={len(env_gaps)}, FDFD={len(fdfd_gaps)}")
for g in env_gaps[:5]:
    print(f"  Env gap: Δ={g['gap']:.6f} at ω={g['center']:.6f}")
for g in fdfd_gaps[:5]:
    print(f"  FDFD gap: Δ={g['gap']:.6f} at ω={g['center']:.6f}")

# ════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) Ordered eigenvalues
ax = axes[0, 0]
ax.plot(np.arange(len(env_freqs)), env_freqs, 'ro-', ms=3, lw=0.8, label=f'Envelope ({len(env_freqs)})', alpha=0.8)
ax.plot(np.arange(len(fdfd_freqs)), fdfd_freqs, 'bs-', ms=2, lw=0.6, label=f'FDFD ({len(fdfd_freqs)})', alpha=0.7)
ax.set_xlabel('Mode index')
ax.set_ylabel('ω (c/a)')
ax.set_title('(a) Ordered eigenvalues')
ax.legend(fontsize=8)

# (b) Level diagram
ax = axes[0, 1]
for f in env_freqs:
    ax.plot([0.0, 0.4], [f, f], 'r-', lw=0.8, alpha=0.6)
for f in fdfd_freqs:
    ax.plot([0.6, 1.0], [f, f], 'b-', lw=0.5, alpha=0.4)
ax.axhline(env_omega_ref, color='green', ls='--', lw=1, alpha=0.6, label=f'ω_ref={env_omega_ref:.4f}')
ax.set_xlim(-0.1, 1.1)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['Envelope', 'FDFD'], fontsize=10)
ax.set_ylabel('ω (c/a)')
ax.set_title('(b) Level diagram')
ax.legend(fontsize=8)

# (c) Spectral density
ax = axes[1, 0]
bin_centers = 0.5 * (bins[:-1] + bins[1:])
ax.step(bin_centers, env_hist, where='mid', color='red', lw=1.5, label='Envelope')
ax.step(bin_centers, fdfd_hist, where='mid', color='blue', lw=1.5, label='FDFD')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Count')
ax.set_title(f'(c) Density of states (corr={corr:.3f})')
ax.legend(fontsize=8)

# (d) NN matching residuals
ax = axes[1, 1]
ax.plot(env_freqs, residuals * 1000, 'o', ms=4, alpha=0.7, color='purple')
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Envelope ω')
ax.set_ylabel('Residual × 10³')
ax.set_title(f'(d) NN residuals: mean|Δ|={np.mean(np.abs(residuals))*1e3:.2f}×10⁻³')

fig.suptitle(f'Envelope vs FDFD: θ={theta_deg:.2f}°, (m,n)=({M},{N}), N_cells={N_cells}, res={RES}',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'comparison_1deg_v2.png'), dpi=150)
print(f"\nSaved comparison_1deg_v2.png")

# ── SUMMARY ──
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"θ = {theta_deg:.4f}°, η = {env_data['eta']:.6f}")
print(f"Envelope: {len(env_freqs)} modes, BW = {env_bw:.6f}, ω∈[{env_min:.6f}, {env_max:.6f}]")
print(f"FDFD: {len(fdfd_freqs)} modes in window, BW = {bw_fdfd:.6f}")
print(f"FDFD in env range: {len(fdfd_in_env)} modes")
print(f"BW ratio: {env_bw/bw_fdfd:.4f}" if bw_fdfd > 0 else "")
print(f"NN match: mean|Δ|={np.mean(np.abs(residuals)):.6f}, max|Δ|={np.max(np.abs(residuals)):.6f}")
print(f"Density corr: {corr:.4f}")
print(f"KS stat: {ks:.4f}")
print(f"FDFD time: {t_factor+t_solve:.1f}s (factor={t_factor:.1f}, solve={t_solve:.1f})")
