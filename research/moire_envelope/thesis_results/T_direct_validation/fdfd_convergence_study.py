"""
Phase 2: FDFD resolution convergence at θ ≈ 1.1° (m,n)=(30,29).
Runs res={12, 16, 20} and compares eigenvalue drift.
res=16 is already computed (loads from file), runs 12 and 20.
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
from scipy.optimize import linear_sum_assignment

out_dir = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
M, N_mn = 30, 29
EPS_BG, EPS_ROD, R_OVER_A = 1.0, 11.56, 0.2
N_FDFD_MODES = 100  # 100 is enough — we only need ~60 in the envelope window
N_cells = M*M + M*N_mn + N_mn*N_mn
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N_mn))

# Load envelope window center (from BH run)
with open('/home/renlephy/msl/research/moire_envelope/runsV3/'
          'thesis_honeycomb_K_b1_20260307_171424/'
          'eta_sweep_20260310_191610/sweep_results.json') as f:
    env_bh = json.load(f)[0]
env_freqs = np.sort(env_bh['omega_ref'] + np.array(env_bh['eigenvalues']))
env_min, env_max = env_freqs.min(), env_freqs.max()
env_center = 0.5 * (env_min + env_max)
env_bw = env_max - env_min
sigma_target = (2 * np.pi * env_center) ** 2

print(f"{'='*70}")
print(f"FDFD RESOLUTION CONVERGENCE")
print(f"(m,n)=({M},{N_mn}), θ={theta_deg:.4f}°, N_cells={N_cells}")
print(f"Envelope window: [{env_min:.6f}, {env_max:.6f}], BW={env_bw:.6f}")
print(f"σ target: ω={env_center:.6f}")
print(f"{'='*70}\n")

# ════════════════════════════════════════════════════════════════
resolutions = [12, 16, 20, 40]
results = {}

for res in resolutions:
    Nx = int(round(np.sqrt(N_cells) * res))
    N_dof = Nx * Nx
    tag = f"res{res}"

    # Check if already computed
    fname = os.path.join(out_dir, f'fdfd_dirac_m{M}_n{N_mn}_res{res}_v2.npz')
    if os.path.exists(fname):
        print(f"\n── res={res}: Loading from {os.path.basename(fname)} ──")
        data = np.load(fname)
        fdfd_freqs = np.sort(data['freqs'])
        results[res] = fdfd_freqs
        print(f"   Loaded {len(fdfd_freqs)} modes, range [{fdfd_freqs.min():.6f}, {fdfd_freqs.max():.6f}]")
        continue

    print(f"\n── res={res}: Nx={Nx}, DOF={N_dof:,} ──")

    t0 = time.time()
    eps, info = build_supercell_eps(
        'honeycomb', m=M, n=N_mn, a=1.0,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=Nx, Ny=Nx,
    )
    print(f"   Epsilon grid built: {time.time()-t0:.1f}s")

    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    print(f"   Operator built: {time.time()-t0:.1f}s")

    sigma = sigma_target
    L_shifted = L - sigma * sp.eye(N_dof, format='csc')

    from sksparse.cholmod import cholesky
    print(f"   CHOLMOD factorization...")
    t0 = time.time()
    factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
    t_factor = time.time() - t0
    print(f"   Factorization: {t_factor:.1f}s")

    OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L.dtype)

    print(f"   eigsh ({N_FDFD_MODES} modes)...")
    t0 = time.time()
    evals, _ = eigsh(L, k=N_FDFD_MODES, sigma=sigma, which='LM',
                     OPinv=OPinv, maxiter=5000, tol=1e-8)
    t_solve = time.time() - t0
    print(f"   Eigensolver: {t_solve:.1f}s")
    print(f"   Total: {t_factor + t_solve:.1f}s")

    idx = np.argsort(evals)
    evals = evals[idx]
    fdfd_freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    fdfd_freqs = np.sort(fdfd_freqs)

    # Save
    np.savez(fname, freqs=fdfd_freqs, evals=evals,
             m=M, n=N_mn, N_cells=N_cells, res=res, Nx=Nx,
             n_modes=N_FDFD_MODES, omega_target=env_center,
             theta_deg=theta_deg, t_factor=t_factor, t_solve=t_solve)
    print(f"   Saved {os.path.basename(fname)}")

    results[res] = fdfd_freqs
    del L, L_shifted, factor, OPinv, evals  # free memory

# ════════════════════════════════════════════════════════════════
# Compare: eigenvalue drift between resolutions
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"CONVERGENCE ANALYSIS")
print(f"{'='*70}")

# For each resolution, extract modes in envelope window and match
def extract_window(freqs, lo, hi, margin=0.001):
    mask = (freqs >= lo - margin) & (freqs <= hi + margin)
    return freqs[mask]

fdfd_by_res = {}
for res in resolutions:
    w = extract_window(results[res], env_min, env_max)
    fdfd_by_res[res] = w
    in_env = np.sum((results[res] >= env_min) & (results[res] <= env_max))
    print(f"res={res}: {len(w)} modes in extended window, {in_env} in exact env range")

# Match res=12→16 and res=16→20 using Hungarian
def hungarian_match(freqs_a, freqs_b):
    cost = np.abs(freqs_a[:, None] - freqs_b[None, :])
    r, c = linear_sum_assignment(cost)
    return freqs_a[r], freqs_b[c]

# Compare to res=16 as reference
ref = 16
f_ref = fdfd_by_res[ref]

for res in resolutions:
    if res == ref:
        continue
    f_test = fdfd_by_res[res]
    n_match = min(len(f_ref), len(f_test))
    matched_ref, matched_test = hungarian_match(f_ref[:n_match], f_test[:n_match])
    drift = np.abs(matched_ref - matched_test)

    print(f"\nres={res} vs res={ref} ({n_match} matched modes):")
    print(f"  mean|drift| = {np.mean(drift)*1e6:.1f}×10⁻⁶")
    print(f"  max|drift|  = {np.max(drift)*1e6:.1f}×10⁻⁶")
    print(f"  mean|drift|/BW = {np.mean(drift)/env_bw*100:.3f}%")

# Also compare EA-FDFD residual at each resolution
print(f"\n--- EA vs FDFD at each resolution ---")
for res in resolutions:
    w = fdfd_by_res[res]
    n_match = min(len(env_freqs), len(w))
    cost = np.abs(env_freqs[:n_match, None] - w[None, :])
    r, c = linear_sum_assignment(cost)
    matched = w[c]
    absres = np.abs(env_freqs[r] - matched)
    N_within_1 = np.sum(absres < np.mean(np.diff(env_freqs)))
    print(f"  res={res}: mean|Δ|={np.mean(absres)*1e6:.1f}×10⁻⁶ ({np.mean(absres)/env_bw*100:.2f}% BW), "
          f"max={np.max(absres)*1e6:.1f}×10⁻⁶, {N_within_1}/50 within 1 spacing")

# ════════════════════════════════════════════════════════════════
# Plot: convergence figure
# ════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
colors = {12: '#EAB308', 16: '#2563EB', 20: '#16A34A', 40: '#7C3AED'}

# (a) Eigenfrequencies at each resolution
ax = axes[0]
for res in resolutions:
    w = fdfd_by_res[res]
    in_range = w[(w >= env_min) & (w <= env_max)]
    ax.plot(np.arange(len(in_range)), in_range * 1e3, 'o-', ms=3, lw=0.6,
            color=colors[res], alpha=0.7, label=f'res={res} ({len(in_range)} modes)')

ax.plot(np.arange(len(env_freqs)), env_freqs * 1e3, 'x', ms=4, color='#DC2626',
        alpha=0.8, label='Envelope')
ax.set_xlabel('Mode index (in env window)', fontsize=10)
ax.set_ylabel('ω  (×10⁻³  c/a)', fontsize=10)
ax.set_title('(a)  Modes at each resolution', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# (b) Drift relative to res=40 (highest)
ax = axes[1]
f_best = fdfd_by_res[40]
markers = {12: 'v', 16: 's', 20: '^'}
for res_test in [12, 16, 20]:
    f_test = fdfd_by_res[res_test]
    n_match = min(len(f_best), len(f_test))
    matched_best, matched_test = hungarian_match(f_best[:n_match], f_test[:n_match])
    drift = (matched_test - matched_best) * 1e6
    ax.scatter(np.arange(len(drift)), drift, s=20, marker=markers[res_test],
               color=colors[res_test], alpha=0.7, label=f'res={res_test}→40')

ax.axhline(0, color='black', lw=0.8)

# Show EA-FDFD(res=40) residual for comparison
cost = np.abs(env_freqs[:, None] - f_best[None, :])
r, c = linear_sum_assignment(cost)
ea_res = (env_freqs[r] - f_best[c]) * 1e6
ax.fill_between([0, max(len(drift), 50)],
                -np.mean(np.abs(ea_res)), np.mean(np.abs(ea_res)),
                alpha=0.1, color='#DC2626', label=f'EA−FDFD mean band')

ax.set_xlabel('Mode index', fontsize=10)
ax.set_ylabel('Δω = ω(test) − ω(res=40)  (×10⁻⁶)', fontsize=10)
ax.set_title('(b)  Eigenvalue drift vs resolution', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# (c) Summary: convergence bar chart
ax = axes[2]
labels = []
values_drift = []
values_ea = []
for res in resolutions:
    w = fdfd_by_res[res]
    n_match = min(len(env_freqs), len(w))
    cost = np.abs(env_freqs[:n_match, None] - w[None, :])
    r, c = linear_sum_assignment(cost)
    values_ea.append(np.mean(np.abs(env_freqs[r] - w[c])) * 1e6)
    labels.append(f'res={res}')

    # Drift relative to res=40 (highest)
    if res != 40:
        f40 = fdfd_by_res[40]
        nm = min(len(w), len(f40))
        mr, mt = hungarian_match(extract_window(results[res], env_min, env_max)[:nm],
                                  extract_window(results[40], env_min, env_max)[:nm])
        values_drift.append(np.mean(np.abs(mr - mt)) * 1e6)
    else:
        values_drift.append(0)

x = np.arange(len(labels))
w_bar = 0.35
ax.bar(x - w_bar/2, values_ea, w_bar, color='#DC2626', alpha=0.8, label='EA−FDFD residual')
ax.bar(x + w_bar/2, values_drift, w_bar, color='#2563EB', alpha=0.8, label='FDFD drift (vs res=40)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Mean |Δω|  (×10⁻⁶  c/a)', fontsize=10)
ax.set_title('(c)  EA error vs FDFD discretization error', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

fig.suptitle(f'FDFD Resolution Convergence  |  θ = {theta_deg:.2f}°  |  (30,29)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_resolution_convergence.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_resolution_convergence.pdf'), bbox_inches='tight')
print(f"\nSaved fig_resolution_convergence.{{png,pdf}}")
