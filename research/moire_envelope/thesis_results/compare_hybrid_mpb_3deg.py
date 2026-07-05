#!/usr/bin/env python3
"""
FDFD Hybrid vs MPB at ~3° — (38,1), FDFD res=64/cell, 50 TM modes at Γ.

Loads existing MPB res=32 data from supercell_3deg_50modes_comparison.npz.
Runs FDFD Hybrid (MINRES) in a subprocess for clean peak-RSS measurement.
Produces comparison plot.
"""

import json, os, subprocess, sys, time
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'

PYTHON = sys.executable
CWD = os.path.dirname(os.path.abspath(__file__))

M, N_IDX = 38, 1
PX_PER_CELL = 64
N_MODES = 50
SIGMA_OMEGA = 0.02

L1 = np.array([M, N_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
THETA_DEG = np.degrees(2 * np.arctan2(N_IDX, M))
GRID = PX_PER_CELL * round(L_SUPER)
DOF = GRID ** 2

print(f"Supercell: ({M},{N_IDX}), θ={THETA_DEG:.2f}°, grid={GRID}, "
      f"DOF={DOF:,}, modes={N_MODES}, σ_ω={SIGMA_OMEGA}")
print()

# ════════════════════════════════════════════════════════════════
# Load existing MPB data
# ════════════════════════════════════════════════════════════════

mpb_file = os.path.join(CWD, 'supercell_3deg_50modes_comparison.npz')
if not os.path.exists(mpb_file):
    print(f"ERROR: MPB data not found: {mpb_file}")
    sys.exit(1)

mpb_data = np.load(mpb_file)
freqs_mpb = mpb_data['freqs_mpb'][:N_MODES]
t_mpb = float(mpb_data['t_mpb'])
mpb_res = int(mpb_data['res'])
print(f"Loaded MPB data: res={mpb_res}, {len(freqs_mpb)} modes, "
      f"t={t_mpb:.1f}s")
print(f"  ω range: [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]")
print()

# ════════════════════════════════════════════════════════════════
# FDFD Hybrid subprocess
# ════════════════════════════════════════════════════════════════

HYBRID_SCRIPT = r'''
import json, os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

m, n, n_modes, px = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sigma_omega = float(sys.argv[5])
wdir = sys.argv[6]

sys.path.insert(0, wdir)
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import solve_fdfd_hybrid

L1 = np.array([m, n], dtype=float)
L = np.sqrt(L1 @ L1)
N_grid = px * round(L)

eps_grid, info = build_supercell_eps(
    lattice_type='square', m=m, n=n,
    r_over_a=0.2, eps_rod=8.9, eps_bg=1.0,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)

evals, _, timings = solve_fdfd_hybrid(
    eps_grid, info,
    q_vec=np.array([0.0, 0.0]),
    n_modes=n_modes,
    sigma_omega=sigma_omega,
    tol_eigsh=1e-10,
    tol_inner=1e-8,
    maxiter_eigsh=20000,
    maxiter_inner=200,
    inner_solver='minres',
    verbose=True,
)

del eps_grid; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print('###JSON###' + json.dumps({"rss_mb": rss_kb/1024, "t_solve": timings['total'],
                   "t_assembly": timings['assembly'],
                   "t_eigsh": timings['solve'],
                   "inner_calls": timings['inner_calls'],
                   "inner_failures": timings['inner_failures'],
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

# ════════════════════════════════════════════════════════════════
# Run Hybrid
# ════════════════════════════════════════════════════════════════

print("Running FDFD Hybrid (MINRES)...", flush=True)
args = [PYTHON, '-c', HYBRID_SCRIPT,
        str(M), str(N_IDX), str(N_MODES), str(PX_PER_CELL),
        str(SIGMA_OMEGA), CWD]

t0 = time.time()
proc = subprocess.run(args, capture_output=True, text=True,
                      cwd=CWD, timeout=3600)
wall = time.time() - t0

if proc.returncode != 0:
    print(f"FAILED (rc={proc.returncode})")
    err = proc.stderr.strip().split('\n')
    for line in err[-15:]:
        print(f"  {line}")
    sys.exit(1)

# Parse JSON
lines = proc.stdout.strip().split('\n')
json_line = None
for line in reversed(lines):
    if line.startswith('###JSON###'):
        json_line = line[len('###JSON###'):]
        break
if json_line is None:
    print("FAILED (no JSON)")
    for line in lines[-10:]:
        print(f"  stdout: {line}")
    sys.exit(1)

r_hyb = json.loads(json_line)
freqs_hybrid = np.array(r_hyb['freqs'])

print(f"  Done: RSS={r_hyb['rss_mb']:.0f} MB, "
      f"t_total={r_hyb['t_solve']:.1f}s, "
      f"t_eigsh={r_hyb['t_eigsh']:.1f}s")
print(f"  Inner calls: {r_hyb['inner_calls']} "
      f"({r_hyb['inner_failures']} failures)")
print(f"  ω range: [{freqs_hybrid[0]:.6f}, {freqs_hybrid[-1]:.6f}]")

# ════════════════════════════════════════════════════════════════
# Compare
# ════════════════════════════════════════════════════════════════

nc = min(len(freqs_mpb), len(freqs_hybrid))
delta = np.abs(freqs_hybrid[:nc] - freqs_mpb[:nc])
mask = freqs_mpb[:nc] > 1e-6
rel = np.zeros(nc)
rel[mask] = delta[mask] / freqs_mpb[:nc][mask]

print(f"\nComparison (FDFD res={PX_PER_CELL} vs MPB res={mpb_res}):")
print(f"  Modes compared: {nc}")
print(f"  max|Δω|     = {delta.max():.2e}")
print(f"  max|Δω/ω|   = {rel[mask].max():.2e}  (excluding ω≈0 modes)")
print(f"  mean|Δω/ω|  = {rel[mask].mean():.2e}")

print(f"\n{'Mode':>5s} {'MPB':>12s} {'Hybrid':>12s} {'|Δω|':>10s} {'|Δω/ω|':>10s}")
print("-" * 55)
for i in range(min(nc, 20)):
    d = abs(freqs_hybrid[i] - freqs_mpb[i])
    r = d / freqs_mpb[i] if freqs_mpb[i] > 1e-6 else 0
    print(f"{i+1:5d} {freqs_mpb[i]:12.8f} {freqs_hybrid[i]:12.8f} "
          f"{d:10.2e} {r:10.2e}")
if nc > 20:
    print(f"  ... ({nc - 20} more modes)")

# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════

outfile = os.path.join(CWD, 'compare_hybrid_mpb_3deg.npz')
np.savez(outfile,
    freqs_mpb=freqs_mpb, freqs_hybrid=freqs_hybrid,
    mpb_res=mpb_res, hybrid_res=PX_PER_CELL,
    m=M, n=N_IDX, theta_deg=THETA_DEG,
    grid=GRID, n_modes=N_MODES,
    rss_hybrid=r_hyb['rss_mb'],
    t_hybrid=r_hyb['t_solve'],
    t_mpb=t_mpb,
    inner_calls=r_hyb['inner_calls'],
)
print(f"\nSaved → {outfile}")

# ════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.15, 3], wspace=0.05)

# ── Eigenvalue ladders ──────────────────────────────────────────
COLORS = {'MPB': '#1f77b4', 'Hybrid': '#d62728'}
y_max = max(freqs_mpb.max(), freqs_hybrid.max()) * 1.05

ax_mpb = fig.add_subplot(gs[0, 0])
for f in freqs_mpb:
    ax_mpb.plot([0.15, 0.85], [f, f], '-', color=COLORS['MPB'], lw=0.9, alpha=0.7)
ax_mpb.set_xlim(0, 1); ax_mpb.set_xticks([])
ax_mpb.set_ylim(0, y_max)
ax_mpb.set_title(f'MPB (res={mpb_res})', color=COLORS['MPB'], fontweight='bold')
ax_mpb.set_ylabel('Frequency  ω  [a / 2πc]')
ax_mpb.grid(True, alpha=0.3, axis='y')

ax_hyb = fig.add_subplot(gs[0, 1])
for f in freqs_hybrid:
    ax_hyb.plot([0.15, 0.85], [f, f], '-', color=COLORS['Hybrid'], lw=0.9, alpha=0.7)
ax_hyb.set_xlim(0, 1); ax_hyb.set_xticks([])
ax_hyb.set_ylim(0, y_max)
ax_hyb.set_title(f'Hybrid (res={PX_PER_CELL})', color=COLORS['Hybrid'], fontweight='bold')
ax_hyb.set_yticklabels([])
ax_hyb.grid(True, alpha=0.3, axis='y')

# ── Mode index overlay ──────────────────────────────────────────
ax_mode = fig.add_subplot(gs[0, 3])
idx_mpb = np.arange(1, len(freqs_mpb) + 1)
idx_hyb = np.arange(1, len(freqs_hybrid) + 1)
ax_mode.plot(idx_mpb, freqs_mpb, 'o-', ms=4, lw=1,
             color=COLORS['MPB'], label=f'MPB (res={mpb_res})')
ax_mode.plot(idx_hyb, freqs_hybrid, 'D-', ms=3, lw=1,
             color=COLORS['Hybrid'], label=f'Hybrid (res={PX_PER_CELL})')
ax_mode.set_xlabel('Mode index')
ax_mode.set_ylabel('Frequency  ω  [a / 2πc]')
ax_mode.set_title('Sorted eigenvalues')
ax_mode.set_xlim(0, N_MODES + 1)
ax_mode.set_ylim(0, y_max)
ax_mode.legend(fontsize=10)
ax_mode.grid(True, alpha=0.3)

fig.suptitle(
    f"({M},{N_IDX}) θ={THETA_DEG:.2f}°, {N_MODES} modes, σ={SIGMA_OMEGA}\n"
    f"MPB: res={mpb_res}, {t_mpb:.1f}s  |  "
    f"Hybrid: res={PX_PER_CELL}, {r_hyb['rss_mb']:.0f} MB, {r_hyb['t_solve']:.1f}s",
    fontsize=11, fontweight='bold',
)
fig.tight_layout()

figfile = os.path.join(CWD, 'compare_hybrid_mpb_3deg.png')
fig.savefig(figfile, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Plot → {figfile}")
