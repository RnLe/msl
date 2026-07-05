#!/usr/bin/env python3
"""
Compare MPB (res=32, existing) vs FDFD-Hybrid (res=64) at 3° with 50 modes.
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
# Load existing MPB data (res=32)
# ════════════════════════════════════════════════════════════════

mpb_file = os.path.join(CWD, 'supercell_3deg_50modes_comparison.npz')
mpb_data = np.load(mpb_file)
freqs_mpb = mpb_data['freqs_mpb'][:N_MODES]
mpb_res = int(mpb_data['res'])
print(f"Loaded MPB data: {len(freqs_mpb)} modes, res={mpb_res}")
print(f"  MPB ω range: [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]")
print()

# ════════════════════════════════════════════════════════════════
# Hybrid subprocess script
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
    tol_inner=1e-10,
    maxiter_eigsh=20000,
    maxiter_inner=500,
    verbose=True,
)

del eps_grid; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print('###JSON###' + json.dumps({"rss_mb": rss_kb/1024, "t_solve": timings['total'],
                   "t_assembly": timings['assembly'],
                   "t_eigsh": timings['solve'],
                   "inner_calls": timings['inner_calls'],
                   "inner_failures": timings.get('inner_failures', 0),
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

def run_solver(name, script, extra_args=None, timeout=7200):
    args = [PYTHON, '-c', script,
            str(M), str(N_IDX), str(N_MODES), str(PX_PER_CELL)]
    if extra_args:
        args.extend(extra_args)
    print(f"  {name} ...", end=' ', flush=True)
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True,
                          cwd=CWD, timeout=timeout)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"FAILED (rc={proc.returncode})")
        err = proc.stderr.strip().split('\n')
        for line in err[-10:]:
            print(f"    {line}")
        return None
    lines = proc.stdout.strip().split('\n')
    json_line = None
    for line in reversed(lines):
        if line.startswith('###JSON###'):
            json_line = line[len('###JSON###'):]
            break
    if json_line is None:
        print(f"FAILED (no JSON)")
        for line in lines[-5:]:
            print(f"    stdout: {line}")
        return None
    result = json.loads(json_line)
    result.setdefault('t_solve', wall)
    print(f"RSS={result['rss_mb']:.0f} MB  t={result['t_solve']:.1f}s  "
          f"(wall {wall:.0f}s)")
    return result

# ════════════════════════════════════════════════════════════════
# Run Hybrid
# ════════════════════════════════════════════════════════════════

print("=" * 70)
r_hyb = run_solver("FDFD-Hybrid (res=64)", HYBRID_SCRIPT,
                   [str(SIGMA_OMEGA), CWD], timeout=7200)
print("=" * 70)

if r_hyb is None:
    print("Hybrid solver failed!")
    sys.exit(1)

freqs_hybrid = np.array(r_hyb['freqs'])

# ════════════════════════════════════════════════════════════════
# Compare
# ════════════════════════════════════════════════════════════════

nc = min(len(freqs_mpb), len(freqs_hybrid))
delta = np.abs(freqs_hybrid[:nc] - freqs_mpb[:nc])
# Skip near-zero modes for relative error
mask = freqs_mpb[:nc] > 1e-4
rel = np.where(mask, delta / freqs_mpb[:nc], 0.0)

print(f"\nMPB (res={mpb_res}) vs Hybrid (res={PX_PER_CELL}):")
print(f"  max|Δω|      = {delta.max():.2e}")
print(f"  max|Δω/ω|    = {rel[mask].max():.2e}  (excluding ω<1e-4)")
print(f"  mean|Δω/ω|   = {rel[mask].mean():.2e}")

print(f"\n{'Solver':>20s} {'res':>5s} {'RSS MB':>8s} {'time s':>8s}")
print("-" * 45)
print(f"{'MPB':>20s} {mpb_res:>5d} {'(prev)':>8s} {'(prev)':>8s}")
print(f"{'FDFD-Hybrid':>20s} {PX_PER_CELL:>5d} {r_hyb['rss_mb']:>8.0f} "
      f"{r_hyb['t_solve']:>8.1f}")

# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════

outfile = os.path.join(CWD, 'compare_mpb_fdfd_3deg_hybrid.npz')
np.savez(outfile,
    freqs_mpb=freqs_mpb, freqs_hybrid=freqs_hybrid,
    mpb_res=mpb_res, hybrid_res=PX_PER_CELL,
    m=M, n=N_IDX, theta_deg=THETA_DEG,
    grid=GRID, n_modes=N_MODES,
    sigma_omega=SIGMA_OMEGA,
    rss_hybrid=r_hyb['rss_mb'],
    time_hybrid=r_hyb['t_solve'],
    inner_calls=r_hyb.get('inner_calls', 0),
    inner_failures=r_hyb.get('inner_failures', 0),
)
print(f"\nSaved → {outfile}")

# ════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

COLORS = {'MPB': '#1f77b4', 'Hybrid': '#d62728'}

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.15, 3], wspace=0.05)

# ── 2 eigenvalue ladders ───────────────────────────────────────
solvers = {'MPB (res=32)': freqs_mpb, f'FDFD-Hybrid (res={PX_PER_CELL})': freqs_hybrid}
colors = {'MPB (res=32)': COLORS['MPB'],
          f'FDFD-Hybrid (res={PX_PER_CELL})': COLORS['Hybrid']}
markers = {'MPB (res=32)': 'o',
           f'FDFD-Hybrid (res={PX_PER_CELL})': 'D'}

y_max = max(freqs_mpb.max(), freqs_hybrid.max())
y_hi = y_max * 1.05

ladder_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
for ax, (name, freqs) in zip(ladder_axes, solvers.items()):
    color = colors[name]
    for f in freqs:
        ax.plot([0.15, 0.85], [f, f], '-', color=color, lw=0.8, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(0, y_hi)
    ax.set_title(name, color=color, fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    if ax != ladder_axes[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Frequency  ω  [a / 2πc]')

# ── Mode index plot ────────────────────────────────────────────
ax_mode = fig.add_subplot(gs[0, 3])
for name, freqs in solvers.items():
    idx = np.arange(1, len(freqs) + 1)
    ax_mode.plot(idx, freqs, f'{markers[name]}-', ms=4, lw=1,
                 color=colors[name], label=name)
ax_mode.set_xlabel('Mode index')
ax_mode.set_ylabel('Frequency  ω  [a / 2πc]')
ax_mode.set_title('Sorted eigenvalues')
ax_mode.set_xlim(0, N_MODES + 1)
ax_mode.set_ylim(0, y_hi)
ax_mode.legend(fontsize=9)
ax_mode.grid(True, alpha=0.3)

# ── Suptitle ───────────────────────────────────────────────────
fig.suptitle(
    f"({M},{N_IDX}) θ={THETA_DEG:.2f}°, {N_MODES} modes at Γ\n"
    f"MPB res={mpb_res}  |  FDFD-Hybrid res={PX_PER_CELL}, "
    f"grid={GRID}, DOF={DOF:,}  |  "
    f"RSS={r_hyb['rss_mb']:.0f} MB, t={r_hyb['t_solve']:.1f}s",
    fontsize=11, fontweight='bold',
)
fig.tight_layout()

figfile = os.path.join(CWD, 'compare_mpb_fdfd_3deg_hybrid.png')
fig.savefig(figfile, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Plot → {figfile}")

# ── Per-mode comparison table (first 15 + last 5) ──────────────
print(f"\n{'Mode':>5s} {'MPB':>12s} {'Hybrid':>12s} {'|Δω|':>10s} {'|Δω/ω|':>10s}")
print("-" * 55)
show = list(range(min(15, nc))) + list(range(max(15, nc-5), nc))
for i in sorted(set(show)):
    d = abs(freqs_hybrid[i] - freqs_mpb[i])
    r = d / freqs_mpb[i] if freqs_mpb[i] > 1e-8 else 0
    print(f"{i+1:5d} {freqs_mpb[i]:12.8f} {freqs_hybrid[i]:12.8f} "
          f"{d:10.2e} {r:10.2e}")
