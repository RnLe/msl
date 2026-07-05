#!/usr/bin/env python3
"""
3-solver comparison at 8° — MPB vs FDFD-CHOLMOD vs FDFD-Hybrid.

(14,1) θ=8.17°, 64 px/cell, 30 modes, σ=0.02 a/2πc.
Each solver runs in an isolated subprocess for clean peak-RSS.
"""

import json, os, subprocess, sys, time
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'

PYTHON = sys.executable
CWD = os.path.dirname(os.path.abspath(__file__))

M, N_IDX = 14, 1
PX_PER_CELL = 64
N_MODES = 30
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
# Subprocess scripts
# ════════════════════════════════════════════════════════════════

MPB_SCRIPT = r'''
import json, os, sys, time, resource
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

m, n, n_modes, px = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

L1 = np.array([m, n], dtype=float)
L2 = np.array([-n, m], dtype=float)
L = np.sqrt(L1 @ L1)
theta = 2 * np.arctan2(n, m)
mpb_res = px * round(L)

import meep as mp
from meep import mpb

c, s = np.cos(theta), np.sin(theta)
R_mat = np.array([[c, -s], [s, c]])
B_inv = np.linalg.inv(np.column_stack([L1, L2]))
r_mpb = 0.2 / L

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1[0], L1[1], 0),
    basis2=mp.Vector3(L2[0], L2[1], 0))

geometry = []
for layer_rot in [np.eye(2), R_mat]:
    a1 = layer_rot @ np.array([1.0, 0.0])
    a2 = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-m - 2, m + n + 2):
        for i2 in range(-n - 2, m + n + 2):
            pos = i1 * a1 + i2 * a2
            frac = B_inv @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(radius=r_mpb,
                center=mp.Vector3(f1, f2, 0),
                material=mp.Medium(epsilon=8.9)))

mp.verbosity(0)
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=1.0),
    num_bands=n_modes, resolution=mpb_res,
    k_points=[mp.Vector3(0, 0, 0)])

fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)
t0 = time.time()
ms.run_tm()
t_solve = time.time() - t0
os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs_raw = np.array(ms.all_freqs)[0]
freqs = (freqs_raw / L).tolist()

print('###JSON###' + json.dumps({"rss_mb": rss_kb/1024, "t_solve": t_solve,
                   "grid": mpb_res, "freqs": freqs}), flush=True)
'''

CHOLMOD_SCRIPT = r'''
import json, os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

m, n, n_modes, px = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sigma_omega = float(sys.argv[5])
wdir = sys.argv[6]

sys.path.insert(0, wdir)
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator

L1 = np.array([m, n], dtype=float)
L = np.sqrt(L1 @ L1)
N_grid = px * round(L)

eps_grid, info = build_supercell_eps(
    lattice_type='square', m=m, n=n,
    r_over_a=0.2, eps_rod=8.9, eps_bg=1.0,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)

L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                           polarization='tm')
del eps_grid; gc.collect()

sigma = (2 * np.pi * sigma_omega) ** 2
N_dof = L_op.shape[0]
L_shifted = L_op - sigma * sp.eye(N_dof, format='csc')

from sksparse.cholmod import cholesky
t0 = time.time()
factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L_op.dtype)
evals, _ = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                 OPinv=OPinv, maxiter=20000, tol=1e-10)
t_solve = time.time() - t0

del factor, OPinv, L_shifted, L_op; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
evals = np.sort(evals)
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print('###JSON###' + json.dumps({"rss_mb": rss_kb/1024, "t_solve": t_solve,
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

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
    maxiter_inner=300,
    verbose=True,
)

del eps_grid; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print('###JSON###' + json.dumps({"rss_mb": rss_kb/1024, "t_solve": timings['total'],
                   "t_assembly": timings['assembly'],
                   "t_eigsh": timings['solve'],
                   "inner_calls": timings['inner_calls'],
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

# ════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════

def run_solver(name: str, script: str, extra_args: list = None,
               timeout: int = 1800) -> dict:
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
        for line in err[-8:]:
            print(f"    {line}")
        return None
    lines = proc.stdout.strip().split('\n')
    json_line = None
    for line in reversed(lines):
        if line.startswith('###JSON###'):
            json_line = line[len('###JSON###'):]
            break
    if json_line is None:
        print(f"FAILED (no JSON in stdout)")
        for line in lines[-5:]:
            print(f"    stdout: {line}")
        err = proc.stderr.strip().split('\n')
        for line in err[-5:]:
            print(f"    stderr: {line}")
        return None
    result = json.loads(json_line)
    result.setdefault('t_solve', wall)
    print(f"RSS={result['rss_mb']:.0f} MB  "
          f"t={result['t_solve']:.1f}s  (wall {wall:.0f}s)")
    return result


# ════════════════════════════════════════════════════════════════
# Run all three
# ════════════════════════════════════════════════════════════════

print("=" * 70)
r_mpb = run_solver("MPB     ", MPB_SCRIPT)
r_chol = run_solver("CHOLMOD ", CHOLMOD_SCRIPT, [str(SIGMA_OMEGA), CWD])
r_hyb = run_solver("Hybrid  ", HYBRID_SCRIPT, [str(SIGMA_OMEGA), CWD])
print("=" * 70)

# ════════════════════════════════════════════════════════════════
# Compare eigenvalues
# ════════════════════════════════════════════════════════════════

solvers = {}
if r_mpb:
    solvers['MPB'] = np.array(r_mpb['freqs'])
if r_chol:
    solvers['CHOLMOD'] = np.array(r_chol['freqs'])
if r_hyb:
    solvers['Hybrid'] = np.array(r_hyb['freqs'])

ref_name = 'CHOLMOD' if 'CHOLMOD' in solvers else 'MPB'
ref = solvers.get(ref_name)

if ref is not None:
    print(f"\nResiduals vs {ref_name}:")
    for name, freqs in solvers.items():
        if name == ref_name:
            continue
        nc = min(len(freqs), len(ref))
        delta = np.abs(freqs[:nc] - ref[:nc])
        rel = np.where(ref[:nc] > 1e-12, delta / ref[:nc], 0.0)
        print(f"  {name:10s}: max|Δω|={delta.max():.2e}  "
              f"max|Δω/ω|={rel.max():.2e}  mean|Δω/ω|={rel.mean():.2e}")

# ════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════

print(f"\n{'Solver':>10s} {'RSS MB':>8s} {'time s':>8s}")
print("-" * 30)
for name, r in [('MPB', r_mpb), ('CHOLMOD', r_chol), ('Hybrid', r_hyb)]:
    if r:
        print(f"{name:>10s} {r['rss_mb']:8.0f} {r['t_solve']:8.1f}")

# ════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════

outfile = os.path.join(CWD, 'compare_3solver_8deg.npz')
save_dict = {
    'm': M, 'n': N_IDX, 'theta_deg': THETA_DEG, 'grid': GRID,
    'px_per_cell': PX_PER_CELL, 'n_modes': N_MODES,
    'sigma_omega': SIGMA_OMEGA,
}
for name, r in [('mpb', r_mpb), ('cholmod', r_chol), ('hybrid', r_hyb)]:
    if r:
        save_dict[f'freqs_{name}'] = np.array(r['freqs'])
        save_dict[f'rss_{name}'] = r['rss_mb']
        save_dict[f'time_{name}'] = r['t_solve']
np.savez(outfile, **save_dict)
print(f"\nSaved → {outfile}")

# ════════════════════════════════════════════════════════════════
# Plot
# ════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

COLORS = {'MPB': '#1f77b4', 'CHOLMOD': '#2ca02c', 'Hybrid': '#d62728'}
MARKERS = {'MPB': 'o', 'CHOLMOD': 's', 'Hybrid': 'D'}

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.15, 3], wspace=0.05)

# ── 3 eigenvalue ladders ───────────────────────────────────────
ladder_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
y_max = 0
for freqs_arr in solvers.values():
    y_max = max(y_max, freqs_arr.max())
y_hi = y_max * 1.05

for ax, (name, freqs) in zip(ladder_axes, solvers.items()):
    color = COLORS[name]
    for f in freqs:
        ax.plot([0.15, 0.85], [f, f], '-', color=color, lw=0.9, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(0, y_hi)
    ax.set_title(name, color=color, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    if ax != ladder_axes[0]:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Frequency  ω  [a / 2πc]')

# ── Mode index plot ────────────────────────────────────────────
ax_mode = fig.add_subplot(gs[0, 4])
for name, freqs in solvers.items():
    idx = np.arange(1, len(freqs) + 1)
    ax_mode.plot(idx, freqs, f'{MARKERS[name]}-', ms=4, lw=1,
                 color=COLORS[name], label=name)
ax_mode.set_xlabel('Mode index')
ax_mode.set_ylabel('Frequency  ω  [a / 2πc]')
ax_mode.set_title('Sorted eigenvalues')
ax_mode.set_xlim(0, N_MODES + 1)
ax_mode.set_ylim(0, y_hi)
ax_mode.legend(fontsize=10)
ax_mode.grid(True, alpha=0.3)

# ── Suptitle with stats ────────────────────────────────────────
stats = []
for name, r in [('MPB', r_mpb), ('CHOLMOD', r_chol), ('Hybrid', r_hyb)]:
    if r:
        stats.append(f"{name}: {r['rss_mb']:.0f} MB, {r['t_solve']:.1f}s")

fig.suptitle(
    f"({M},{N_IDX}) θ={THETA_DEG:.2f}°, {PX_PER_CELL} px/cell, "
    f"{N_MODES} modes, σ={SIGMA_OMEGA}\n" + "  |  ".join(stats),
    fontsize=11, fontweight='bold',
)
fig.tight_layout()

figfile = os.path.join(CWD, 'compare_3solver_8deg.png')
fig.savefig(figfile, dpi=180, bbox_inches='tight')
plt.close(fig)
print(f"Plot → {figfile}")

# ── RAM bar chart ───────────────────────────────────────────────
fig2, (ax_ram, ax_time) = plt.subplots(1, 2, figsize=(10, 4))

names_ok = []
rss_vals = []
time_vals = []
colors_ok = []
for name, r in [('MPB', r_mpb), ('CHOLMOD', r_chol), ('Hybrid', r_hyb)]:
    if r:
        names_ok.append(name)
        rss_vals.append(r['rss_mb'])
        time_vals.append(r['t_solve'])
        colors_ok.append(COLORS[name])

ax_ram.bar(names_ok, rss_vals, color=colors_ok, alpha=0.8)
ax_ram.set_ylabel('Peak RSS [MB]')
ax_ram.set_title('Peak RAM')
for i, v in enumerate(rss_vals):
    ax_ram.text(i, v + 10, f'{v:.0f}', ha='center', fontsize=10)
ax_ram.grid(True, alpha=0.2, axis='y')

ax_time.bar(names_ok, time_vals, color=colors_ok, alpha=0.8)
ax_time.set_ylabel('Solve time [s]')
ax_time.set_title('Solve time')
for i, v in enumerate(time_vals):
    ax_time.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)
ax_time.grid(True, alpha=0.2, axis='y')

fig2.suptitle(f"RAM & Time — ({M},{N_IDX}) θ={THETA_DEG:.2f}°, "
              f"{N_MODES} modes", fontweight='bold')
fig2.tight_layout()
figfile2 = os.path.join(CWD, 'compare_3solver_8deg_bars.png')
fig2.savefig(figfile2, dpi=180, bbox_inches='tight')
plt.close(fig2)
print(f"Plot → {figfile2}")
