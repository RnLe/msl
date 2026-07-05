#!/usr/bin/env python3
"""
RAM benchmark: MPB vs FDFD across angles, mode counts, and resolutions.

Runs each solve in an isolated subprocess for clean peak-RSS measurement.
Square lattice: r/a=0.2, eps_rod=8.9, eps_bg=1.0, TM polarization.

Angles:      (8,1)≈14.25°  (9,1)≈12.68°  (11,1)≈10.39°  (14,1)≈8.17°
Modes:       10, 20, 30
px/cell:     32, 64
Total runs:  4 × 3 × 2 = 24 per solver, 48 total.
"""

import json, os, subprocess, sys, tempfile, time
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'

# ── Configuration ───────────────────────────────────────────────
ANGLES = [(8, 1), (9, 1), (11, 1), (14, 1)]
MODE_COUNTS = [10, 20, 30]
PX_CELLS = [32, 64]
PYTHON = sys.executable

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Child scripts write JSON to a temp file (avoids stdout pollution) ─

MPB_CHILD = r'''
import json, os, sys, time, resource, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'

m, n, n_modes, px, outpath = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
R, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0

L1 = np.array([m, n], dtype=float)
L2 = np.array([-n, m], dtype=float)
L = np.sqrt(L1 @ L1)
theta = 2 * np.arctan2(n, m)
mpb_res = px * round(L)

# Redirect stdout/stderr to devnull BEFORE importing meep
_devnull = open(os.devnull, 'w')
os.dup2(_devnull.fileno(), 1)
os.dup2(_devnull.fileno(), 2)

import meep as mp
from meep import mpb

c, s = np.cos(theta), np.sin(theta)
R_mat = np.array([[c, -s], [s, c]])
B_inv = np.linalg.inv(np.column_stack([L1, L2]))
r_mpb = R / L

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
                material=mp.Medium(epsilon=EPS_ROD)))

mp.verbosity(0)
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=n_modes, resolution=mpb_res,
    k_points=[mp.Vector3(0, 0, 0)])

t0 = time.time()
ms.run_tm()
t_solve = time.time() - t0

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs_raw = np.array(ms.all_freqs)[0]
freqs = (freqs_raw / L).tolist()

result = {"rss_mb": rss_kb / 1024, "t_solve": t_solve, "grid": mpb_res, "freqs": freqs}
with open(outpath, 'w') as f:
    json.dump(result, f)
'''

FDFD_CHILD = r'''
import json, os, sys, time, resource, gc, numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
os.environ['OMP_NUM_THREADS'] = '1'

m, n, n_modes, px, outpath = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
R, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
SIGMA_OMEGA = 0.01

t0_total = time.time()

L1 = np.array([m, n], dtype=float)
L2 = np.array([-n, m], dtype=float)
L = np.sqrt(L1 @ L1)
N_grid = px * round(L)

sys.path.insert(0, os.path.join(os.path.dirname(__file__) or '.'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator

eps_grid, info = build_supercell_eps(
    lattice_type='square', m=m, n=n,
    r_over_a=R, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)

L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                           polarization='tm')
del eps_grid; gc.collect()

sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
N_dof = L_op.shape[0]
L_shifted = L_op - sigma * sp.eye(N_dof, format='csc')

try:
    from sksparse.cholmod import cholesky
    factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
    OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L_op.dtype)
    evals, _ = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                     OPinv=OPinv, maxiter=20000, tol=1e-10)
    del factor, OPinv
except ImportError:
    evals, _ = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                     maxiter=20000, tol=1e-10)

del L_op, L_shifted; gc.collect()

t_total = time.time() - t0_total
rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

evals = np.sort(evals)
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

result = {"rss_mb": rss_kb / 1024, "t_solve": t_total, "grid": N_grid, "freqs": freqs}
with open(outpath, 'w') as f:
    json.dump(result, f)
'''

# ── Harness ─────────────────────────────────────────────────────
def run_one(solver: str, m: int, n: int, n_modes: int, px: int) -> dict:
    """Run a single benchmark in an isolated subprocess, return results via temp file."""
    script = MPB_CHILD if solver == 'mpb' else FDFD_CHILD
    fd, tmppath = tempfile.mkstemp(suffix='.json', prefix='bench_')
    os.close(fd)
    try:
        args = [PYTHON, '-c', script, str(m), str(n), str(n_modes), str(px), tmppath]
        t0 = time.time()
        proc = subprocess.run(args, capture_output=True, text=True, cwd=OUTDIR,
                              timeout=600)
        wall = time.time() - t0
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode})")
            stderr_snip = proc.stderr.strip()[-300:] if proc.stderr else ''
            if stderr_snip:
                print(f"  stderr: ...{stderr_snip}")
            return {"rss_mb": -1, "t_solve": wall, "grid": -1, "freqs": []}
        with open(tmppath) as f:
            result = json.load(f)
        result.setdefault('t_solve', wall)
        return result
    finally:
        if os.path.exists(tmppath):
            os.unlink(tmppath)


def main():
    results = []
    total_runs = 2 * len(ANGLES) * len(MODE_COUNTS) * len(PX_CELLS)
    run_idx = 0

    for solver in ['mpb', 'fdfd']:
        for px in PX_CELLS:
            for (m, n) in ANGLES:
                theta = np.degrees(2 * np.arctan2(n, m))
                L = np.sqrt(m**2 + n**2)
                grid = px * round(L)
                for n_modes in MODE_COUNTS:
                    run_idx += 1
                    tag = (f"[{run_idx}/{total_runs}] {solver.upper():4s} "
                           f"({m},{n}) θ={theta:5.2f}° {px}px {n_modes:2d}m")
                    print(f"{tag}  grid={grid} ...", end=' ', flush=True)

                    r = run_one(solver, m, n, n_modes, px)
                    rss = r['rss_mb']
                    t = r['t_solve']
                    print(f"RSS={rss:.0f} MB  t={t:.1f}s")

                    results.append({
                        'solver': solver, 'm': m, 'n': n,
                        'theta_deg': theta, 'n_modes': n_modes,
                        'px_per_cell': px,
                        'grid': grid, 'dof': grid**2,
                        'rss_mb': rss, 't_solve': t,
                    })

    # ── Summary table ───────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'Solver':>6s} {'(m,n)':>6s} {'θ°':>7s} {'px':>4s} {'grid':>6s} "
          f"{'DOF':>10s} {'modes':>5s} {'RSS MB':>8s} {'time s':>8s}")
    print("-" * 95)
    for r in results:
        print(f"{r['solver'].upper():>6s} ({r['m']},{r['n']})"
              f" {r['theta_deg']:7.2f} {r['px_per_cell']:4d} {r['grid']:6d}"
              f" {r['dof']:10,d} {r['n_modes']:5d}"
              f" {r['rss_mb']:8.0f} {r['t_solve']:8.1f}")

    # ── Save ────────────────────────────────────────────────────
    outfile = os.path.join(OUTDIR, 'ram_benchmark.npz')

    solvers  = np.array([r['solver'] for r in results])
    ms       = np.array([r['m'] for r in results])
    ns       = np.array([r['n'] for r in results])
    thetas   = np.array([r['theta_deg'] for r in results])
    modes    = np.array([r['n_modes'] for r in results])
    pxs      = np.array([r['px_per_cell'] for r in results])
    grids    = np.array([r['grid'] for r in results])
    dofs     = np.array([r['dof'] for r in results])
    rss_vals = np.array([r['rss_mb'] for r in results])
    times    = np.array([r['t_solve'] for r in results])

    np.savez(outfile, solvers=solvers, m=ms, n=ns, theta_deg=thetas,
             n_modes=modes, px_per_cell=pxs, grid=grids, dof=dofs,
             rss_mb=rss_vals, t_solve=times)
    print(f"\nSaved → {outfile}")

    # ── Plot: 2×2  [RAM vs DOF, Time vs DOF] × [32px, 64px] ───
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col')

    for col, px in enumerate(PX_CELLS):
        # RAM
        ax = axes[0, col]
        for solver, color, marker in [('mpb', '#1f77b4', 'o'), ('fdfd', '#d62728', 's')]:
            for nm, ls in [(10, '--'), (20, '-'), (30, '-.')]:
                mask = (solvers == solver) & (modes == nm) & (pxs == px)
                if mask.sum() == 0:
                    continue
                order = np.argsort(dofs[mask])
                ax.plot(dofs[mask][order], rss_vals[mask][order],
                        f'{marker}{ls}', color=color, ms=5, lw=1.2,
                        label=f'{solver.upper()} {nm}m')
        ax.set_ylabel('Peak RSS  [MB]')
        ax.set_title(f'{px} px/cell — Peak RAM')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Time
        ax = axes[1, col]
        for solver, color, marker in [('mpb', '#1f77b4', 'o'), ('fdfd', '#d62728', 's')]:
            for nm, ls in [(10, '--'), (20, '-'), (30, '-.')]:
                mask = (solvers == solver) & (modes == nm) & (pxs == px)
                if mask.sum() == 0:
                    continue
                order = np.argsort(dofs[mask])
                ax.plot(dofs[mask][order], times[mask][order],
                        f'{marker}{ls}', color=color, ms=5, lw=1.2,
                        label=f'{solver.upper()} {nm}m')
        ax.set_xlabel('DOF (grid²)')
        ax.set_ylabel('Solve time  [s]')
        ax.set_title(f'{px} px/cell — Solve time')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('RAM & Time benchmark — MPB vs FDFD (CHOLMOD), square lattice',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    figfile = os.path.join(OUTDIR, 'ram_benchmark.png')
    fig.savefig(figfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {figfile}")


if __name__ == '__main__':
    main()
