#!/usr/bin/env python3
"""
Memory scaling study: FDFD vs MPB at 64 px/cell, Gamma point, TM, hex rods.
Measures peak RSS for commensurate angles from ~10° down to ~4°.
50 modes, 1 k-point.

Runs each solver in a subprocess to get clean RSS measurements.
"""
import subprocess
import sys
import os
import json
import math
import numpy as np

PYTHON = sys.executable
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data_gamma_tm_hex')
os.makedirs(DATA_DIR, exist_ok=True)

# Commensurate angles chosen for ~10° to ~4° in ~1° steps
# (m, n, theta_approx)
CASES = [
    (19, 14,  10.0),  # N=823,  θ=9.999°
    (25, 19,   9.0),  # N=1461, θ=9.003°
    (37, 29,   8.0),  # N=3283, θ=8.006°
    (26, 21,   7.0),  # N=1663, θ=7.029°
    ( 6,  5,   6.0),  # N=91,   θ=6.009°
    (43, 37,   5.0),  # N=4809, θ=4.959°
    (35, 31,   4.0),  # N=3271, θ=4.008°
]

RES_PER_CELL = 64
N_MODES = 50


def make_worker_script(solver, m, n, res_per_cell, n_modes):
    """Return a Python script string that runs one solver and prints JSON result."""
    return f'''
import os, sys, time, math, json, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

m, n = {m}, {n}
EPS_ROD, EPS_BG, R_OVER_A = 12.0, 1.0, 0.22
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, math.sqrt(3)/2])
N_cells = m*m + m*n + n*n
L_super = math.sqrt(N_cells)

# Correct coincidence lattice vectors
L1 = n*a1 + m*a2
L2 = -m*a1 + (n+m)*a2
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)

cos_theta = (m*m + 4*m*n + n*n) / (2.0 * N_cells)
theta_rad = math.acos(max(-1, min(1, cos_theta)))
c, s = math.cos(theta_rad), math.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])

rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if "{solver}" == "mpb":
    import meep as mp
    from meep import mpb
    r_mpb = R_OVER_A / L_super
    lattice = mp.Lattice(size=mp.Vector3(1,1,0),
        basis1=mp.Vector3(float(L1[0]),float(L1[1]),0),
        basis2=mp.Vector3(float(L2[0]),float(L2[1]),0))
    geometry = []
    seen = set()
    sr = m + n + 3
    for rot in [np.eye(2), R_mat]:
        a1_l, a2_l = rot@a1, rot@a2
        for i1 in range(-sr, sr+1):
            for i2 in range(-sr, sr+1):
                pos = i1*a1_l + i2*a2_l
                frac = B_inv @ pos
                f1, f2 = frac[0]%1.0, frac[1]%1.0
                if f1>=0.5: f1-=1.0
                if f2>=0.5: f2-=1.0
                key = (round(f1,6), round(f2,6))
                if key not in seen and abs(f1)<=0.5 and abs(f2)<=0.5:
                    seen.add(key)
                    geometry.append(mp.Cylinder(radius=r_mpb,
                        center=mp.Vector3(f1,f2,0),
                        material=mp.Medium(epsilon=EPS_ROD)))
    mpb_res = int({res_per_cell} * round(L_super))
    mp.verbosity(0)
    ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands={n_modes}, resolution=mpb_res, k_points=[mp.Vector3(0,0,0)])
    t0 = time.time()
    ms.run_tm()
    dt = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    freqs = np.array(ms.all_freqs)[0] / L_super
    Nx = mpb_res
    
elif "{solver}" == "fdfd":
    sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'moire_envelope', 'thesis_results')))
    from T_direct_validation.fdfd_solver import solve_fdfd_supercell
    from T_direct_validation.supercell_geometry import build_supercell_eps
    
    Nx = int(round(L_super * {res_per_cell}))
    eps_grid, info = build_supercell_eps('hex', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=Nx, Ny=Nx, subpixel_smoothing=True, smoothing_Nsub=8)
    t0 = time.time()
    evals, _ = solve_fdfd_supercell(eps_grid, info,
        q_vec=np.array([0.0, 0.0]), n_modes={n_modes})
    dt = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    freqs = np.sqrt(np.maximum(evals, 0)) / (2*math.pi)

peak_rss_mb = rss_after / 1024  # ru_maxrss is KB on Linux
result = dict(
    solver="{solver}", m=m, n=n, N_cells=N_cells,
    L_super=L_super, Nx=Nx, DOF=Nx*Nx,
    theta_deg=math.degrees(theta_rad),
    n_modes={n_modes}, res_per_cell={res_per_cell},
    peak_rss_mb=peak_rss_mb, time_s=dt,
    freq_band2=float(freqs[1]) if len(freqs)>1 else 0.0
)
print("RESULT_JSON:" + json.dumps(result))
'''


def run_one(solver, m, n):
    """Run a single solver in a subprocess and return the result dict."""
    script = make_worker_script(solver, m, n, RES_PER_CELL, N_MODES)
    N = m*m + m*n + n*n
    L = math.sqrt(N)
    Nx = int(round(L * RES_PER_CELL)) if solver == 'fdfd' else int(RES_PER_CELL * round(L))
    theta = math.degrees(math.acos((m*m+4*m*n+n*n)/(2.0*N)))
    
    print(f'  {solver.upper():4s} (m={m},n={n}) θ={theta:.2f}° N={N} Nx={Nx} ...', end=' ', flush=True)
    
    try:
        result = subprocess.run(
            [PYTHON, '-c', script],
            capture_output=True, text=True, timeout=3600,
            cwd=STUDY_DIR)
        
        if result.returncode != 0:
            print(f'FAILED (rc={result.returncode})')
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-3:]:
                print(f'    {line}')
            return None
        
        for line in result.stdout.split('\n'):
            if line.startswith('RESULT_JSON:'):
                data = json.loads(line[len('RESULT_JSON:'):])
                print(f'{data["peak_rss_mb"]:.0f} MB, {data["time_s"]:.1f}s')
                return data
        
        print('NO RESULT (no JSON found)')
        return None
        
    except subprocess.TimeoutExpired:
        print('TIMEOUT (>3600s)')
        return None


def main():
    results = []
    
    for m, n, tgt in CASES:
        N = m*m + m*n + n*n
        L = math.sqrt(N)
        theta = math.degrees(math.acos((m*m+4*m*n+n*n)/(2.0*N)))
        print(f'\n--- Target ~{tgt:.0f}°: (m={m},n={n}) θ={theta:.2f}° N={N} L={L:.1f} ---')
        
        for solver in ['fdfd', 'mpb']:
            r = run_one(solver, m, n)
            if r is not None:
                results.append(r)
    
    # Save results
    outpath = os.path.join(DATA_DIR, 'memory_scaling_64px.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {len(results)} results to {outpath}')
    
    # Print summary table
    print(f'\n{"Solver":>6} {"θ°":>7} {"N":>6} {"Nx":>6} {"DOF":>10} {"RSS MB":>8} {"Time s":>8}')
    print('-' * 60)
    for r in sorted(results, key=lambda x: (-x['theta_deg'], x['solver'])):
        print(f'{r["solver"]:>6} {r["theta_deg"]:>7.2f} {r["N_cells"]:>6} '
              f'{r["Nx"]:>6} {r["DOF"]:>10} {r["peak_rss_mb"]:>8.0f} {r["time_s"]:>8.1f}')


if __name__ == '__main__':
    main()
