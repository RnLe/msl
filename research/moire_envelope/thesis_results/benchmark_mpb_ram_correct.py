"""Benchmark MPB RAM with CORRECT resolution = px_per_cell * L_SUPER."""
import subprocess, sys, json

cases = []
# (m, n, px_per_cell, n_bands)
for m in [8, 11, 14, 19, 25, 30, 38]:
    for ppc in [32, 64]:
        cases.append((m, 1, ppc, 10))

# Also test 128 px/cell for smaller angles
for m in [8, 11, 14, 19]:
    cases.append((m, 1, 128, 10))

# Also test n_bands=100 at 64 px/cell for key angles
for m in [11, 19, 30, 38]:
    cases.append((m, 1, 64, 100))

SCRIPT = '''
import os, sys, time, resource, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp; from meep import mpb

m_idx, n_idx, ppc, n_bands = {m}, {n}, {ppc}, {nb}
L1 = np.array([m_idx, n_idx], dtype=float)
L2 = np.array([-n_idx, m_idx], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2*np.arctan2(n_idx, m_idx)
c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c,-s],[s,c]])
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)
r_mpb = 0.2 / L_SUPER

# CORRECT resolution: px_per_cell * round(L_SUPER)
actual_res = ppc * round(L_SUPER)

lattice = mp.Lattice(size=mp.Vector3(1,1,0),
    basis1=mp.Vector3(L1[0],L1[1],0), basis2=mp.Vector3(L2[0],L2[1],0))
geometry = []
for layer_rot in [np.eye(2), R_mat]:
    a1 = layer_rot @ np.array([1.0, 0.0])
    a2 = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-m_idx-2, m_idx+n_idx+2):
        for i2 in range(-n_idx-2, m_idx+n_idx+2):
            pos = i1*a1 + i2*a2
            frac = B_inv @ pos
            f1, f2 = frac[0]%1.0, frac[1]%1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1,f2,0),
                material=mp.Medium(epsilon=8.9)))

mp.verbosity(0)
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=1.0), num_bands=n_bands,
    resolution=actual_res, k_points=[mp.Vector3(0,0,0)])

fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)
t0 = time.time()
ms.run_tm()
dt = time.time() - t0
os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
result = dict(m=m_idx, n=n_idx, ppc=ppc, nb=n_bands, theta=np.degrees(theta_rad),
              N_cells=m_idx**2+n_idx**2, L_super=float(L_SUPER),
              actual_res=actual_res, grid=actual_res, time=dt, rss_mb=rss_mb)
print(json.dumps(result))
'''

header = "%8s %6s %6s %6s %6s %6s %8s %8s %10s" % (
    "(m,n)", "theta", "N", "px/c", "bands", "grid", "time", "RSS_MB", "RSS_GB")
print(header)
print("-" * 80)

for m, n, ppc, nb in cases:
    code = SCRIPT.format(m=m, n=n, ppc=ppc, nb=nb)
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "meep", "python3", "-c", code],
            capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            L = (m**2 + n**2)**0.5
            grid = ppc * round(L)
            print("(%2d,%d) %5.2f  %6d %5d %5d %6d  FAILED (probably OOM)" % (
                m, n, 2*57.2958*n/m if m>0 else 0, m**2+n**2, ppc, nb, grid))
            continue
        # Parse last JSON line from output
        lines = result.stdout.strip().split('\n')
        d = None
        for line in reversed(lines):
            try:
                d = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if d is None:
            import math
            L = math.sqrt(m**2 + n**2)
            grid = ppc * round(L)
            print("(%2d,%d)    --  %6d %5d %5d %6d  PARSE ERROR" % (
                m, n, m**2+n**2, ppc, nb, grid))
            if result.stderr:
                print("  stderr:", result.stderr[:200])
            continue
        print("(%2d,%d) %5.2f° %6d %5d %5d %6d %7.2fs %7.0fMB %8.1fGB" % (
            d['m'], d['n'], d['theta'], d['N_cells'], d['ppc'], d['nb'],
            d['grid'], d['time'], d['rss_mb'], d['rss_mb']/1024))
    except subprocess.TimeoutExpired:
        import math
        L = math.sqrt(m**2 + n**2)
        grid = ppc * round(L)
        print("(%2d,%d)    --  %6d %5d %5d %6d  TIMEOUT (>5min)" % (
            m, n, m**2+n**2, ppc, nb, grid))

sys.stdout.flush()
