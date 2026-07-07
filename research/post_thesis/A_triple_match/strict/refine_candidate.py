#!/usr/bin/env python3
"""Refine the asymmetric candidate (eps=8.9, r1=0.20, r2=0.10):
finer s-grid (9x9 half) and k-grid (16x16 full BZ), res 48, 4 bands.
Outputs: exact common gap, per-s band-1 minimum and its k location
(is the manifold bottom in the X/X' k.p patches?), landscape values at X/X'.
"""
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

EPS, R1, R2 = 8.9, 0.20, 0.10
N_BANDS = 4
RES = 48
NK = 16
SVALS = [i / 16 for i in range(9)]     # 0 .. 0.5

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
ks = np.linspace(-0.5, 0.5, NK, endpoint=False)
kpts = [mp.Vector3(kx, ky, 0) for kx in ks for ky in ks]
iX = min(range(len(kpts)), key=lambda i: (kpts[i].x - 0.5) ** 2 + kpts[i].y ** 2)
iXp = min(range(len(kpts)), key=lambda i: kpts[i].x ** 2 + (kpts[i].y - 0.5) ** 2)

mp.verbosity(0)
t0 = time.time()
b0max, b1min = -1.0, 99.0
b1min_at = None
rows = []
for sx in SVALS:
    for sy in SVALS:
        if sy < sx:
            continue
        geometry = [
            mp.Cylinder(radius=R1, center=mp.Vector3(0, 0, 0),
                        material=mp.Medium(epsilon=EPS)),
            mp.Cylinder(radius=R2, center=mp.Vector3(sx, sy, 0),
                        material=mp.Medium(epsilon=EPS)),
        ]
        ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
                            default_material=mp.Medium(epsilon=1.0),
                            num_bands=N_BANDS, resolution=RES,
                            k_points=kpts, mesh_size=3)
        ms.run_tm()
        fr = np.array(ms.all_freqs)
        j = int(np.argmin(fr[:, 1]))
        b0max = max(b0max, float(fr[:, 0].max()))
        if fr[j, 1] < b1min:
            b1min = float(fr[j, 1])
            b1min_at = ((sx, sy), (kpts[j].x, kpts[j].y))
        rows.append(((sx, sy), float(fr[:, 1].min()),
                     (kpts[j].x, kpts[j].y),
                     float(fr[iX, 1]), float(fr[iXp, 1]),
                     float(fr[:, 2].min())))

print(f"common gap: [{b0max:.5f}, {b1min:.5f}]  width {b1min-b0max:.5f}")
print(f"global band-1 min at s={b1min_at[0]}, k={b1min_at[1]}")
b2min = min(r[5] for r in rows)
print(f"band-2 global min: {b2min:.5f}   (headroom above manifold bottom: "
      f"{b2min - b1min:.5f})")
print("\n  s          min_b1    argmin k         L1(X)     L1(X')")
for (s, m1, kmin, lX, lXp, _) in sorted(rows, key=lambda r: r[1])[:12]:
    print(f"  {str(s):10s} {m1:.5f}  ({kmin[0]:+.3f},{kmin[1]:+.3f})   "
          f"{lX:.5f}   {lXp:.5f}")
print(f"\nt={time.time()-t0:.0f}s")
