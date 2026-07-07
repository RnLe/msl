#!/usr/bin/env python3
"""Candidate search round 2: ASYMMETRIC bilayer (layer 2 = weak perturber).

Layer 1: square rods r1, eps. Layer 2: rods r2 < r1 at registry s.
For small r2 the local crystal is layer-1's gapped crystal + weak
modulation -> the TM gap stays open for ALL s (continuity) = registry-common
gap, while the registry dependence Lambda_1(s) provides the moire potential.
Report the common gap AND the modulation depth of the band-1-at-X landscape
(the EA target): Lambda_1(X; s) range over s.
"""
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

N_BANDS = 3
RES = 32
NK = 8
SVALS = [i / 8 for i in range(5)]

PARAMS = [  # (eps, r1, r2)
    (8.9, 0.20, 0.20),   # control (known: no common gap)
    (8.9, 0.20, 0.14),
    (8.9, 0.20, 0.10),
    (8.9, 0.20, 0.07),
    (8.9, 0.20, 0.05),
    (11.4, 0.20, 0.10),
    (11.4, 0.20, 0.07),
]

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
kfull = [mp.Vector3(kx, ky, 0)
         for kx in np.linspace(0, 0.5, NK)
         for ky in np.linspace(0, 0.5, NK)]
kfull += [mp.Vector3(kx, -ky, 0)
          for kx in np.linspace(0, 0.5, NK)
          for ky in np.linspace(0.5 / (NK - 1), 0.5, NK - 1)]
kX = [mp.Vector3(0.5, 0, 0)]

mp.verbosity(0)
print(f"# k: {len(kfull)}, bands: {N_BANDS}, res {RES}, s-grid {len(SVALS)}^2/2")
print("eps   r1    r2  |  max_b0    min_b1   common gap        |  L1(X;s) range        depth")

for eps_rod, r1, r2 in PARAMS:
    t0 = time.time()
    b0max, b1min = -1.0, 99.0
    l1x = []
    for sx in SVALS:
        for sy in SVALS:
            if sy < sx:
                continue
            geometry = [
                mp.Cylinder(radius=r1, center=mp.Vector3(0, 0, 0),
                            material=mp.Medium(epsilon=eps_rod)),
                mp.Cylinder(radius=r2, center=mp.Vector3(sx, sy, 0),
                            material=mp.Medium(epsilon=eps_rod)),
            ]
            ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
                                default_material=mp.Medium(epsilon=1.0),
                                num_bands=N_BANDS, resolution=RES,
                                k_points=kfull, mesh_size=3)
            ms.run_tm()
            fr = np.array(ms.all_freqs)
            b0max = max(b0max, float(fr[:, 0].max()))
            b1min = min(b1min, float(fr[:, 1].min()))
            ms2 = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
                                 default_material=mp.Medium(epsilon=1.0),
                                 num_bands=N_BANDS, resolution=RES,
                                 k_points=kX, mesh_size=3)
            ms2.run_tm()
            l1x.append(float(np.array(ms2.all_freqs)[0, 1]))
    gap = (f"GAP [{b0max:.4f},{b1min:.4f}] w={b1min-b0max:+.4f}"
           if b1min > b0max else f"none (overlap {b0max-b1min:.4f})")
    print(f"{eps_rod:4.1f}  {r1:.2f}  {r2:.2f} |  {b0max:.4f}   {b1min:.4f}   "
          f"{gap:34s} |  [{min(l1x):.4f},{max(l1x):.4f}]  {max(l1x)-min(l1x):.4f}"
          f"   ({time.time()-t0:.0f}s)", flush=True)
