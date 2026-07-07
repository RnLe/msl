#!/usr/bin/env python3
"""Candidate search: does the two-rod local crystal have a COMMON gap
across ALL registries s?

The EA's discrete envelope spectrum corresponds to individual FDFD levels
only if the target manifold is spectrally isolated: the frequency window
must lie inside the local band gap of EVERY registry (otherwise some moire
region propagates and the envelope states dissolve into that continuum —
proven at (eps=8.9, r=0.2) by the X-weight classification).

For each (eps_rod, r): for s on a grid over the registry torus (two-rod
square crystal, rods at (0,0) and s), solve the lowest bands on a full-BZ
k-grid with MPB; record per-s band extrema; report common gap intervals
I_n = [max_s max_k w_n, min_s min_k w_{n+1}].
"""
import os
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

N_BANDS = 5
RES = 32
NK = 8            # full-BZ k-grid NK x NK
# s-grid over [0, 1/2]^2 with sx <= sy (mirror + inversion redundancy)
SVALS = [i / 8 for i in range(5)]           # 0, .125, .25, .375, .5

PARAMS = [
    (8.9, 0.20),   # thesis case — expect NO common 0-1 gap (control)
    (8.9, 0.15), (8.9, 0.25), (8.9, 0.30),
    (11.4, 0.15), (11.4, 0.20), (11.4, 0.25), (11.4, 0.30),
    (13.0, 0.20), (13.0, 0.25),
]

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
kpts = [mp.Vector3(kx, ky, 0)
        for kx in np.linspace(0, 0.5, NK)
        for ky in np.linspace(0, 0.5, NK)]      # full BZ by C4v of the k-sum?
# NOTE: at generic s there is no point symmetry, but bands satisfy
# w(k) = w(-k) (real eps, TR), so [0,1/2]^2 covers half the BZ; the missing
# half is (kx,-ky) — include it:
kpts += [mp.Vector3(kx, -ky, 0)
         for kx in np.linspace(0, 0.5, NK)
         for ky in np.linspace(0.5 / (NK - 1), 0.5, NK - 1)]

mp.verbosity(0)

print(f"# k-points: {len(kpts)}, bands: {N_BANDS}, res: {RES}")
print("eps   r     s-worst    max_b0    min_b1   common01 |  max_b1    min_b2   common12")

for eps_rod, r in PARAMS:
    t0 = time.time()
    per_s = []
    for sx in SVALS:
        for sy in SVALS:
            if sy < sx:
                continue
            geometry = [
                mp.Cylinder(radius=r, center=mp.Vector3(0, 0, 0),
                            material=mp.Medium(epsilon=eps_rod)),
                mp.Cylinder(radius=r, center=mp.Vector3(sx, sy, 0),
                            material=mp.Medium(epsilon=eps_rod)),
            ]
            ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
                                default_material=mp.Medium(epsilon=1.0),
                                num_bands=N_BANDS, resolution=RES,
                                k_points=kpts, mesh_size=3)
            ms.run_tm()
            fr = np.array(ms.all_freqs)          # (nk, nbands)
            per_s.append(((sx, sy),
                          [float(fr[:, b].max()) for b in range(N_BANDS)],
                          [float(fr[:, b].min()) for b in range(N_BANDS)]))
    for lo_band in (0, 1):
        hi_band = lo_band + 1
        w0max_all = max(p[1][lo_band] for p in per_s)
        w1min_all = min(p[2][hi_band] for p in per_s)
        worst = max(per_s, key=lambda p: p[1][lo_band] - 0)  # s of highest b0
        if lo_band == 0:
            line = (f"{eps_rod:4.1f}  {r:.2f}  {worst[0]}  "
                    f"{w0max_all:.4f}   {w1min_all:.4f}   ")
            line += (f"GAP [{w0max_all:.4f},{w1min_all:.4f}] "
                     f"width {w1min_all - w0max_all:.4f}"
                     if w1min_all > w0max_all else "none")
        else:
            line += (f" | {w0max_all:.4f}   {w1min_all:.4f}   ")
            line += (f"GAP [{w0max_all:.4f},{w1min_all:.4f}] "
                     f"width {w1min_all - w0max_all:.4f}"
                     if w1min_all > w0max_all else "none")
    print(line + f"   ({time.time()-t0:.0f}s)", flush=True)
