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

# --- V/E_kin calibration (fixed geometry m=57, theta=2 deg) -----------------
# E_kin = <direct_metric> * (2*pi*eta)^2,  eta = 2 sin(theta/2).  The band-1
# curvature <direct_metric>=0.585 comes from the r2=0.10 phase-1 extraction and
# is ~const in r2 at fixed angle (set by the strong layer). This reproduces the
# documented anchor: dLam=2.438 lam / E_kin=0.0284 lam -> V/E_kin=85.8 ~ 86.
M_MOIRE = 57
DIRECT_METRIC = 0.585
THETA = 2.0 * np.arctan2(1.0, M_MOIRE)
ETA = 2.0 * np.sin(THETA / 2.0)
E_KIN = DIRECT_METRIC * (2 * np.pi * ETA) ** 2

PARAMS = [  # (eps, r1, r2)
    (8.9, 0.20, 0.20),   # control (known: no common gap)
    (8.9, 0.20, 0.14),
    (8.9, 0.20, 0.10),   # strong anchor (V/E_kin ~ 86)
    (8.9, 0.20, 0.07),   # ~ V/E_kin 42
    (8.9, 0.20, 0.054),  # ~ V/E_kin 25
    (8.9, 0.20, 0.05),
    (8.9, 0.20, 0.045),
    (8.9, 0.20, 0.04),   # ~ V/E_kin 14 (finalist target)
    (8.9, 0.20, 0.035),
    (8.9, 0.20, 0.031),  # ~ V/E_kin 8  (weakest finalist target)
    (8.9, 0.20, 0.03),
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
print(f"# V/E_kin calib: m={M_MOIRE} theta={np.degrees(THETA):.4f}deg eta={ETA:.5f} "
      f"E_kin={E_KIN:.5f} lam (direct_metric={DIRECT_METRIC})")
print("eps   r1    r2   |  max_b0   min_b1   common gap       | dLam(lam) V/E_kin  gamma  beta"
      "  | L1(X;s) f-range")

records = []
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
    l1x = np.array(l1x)
    # moire depth: dLam in lambda=(2 pi f)^2 units (V/E_kin is defined in lambda)
    lam1 = (2 * np.pi * l1x) ** 2
    dLam_lambda = float(lam1.max() - lam1.min())
    dLam_f = float(l1x.max() - l1x.min())
    v_over_ekin = dLam_lambda / E_KIN
    isolated = b1min > b0max
    width = b1min - b0max
    midgap = 0.5 * (b1min + b0max)
    gamma = width / midgap if (isolated and midgap > 0) else float('nan')
    beta = THETA / gamma if gamma == gamma else float('nan')  # theta_rad/gamma
    gap = (f"GAP [{b0max:.4f},{b1min:.4f}] w={width:+.4f}"
           if isolated else f"none (overlap {b0max-b1min:.4f})")
    print(f"{eps_rod:4.1f}  {r1:.2f}  {r2:.3f} |  {b0max:.4f}  {b1min:.4f}  "
          f"{gap:32s} | {dLam_lambda:7.4f}  {v_over_ekin:6.1f}  "
          f"{gamma:.4f} {beta:.4f} | [{l1x.min():.4f},{l1x.max():.4f}]"
          f"   ({time.time()-t0:.0f}s)", flush=True)
    records.append(dict(eps=eps_rod, r1=r1, r2=r2, b0max=b0max, b1min=b1min,
                        isolated=isolated, gap_width=width, midgap=midgap,
                        gamma=gamma, beta=beta, dLam_lambda=dLam_lambda,
                        dLam_f=dLam_f, v_over_ekin=v_over_ekin,
                        l1x_min=float(l1x.min()), l1x_max=float(l1x.max())))

# persist for the crossover-law aggregator + candidate selection
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "scan_common_gap2.npz")
np.savez(out, records=np.array(records, dtype=object), m_moire=M_MOIRE,
         theta_rad=THETA, eta=ETA, e_kin=E_KIN, direct_metric=DIRECT_METRIC,
         res=RES, nk=NK, svals=np.array(SVALS))
print(f"\nsaved {out}  ({len(records)} candidates)", flush=True)
