#!/usr/bin/env python3
"""Reference local band-1 dispersion E_1(X + q; s_bar) for the momentum-space
moire model. Computes the ASYM local crystal (rod1 r=0.20 @ 0, rod2 r=0.10 @
s_bar) band structure via MPB at momenta X + (j1/2) g1 + (j2/2) g2 (a half-
moire-reciprocal grid covering the plane-wave cutoff), and caches it.

Registry-AVERAGED variant (--avg): averages E_1(X+q; s) over a coarse
registry grid to reduce the separable-approximation error.
"""
import os
import sys
import argparse

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import meep as mp
from meep import mpb

HERE = os.path.dirname(os.path.abspath(__file__))
X = np.array([np.pi, 0.0])
RES = 64
NBANDS = 4          # need band index 1 (0-based)

ap = argparse.ArgumentParser()
ap.add_argument("--avg", action="store_true", help="registry-average E_ref")
ap.add_argument("--sbar", type=float, nargs=2, default=[0.23046875, 0.10546875])
ap.add_argument("--jmax", type=int, default=14, help="half-g index range")
ap.add_argument("--gvec", default=os.path.join(HERE, "gvec.npy"))

ap.add_argument("--out", default=os.path.join(HERE, "eref_disp.npz"))
args = ap.parse_args()
JMAX = args.jmax
G = np.load(args.gvec)  # cols g1,g2

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
# reciprocal-lattice conversion: MPB k in reciprocal-lattice (fractional) units.
# For the square a=1 lattice, cartesian k = 2*pi*(kx_frac, ky_frac), so
# kfrac = kcart/(2 pi).
js = np.arange(-JMAX, JMAX + 1)
J1, J2 = np.meshgrid(js, js, indexing="ij")
q_cart = (0.5 * J1[..., None] * G[:, 0] + 0.5 * J2[..., None] * G[:, 1])  # (nj,nj,2)
k_cart = X[None, None, :] + q_cart
k_frac = k_cart / (2 * np.pi)
kpts = [mp.Vector3(float(k_frac[i, j, 0]), float(k_frac[i, j, 1]), 0)
        for i in range(len(js)) for j in range(len(js))]

def band1_at(sx, sy):
    geom = [mp.Cylinder(radius=0.20, center=mp.Vector3(0, 0, 0),
                        material=mp.Medium(epsilon=8.9)),
            mp.Cylinder(radius=0.10, center=mp.Vector3(sx, sy, 0),
                        material=mp.Medium(epsilon=8.9))]
    ms = mpb.ModeSolver(geometry=geom, geometry_lattice=lattice,
                        default_material=mp.Medium(epsilon=1.0),
                        num_bands=NBANDS, resolution=RES, k_points=kpts, mesh_size=3)
    mp.verbosity(0)
    ms.run_tm()
    fr = np.array(ms.all_freqs)               # (nk, nbands), freqs (c/a)
    lam = (2 * np.pi * fr[:, 1]) ** 2         # band index 1 -> lambda
    return lam.reshape(len(js), len(js))

if args.avg:
    SG = [i / 6 for i in range(6)]
    acc = np.zeros((len(js), len(js))); cnt = 0
    for sx in SG:
        for sy in SG:
            acc += band1_at(sx, sy); cnt += 1
            print(f"  avg registry ({sx:.3f},{sy:.3f}) done", flush=True)
    Eref = acc / cnt
    sbar = None
else:
    sbar = args.sbar
    Eref = band1_at(sbar[0], sbar[1])

np.savez(args.out, Eref=Eref, js=js, G=G, X=X, sbar=np.array(sbar) if sbar else np.array([np.nan, np.nan]),
         res=RES, jmax=JMAX, averaged=args.avg)
print(f"saved {args.out}  Eref shape {Eref.shape}  "
      f"range [{Eref.min():.4f},{Eref.max():.4f}]  "
      f"E_ref(X)={Eref[JMAX, JMAX]:.5f}")
