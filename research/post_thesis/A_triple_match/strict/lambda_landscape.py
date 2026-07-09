#!/usr/bin/env python3
"""Λ₁(X;s) landscape via a pure-MPB registry sweep — the moiré-potential source
for the momentum-space model's Ṽ = FFT_s[Λ₁].

Blaze phase-1 (lib/phase1_blaze_v4.py) also produces this (as eigenvalues[:,1]
of its 16384-point reg128 sweep) but is heavy (9.4 GB) and occasionally aborts
with heap corruption at reg128. For the weak-coupling sweep the model only needs
eigenvalues[:,1], so we compute exactly that with MPB — robust, tiny, and
method-consistent with momentum_kp_ref.py (same 2-rod local crystal, same X
carrier). Output matches build_vtilde()'s expectation: key `eigenvalues`, shape
(nreg², nbands), band index 1 = band-1-at-X λ=(2πf)².

Registry axis convention MUST match blaze phase-1 for the anisotropic model to
pair Ṽ with E_ref correctly (mass_xx≠mass_yy; carrier X=(π,0) breaks x↔y): blaze
orders the sweep with sx as the FAST index, so ev[:,1].reshape(K,K) is Λ[sy,sx].
We replicate that (outer loop = sy, inner = sx). With this convention the MPB Λ
reproduces the blaze reg128 momentum spectrum to <1e-5 (validated at r₂=0.10:
span 1.113, offset +2.74e-3, detrend 8.16e-4 — identical to the documented
anchor). The half-cell (i+0.5)/K centering blaze uses is irrelevant (it only
adds a spectrum-invariant linear phase to Ṽ). Row-checkpointed: a crash resumes
from the last completed registry row.
"""
import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import meep as mp
from meep import mpb

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", type=float, default=0.20)
    ap.add_argument("--r2", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=8.9, help="layer-1 rod epsilon")
    ap.add_argument("--eps2", type=float, default=None,
                    help="layer-2 rod epsilon (weak-contrast knob); default = --eps")
    ap.add_argument("--nreg", type=int, default=32, help="K for K×K registry grid")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--nbands", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "lambda_landscape.npz"))
    args = ap.parse_args()

    eps2 = args.eps if args.eps2 is None else args.eps2
    K = args.nreg
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                         basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
    kX = [mp.Vector3(0.5, 0.0, 0.0)]      # X carrier (fractional)
    mp.verbosity(0)

    ck = args.out.replace(".npz", "_ck.npz")
    ev = np.full((K * K, args.nbands), np.nan)
    row0 = 0
    if os.path.isfile(ck):
        d = np.load(ck)
        if int(d["K"]) == K and int(d["res"]) == args.res \
                and float(d["r2"]) == args.r2:
            ev = d["ev"]; row0 = int(d["row_done"])
            print(f"  resume from row {row0}/{K}", flush=True)

    # outer loop = sy, inner = sx (sx fast) -> reshape(K,K) is Λ[sy,sx], the
    # blaze convention required for correct Ṽ↔E_ref pairing under anisotropy.
    for iy in range(row0, K):
        for ix in range(K):
            sx, sy = ix / K, iy / K
            geom = [mp.Cylinder(radius=args.r1, center=mp.Vector3(0, 0, 0),
                                material=mp.Medium(epsilon=args.eps)),
                    mp.Cylinder(radius=args.r2, center=mp.Vector3(sx, sy, 0),
                                material=mp.Medium(epsilon=eps2))]
            ms = mpb.ModeSolver(geometry=geom, geometry_lattice=lattice,
                                default_material=mp.Medium(epsilon=1.0),
                                num_bands=args.nbands, resolution=args.res,
                                k_points=kX, mesh_size=3)
            ms.run_tm()
            fr = np.array(ms.all_freqs)[0]         # (nbands,) freqs at X
            ev[iy * K + ix] = (2 * np.pi * fr) ** 2  # -> lambda
        np.savez(ck, ev=ev, row_done=iy + 1, K=K, res=args.res,
                 r1=args.r1, r2=args.r2)
        if (iy + 1) % 8 == 0 or iy == K - 1:
            print(f"  row {iy+1}/{K}", flush=True)

    reg = np.array([[ix / K, iy / K] for iy in range(K) for ix in range(K)])
    lam1 = ev[:, 1]
    np.savez(args.out, eigenvalues=ev, registry=reg, K=K, res=args.res,
             r1=args.r1, r2=args.r2, eps=args.eps, eps2=eps2)
    if os.path.isfile(ck):
        os.remove(ck)
    f1 = np.sqrt(lam1) / (2 * np.pi)
    print(f"saved {args.out}  K={K} res={args.res} r2={args.r2}", flush=True)
    print(f"  Λ₁ range [{lam1.min():.4f},{lam1.max():.4f}] λ  "
          f"dLam={lam1.max()-lam1.min():.4f}  "
          f"f-range [{f1.min():.5f},{f1.max():.5f}]", flush=True)


if __name__ == "__main__":
    main()
