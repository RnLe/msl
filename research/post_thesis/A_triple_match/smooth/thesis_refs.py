#!/usr/bin/env python3
"""Reference legs for the thesis-crystal port: FDFD (inertia-certified, Richardson)
and MPB, both consuming the identical truncated-Fourier bilayer.

  fdfd --m M [--res 16,24,32]   window census by LDL inertia on the FDFD matrix,
                                per-state fitted-order extrapolation
  mpb  --m M [--res 24]         MPB on the supercell with an epsilon function
                                evaluating the same analytic coefficients
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
POST = "/home/renlephy/msl/research/post_thesis"
sys.path.insert(0, os.path.join(POST, "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, POST)
sys.path.insert(0, HERE)

import thesis_port as tp  # noqa: E402
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402

WIN = (5.15, 5.95)          # lambda window: the manifold tower above the gap top


def sector_q(m):
    A = lat.supercell_A(tp.LATTICE, m, 1)
    Bs = lf.supercell_basis(tp.LATTICE, m, 1)
    from fractions import Fraction
    ks = lat.fold_sector(A, (Fraction(1, 2), Fraction(0)))
    return Bs, lat.sector_to_cartesian(Bs, ks), ks


def fdfd_run(m, res_list):
    from sksparse.cholmod import cholesky
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    l1, l2 = tp.layers()
    N = lat.n_cells(lat.supercell_A(tp.LATTICE, m, 1))
    Bs, q_vec, ks = sector_q(m)
    sigma = 0.5 * (WIN[0] + WIN[1])
    print(f"=== fdfd (m,1)=({m},1)  N={N}  kappa_s=({ks[0]},{ks[1]}) ===",
          flush=True)
    ladders, grids = [], []
    for res in res_list:
        Nx = Ny = round(res * np.sqrt(N))
        t0 = time.time()
        s1 = np.arange(Nx) / Nx
        s2 = np.arange(Ny) / Ny
        S1, S2 = np.meshgrid(s1, s2, indexing="ij")
        Xg = Bs[0, 0] * S1 + Bs[0, 1] * S2
        Yg = Bs[1, 0] * S1 + Bs[1, 1] * S2
        e = lf.direct_eval(tp.LATTICE, m, 1, tp.EPS0, l1, l2, Xg, Yg)
        assert np.max(np.abs(e.imag)) < 1e-9
        e = e.real
        assert e.min() > 0.05, e.min()
        L = build_fdfd_operator(e, {"B_super": Bs}, q_vec,
                                polarization="tm").tocsc()
        eye = sp.eye(L.shape[0], format="csc")

        def below(lam):
            f = cholesky((L - lam * eye), beta=0, mode="simplicial")
            return int(np.sum(f.D() < 0.0))

        cen = below(WIN[1]) - below(WIN[0])
        fac = cholesky((L - sigma * eye), beta=0, mode="simplicial")
        OPinv = spla.LinearOperator(L.shape, matvec=fac, dtype=L.dtype)
        k = cen + 12
        for _ in range(5):
            w = np.sort(spla.eigsh(L, k=k, sigma=sigma, which="LM",
                                   OPinv=OPinv, tol=1e-10,
                                   return_eigenvectors=False))
            if float(np.max(np.abs(w - sigma))) >= 0.5 * (WIN[1] - WIN[0]):
                break
            k = int(1.6 * k) + 8
        lad = w[(w >= WIN[0]) & (w <= WIN[1])]
        print(f"  res{res}: {Nx}x{Ny} ({Nx*Ny:,}) census={cen} "
              f"extracted={len(lad)}  {time.time()-t0:.0f}s", flush=True)
        ladders.append(np.sort(lad))
        grids.append(Nx)
    L0 = min(len(x) for x in ladders)
    F = np.stack([x[:L0] for x in ladders])
    h = 1.0 / np.array(grids, float)
    if len(res_list) >= 3:
        import fdfd_leg as fl
        ext = np.zeros(L0)
        ps = np.zeros(L0)
        unc = np.zeros(L0)
        for j in range(L0):
            ext[j], ps[j], e12, e23 = fl.fit_state(h, F[:, j])
            unc[j] = abs(e12 - e23)
        print(f"  fitted order p: med {np.nanmedian(ps):.2f}", flush=True)
    else:
        ext = F[-1] + (F[-1] - F[-2]) / ((h[-2] / h[-1]) ** 2 - 1.0)
        ps = np.full(L0, np.nan)
        unc = np.abs(ext - F[-1])
    np.savez(os.path.join(HERE, f"thesis_fdfd_{m}.npz"),
             raw=F, grids=np.array(grids), extrap=ext, p=ps, unc=unc,
             win=np.array(WIN))
    print(f"saved thesis_fdfd_{m}.npz  ({L0} states)", flush=True)


def mpb_run(m, res):
    import meep as mp_
    from meep import mpb
    l1, l2 = tp.layers()
    N = lat.n_cells(lat.supercell_A(tp.LATTICE, m, 1))
    Bs, q_vec, ks = sector_q(m)
    side = float(np.sqrt(N))
    # epsilon from the identical analytic coefficients, evaluated pointwise
    from lib_v5.materials import _sym
    hs1 = list(_sym(l1).items())
    hs2 = list(_sym(l2).items())
    th = lat.twist_angle(tp.LATTICE, m, 1)
    inv1 = np.linalg.inv(tp.B0)
    inv2 = np.linalg.inv(lf.rotation(th) @ tp.B0)

    def eps_func(p):
        cx, cy = p.x, p.y                 # cartesian, lattice vectors = Bs columns
        val = tp.EPS0
        for (h1, h2), c in hs1:
            s1 = inv1[0, 0] * cx + inv1[0, 1] * cy
            s2 = inv1[1, 0] * cx + inv1[1, 1] * cy
            val += (c * np.exp(2j * np.pi * (h1 * s1 + h2 * s2))).real
        for (h1, h2), c in hs2:
            s1 = inv2[0, 0] * cx + inv2[0, 1] * cy
            s2 = inv2[1, 0] * cx + inv2[1, 1] * cy
            val += (c * np.exp(2j * np.pi * (h1 * s1 + h2 * s2))).real
        return mp_.Medium(epsilon=max(float(val), 0.05))

    a1 = mp_.Vector3(Bs[0, 0], Bs[1, 0]) * (1.0 / side)
    a2 = mp_.Vector3(Bs[0, 1], Bs[1, 1]) * (1.0 / side)
    lattice = mp_.Lattice(size=mp_.Vector3(side, side),
                          basis1=a1, basis2=a2)
    kpt = mp_.Vector3(float(ks[0]), float(ks[1]))
    # bands: everything below the window bottom plus the window
    nb = int(N * 1.15) + 60
    ms = mpb.ModeSolver(geometry_lattice=lattice, k_points=[kpt],
                        resolution=res, num_bands=nb,
                        default_material=eps_func)
    ms.mesh_size = 1                      # the material is smooth: no subpixel mesh
    t0 = time.time()
    ms.run_tm()
    fr = np.array(ms.all_freqs[0])   # lattice sized in monolayer units
    lam = (2 * np.pi * fr) ** 2
    lad = np.sort(lam[(lam >= WIN[0]) & (lam <= WIN[1])])
    print(f"mpb (m,1)=({m},1) res {res}: {nb} bands, {len(lad)} in window "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, f"thesis_mpb_{m}_r{res}.npz"), lam=lad,
             all_lam=lam, res=res)
    print(f"saved thesis_mpb_{m}_r{res}.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fdfd", "mpb"])
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--res", type=str, default="16,24,32")
    args = ap.parse_args()
    if args.cmd == "fdfd":
        fdfd_run(args.m, tuple(int(x) for x in args.res.split(",")))
    else:
        mpb_run(args.m, int(args.res.split(",")[0]))
