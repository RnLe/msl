#!/usr/bin/env python3
"""Independent FDFD reference ladders for the smooth hexM validation family.

For each commensurate angle (m, n), solve the TM supercell eigenproblem at the
folded M2 Bloch sector with the EXACT analytic dielectric sampled pointwise on
the FDFD fractional grid (lib_v5.lifted.direct_eval; no rods, no smoothing).
Three resolutions (res = 16, 24, 32 pts per sqrt(cell)), per-state order fit
f(h) = f0 + C h^p, extrapolated ladder + pairwise-extrapolant uncertainty.
"""
import os
import sys
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
POST_THESIS = os.path.abspath(os.path.join(HERE, "..", ".."))
THESIS_RESULTS = os.path.abspath(os.path.join(
    POST_THESIS, "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, THESIS_RESULTS)
sys.path.insert(0, POST_THESIS)
sys.path.insert(0, HERE)

from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted  # noqa: E402
from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
import candidate_hexM as cand  # noqa: E402

try:
    from sksparse.cholmod import cholesky
    HAVE_CHOLMOD = True
except ImportError:
    from scipy.sparse.linalg import splu
    HAVE_CHOLMOD = False

ANGLES = [(4, 3), (5, 4), (6, 5), (7, 6), (9, 8)]
RES_LIST = (16, 24, 32)
WIN_EXTRACT = (1.630, 1.670)           # lambda = (2 pi f)^2
SIGMA = 0.5 * (WIN_EXTRACT[0] + WIN_EXTRACT[1])
R_COVER = 0.035                        # required eigsh completeness radius
BAND1_FLOOR_CUT = 1.5                  # below-window spectral gap anchor
OUT = os.path.join(HERE, "fdfd_leg_ladders.npz")


def eps_on_grid(m, n, Nx, Ny, Bs, l1, l2):
    """Exact analytic dielectric on the supercell fractional grid (pointwise)."""
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    X = Bs[0, 0] * S1 + Bs[0, 1] * S2
    Y = Bs[1, 0] * S1 + Bs[1, 1] * S2
    e = lifted.direct_eval("hex", m, n, cand.EPS0, l1, l2, X, Y)
    imax = float(np.max(np.abs(e.imag)))
    assert imax < 1e-11, f"imag part {imax:.2e} not ~0"
    e = e.real
    assert e.min() > 0.5, e.min()
    return e, imax


def make_opinv(L_op):
    n = L_op.shape[0]
    shifted = (L_op - SIGMA * sp.eye(n, format="csc")).tocsc()
    if HAVE_CHOLMOD:
        factor = cholesky(shifted, beta=0, mode="simplicial")
        return LinearOperator((n, n), matvec=factor, dtype=L_op.dtype)
    lu = splu(shifted)
    return LinearOperator((n, n), matvec=lu.solve, dtype=L_op.dtype)


def solve_res(eps, Bs, q_vec, k0):
    """Shift-invert eigsh around SIGMA; grow k until the completeness radius
    covers the extraction window plus margin. Returns sorted eigenvalues."""
    L_op = build_fdfd_operator(eps, {"B_super": Bs}, q_vec, polarization="tm")
    op_inv = make_opinv(L_op)
    n = L_op.shape[0]
    k = min(k0, n - 2)
    for _ in range(4):
        vals = np.sort(eigsh(L_op, k=k, sigma=SIGMA, which="LM", OPinv=op_inv,
                             maxiter=20000, tol=1e-10,
                             return_eigenvectors=False))
        r_max = float(np.max(np.abs(vals - SIGMA)))
        if r_max >= R_COVER or k >= n - 2:
            return vals, r_max, k
        k = min(int(1.7 * k) + 4, n - 2)
        print(f"    coverage {r_max:.4f} < {R_COVER}; retry k={k}", flush=True)
    raise RuntimeError("eigsh coverage not reached")


def fit_state(h, f):
    """Fit f(h) = f0 + C h^p through 3 points; also the two pairwise h^2
    Richardson extrapolants. Returns (f0, p, E12, E23)."""
    h = np.asarray(h, float)
    f = np.asarray(f, float)
    E12 = f[1] + (f[1] - f[0]) / ((h[0] / h[1]) ** 2 - 1.0)
    E23 = f[2] + (f[2] - f[1]) / ((h[1] / h[2]) ** 2 - 1.0)
    d1, d2 = f[0] - f[1], f[1] - f[2]
    p = np.nan
    f0 = E23
    if d1 * d2 > 0:
        r = d1 / d2

        def g(p_):
            return (h[0] ** p_ - h[1] ** p_) / (h[1] ** p_ - h[2] ** p_) - r

        try:
            if g(0.2) * g(8.0) < 0:
                p = brentq(g, 0.2, 8.0, xtol=1e-12)
                f0 = f[2] - (f[1] - f[2]) * h[2] ** p / (h[1] ** p - h[2] ** p)
        except Exception:
            p = np.nan
    return f0, p, E12, E23


def run_angle(m, n, l1, l2, store):
    A = lat.supercell_A("hex", m, n)
    ncell = lat.n_cells(A)
    Bs = lifted.supercell_basis("hex", m, n)
    kappa_s = lat.fold_sector(A, (Fraction(0), Fraction(1, 2)))
    q_vec = lat.sector_to_cartesian(Bs, kappa_s)
    tag = f"{m}_{n}"
    print(f"\n=== (m,n)=({m},{n})  N_cells={ncell}  "
          f"kappa_s=({kappa_s[0]},{kappa_s[1]})  q=({q_vec[0]:+.6f},"
          f"{q_vec[1]:+.6f}) ===", flush=True)
    anomalies = []
    band1, grids, rcovs, imaxs = [], [], [], []
    k0 = max(40, ncell // 3 + 20)
    for res in RES_LIST:
        Nx = Ny = round(res * np.sqrt(ncell))
        t0 = time.time()
        eps, imax = eps_on_grid(m, n, Nx, Ny, Bs, l1, l2)
        vals, r_max, k_used = solve_res(eps, Bs, q_vec, k0)
        b1 = vals[(vals > BAND1_FLOOR_CUT) & (vals <= SIGMA + R_COVER)]
        band1.append(b1)
        grids.append(Nx)
        rcovs.append(r_max)
        imaxs.append(imax)
        nw = int(np.sum((b1 >= WIN_EXTRACT[0]) & (b1 <= WIN_EXTRACT[1])))
        print(f"  res{res}: grid {Nx}x{Ny} ({Nx*Ny:,} DOF)  k={k_used} "
              f"cover={r_max:.4f}  band1(<=+{R_COVER})={len(b1)}  "
              f"in-window={nw}  imag_max={imax:.1e}  "
              f"t={time.time()-t0:.0f}s", flush=True)
    counts = [int(np.sum((b >= WIN_EXTRACT[0]) & (b <= WIN_EXTRACT[1])))
              for b in band1]
    if len(set(counts)) != 1:
        anomalies.append(f"({m},{n}) in-window count changes across res: "
                         f"{dict(zip(RES_LIST, counts))} (aligned from the "
                         f"band-1 floor through the below-window gap)")
    # Align by sorted index anchored at the band-1 floor (real spectral gap
    # below the valley), robust to states drifting across the window edges.
    L = min(len(b) for b in band1)
    F = np.stack([b[:L] for b in band1])           # (3, L)
    jump = np.max(np.abs(F[2] - F[1]))
    if jump > 0.01:
        anomalies.append(f"({m},{n}) res24->32 shift up to {jump:.4f} "
                         f"(alignment suspect)")
    h = 1.0 / np.array(grids, float)
    f0s, ps, e12s, e23s = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
    for j in range(L):
        f0s[j], ps[j], e12s[j], e23s[j] = fit_state(h, F[:, j])
    unc = np.abs(e12s - e23s)
    keep = (f0s >= WIN_EXTRACT[0]) & (f0s <= WIN_EXTRACT[1])
    nbad = int(np.sum(~np.isfinite(ps[keep])))
    if nbad:
        anomalies.append(f"({m},{n}) {nbad}/{int(keep.sum())} window states "
                         f"with failed order fit (non-monotone differences); "
                         f"h^2 pair extrapolant E23 used there")
    store.update({
        f"{tag}_raw": F, f"{tag}_grids": np.array(grids),
        f"{tag}_extrap_all": f0s, f"{tag}_unc_all": unc,
        f"{tag}_p_all": ps, f"{tag}_E12_all": e12s, f"{tag}_E23_all": e23s,
        f"{tag}_keep": keep, f"{tag}_counts": np.array(counts),
        f"{tag}_extrap": f0s[keep], f"{tag}_unc": unc[keep],
        f"{tag}_p": ps[keep],
        f"{tag}_kappa_s": np.array([float(kappa_s[0]), float(kappa_s[1])]),
        f"{tag}_q_vec": q_vec, f"{tag}_ncells": ncell,
        f"{tag}_cover": np.array(rcovs), f"{tag}_imag_max": np.array(imaxs),
    })
    return f0s[keep], unc[keep], ps[keep], counts, anomalies


def main():
    l1, l2 = cand.layers()
    store = {
        "angles": np.array(ANGLES), "res_list": np.array(RES_LIST),
        "win_extract": np.array(WIN_EXTRACT), "sigma": SIGMA,
        "window": np.array(cand.WINDOW), "carrier_frac": np.array([0.0, 0.5]),
    }
    results = {}
    all_anoms = []
    for m, n in ANGLES:
        lam, unc, ps, counts, anoms = run_angle(m, n, l1, l2, store)
        results[(m, n)] = (lam, unc, ps, counts)
        all_anoms.extend(anoms)
    np.savez(OUT, **store)
    print(f"\nsaved {OUT}")
    print("\n" + "=" * 72)
    print("FDFD extrapolated window ladders  (lambda = (2 pi f)^2, "
          f"extract [{WIN_EXTRACT[0]}, {WIN_EXTRACT[1]}])")
    print("=" * 72)
    for (m, n), (lam, unc, ps, counts) in results.items():
        cnt = "/".join(str(c) for c in counts)
        print(f"(m,n)=({m},{n})  n_win={len(lam)}  "
              f"raw in-window counts res16/24/32: {cnt}")
        for j in range(len(lam)):
            pstr = f"{ps[j]:.2f}" if np.isfinite(ps[j]) else " nan"
            print(f"   {j:3d}  lam={lam[j]:.6f}  unc={unc[j]:.2e}  p={pstr}")
    if all_anoms:
        print("\nANOMALIES:")
        for a in all_anoms:
            print("  ! " + a)
    else:
        print("\nno anomalies (counts stable, fits clean)")


if __name__ == "__main__":
    main()




