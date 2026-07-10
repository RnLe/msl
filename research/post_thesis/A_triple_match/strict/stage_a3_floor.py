#!/usr/bin/env python3
"""Stage A3 (revised) — SUPERSEDED / WRONG-OBJECT WARNING (kept for the record, §15.8).

The construction below (FD stencil-Richardson at trig-upsampled fixed eps) does NOT define the
Galerkin engine's operator: trigonometric interpolation of the sampled eps introduces Gibbs
over/undershoot (clipped ad hoc at eps>=1 here), and the resulting "floor" (0.3651 at 2 deg) is
an artifact of that different operator. The engine's true complete-basis operator is
diag(eps^-1/2) F^-1 |Q+G|^2 F diag(eps^-1/2) with the SAMPLED eps — solved exactly (dense) in
stage_a3_dense.py, giving an eps-sampling offset of only ~+2.4e-5 vs the continuum. Use that.
"""
"""ORIGINAL (superseded) docstring: the operator-consistent floor by stencil-Richardson at FIXED eps.

GOAL. The Galerkin engine's complete-basis limit is the operator with exact spectral kinetic and
the px16-SAMPLED eps (the primitive-cell N0=640 sample set). Its ground = "the operator-consistent
floor" every exactness claim must target. Direct interior spectral solves are slow; instead:

  solve the FD operator at FIXED eps (the N0 samples, trigonometrically upsampled to N=2N0, 3N0)
  and Richardson the STENCIL error away (O(h^2), h=1/N at fixed eps). The limit is the ground of
  the continuum-kinetic + (trig-interpolant of the px16 eps) operator = the spectral engine's
  floor (the trig interpolant is exactly the eps the 640^2 plane-wave engine implies).

BONUS decomposition at N=1280: f_FD(eps640->1280) vs f_FD(eps1280) isolates the EPS-SAMPLING part
of the discretization error from the FD-STENCIL part at matched stencil.

Usage: stage_a3_floor.py [m] [px0] [upfactors...]
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, HERE)
from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

Q = np.array([np.pi, 0.0])


def trig_upsample(a, f):
    """Trigonometric (Fourier) upsampling of a periodic real field by integer factor f."""
    N = a.shape[0]
    A = np.fft.fft2(a)
    M = N * f
    B = np.zeros((M, M), complex)
    h = N // 2
    # place the N x N spectrum into the M x M grid (split Nyquist row/col evenly)
    ix = np.r_[0:h, M - h:M]
    B[np.ix_(ix, ix)] = A[np.ix_(np.r_[0:h, N - h:N], np.r_[0:h, N - h:N])]
    out = np.fft.ifft2(B).real * (f * f)
    return out


def ground_quad(eps, info, sig_w=0.367, k=8):
    L = build_fdfd_operator(eps, info, Q, "tm")
    sig = (2 * np.pi * sig_w) ** 2
    t0 = time.time()
    fac = cholesky((L - sig * sp.eye(L.shape[0], format="csc")).tocsc(), beta=0, mode="simplicial")
    vals = eigsh(L, k=k, sigma=sig, which="LM",
                 OPinv=LinearOperator(L.shape, matvec=fac, dtype=L.dtype),
                 maxiter=20000, tol=1e-10, return_eigenvectors=False)
    f = np.sort(np.sqrt(np.maximum(vals, 0)) / (2 * np.pi))
    print(f"    ({time.time()-t0:.0f}s) lowest: " + " ".join(f"{x:.6f}" for x in f[:4]))
    return float(f[:4].mean())


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    px0 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    ups = [int(x) for x in sys.argv[3:]] or [2]
    N0 = px0 * round(((m * m + 1) / 2) ** 0.5)
    eps0, info0 = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N0, N0, 8, "primitive")
    print(f"m={m} primitive px{px0}: N0={N0}. FIXED-eps stencil ladder:", flush=True)
    print(f"  N={N0} (native):", flush=True)
    g0 = ground_quad(eps0, info0)
    gs = [(N0, g0)]
    for f_ in ups:
        N = N0 * f_
        epsU = trig_upsample(eps0, f_)
        # clip tiny Gibbs negatives (trig interp of discontinuous-ish eps); eps must stay >= 1
        epsU = np.maximum(epsU, 1.0)
        infoU = dict(info0); infoU["Nx"] = N; infoU["Ny"] = N
        print(f"  N={N} (eps640 trig-upsampled x{f_}):", flush=True)
        g = ground_quad(epsU, infoU)
        gs.append((N, g))
        # decomposition at this N: native eps at the same stencil
        epsN, infoN = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "primitive")
        print(f"  N={N} (native px{px0*f_} eps):", flush=True)
        gn = ground_quad(epsN, infoN)
        print(f"    -> eps-sampling effect at matched stencil: {g - gn:+.2e}")
    # stencil Richardson at fixed eps (O(h^2))
    (N1, g1), (N2, g2) = gs[0], gs[1]
    h1, h2 = 1.0 / N1 ** 2, 1.0 / N2 ** 2
    ginf = g2 + (g2 - g1) * h2 / (h1 - h2)
    print(f"\n  OPERATOR-CONSISTENT FLOOR (fixed px{px0} eps, stencil->0): {ginf:.6f}")
    print(f"  vs continuum Richardson 0.370907: eps-sampling part = {ginf-0.370907:+.2e}")
    print(f"  vs native res16 FD 0.370047 (centered) / {g0:.6f} (primitive px16)")
    np.savez(os.path.join(HERE, f"stage_a3_floor_m{m}.npz"),
             ladder=np.array(gs), floor=ginf)
    print("saved stage_a3_floor npz")


if __name__ == "__main__":
    main()
