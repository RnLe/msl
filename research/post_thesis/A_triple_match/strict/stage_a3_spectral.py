#!/usr/bin/env python3
"""Stage A3 — SUPERSEDED in practice (kept for the record): the MINRES-ARPACK interior solve of
the Galerkin-limit operator is correct but too slow at 832k DOF (hours). The floor was instead
obtained exactly by dense diagonalization at small cells (stage_a3_dense.py; the eps-sampling
offset is a local rod property, ~+2.4e-5, transferable across m). This formulation remains the
reference definition of the engine's operator.
"""
"""ORIGINAL docstring: the operator-consistent floor: exact spectral solve of the Galerkin-limit operator.

The variational Galerkin's complete-basis limit is NOT the continuum: it is the operator with the
EXACT spectral kinetic |Q+G|^2 but eps_bl SAMPLED on the engine's grid (centered cell, Ngrid=912
= px16, Nsub=8). §13.1 flagged the ~1e-4 gap between that operator's ground and the continuum
Richardson floor (0.370907) as the residual uncertainty in every sub-1e-4 exactness statement.
This script solves that operator DIRECTLY (matrix-free FFT + shift-invert with an iterative inner
solver), giving the airtight floor the Stage-B ladder must converge to, and separating the
eps-sampling from the FD-stencil parts of the res16 -> continuum shift (8.6e-4).

Operator (TM, Bloch Q=X): with E = e^{iQ.r} u and v = sqrt(eps) u,
    L v = eps^{-1/2} . F^{-1} |Q+G|^2 F . eps^{-1/2} v = lambda v,   lambda=(2pi f)^2,
Hermitian PSD, matrix-free (two FFTs per apply). Interior eigenvalues near sigma=(2pi*0.367)^2
via eigsh shift-invert; the inner solves (L - sigma) y = x use MINRES with a Fourier-diagonal
preconditioner 1/(|Q+G|^2/eps_ref + sigma).

Usage: stage_a3_spectral.py [cell=centered|primitive] [px] [k] [sigma_w]
"""
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, minres

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

M = 57
Q = np.array([np.pi, 0.0])


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "centered"
    px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    sig_w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.367
    if cell == "centered":
        N = px * round((M * M + 1) ** 0.5)
    else:
        N = px * round(((M * M + 1) / 2) ** 0.5)
    eps, info = build_bilayer_eps_asym(M, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, cell)
    B = np.asarray(info["B_super"], float)
    print(f"cell={cell} px={px} N={N} ({N*N} DOF)  spectral-kinetic + grid-sampled eps", flush=True)

    b = 2 * np.pi * np.linalg.inv(B).T
    fr = np.fft.fftfreq(N) * N
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b[0, 0] + N2 * b[0, 1]
    Gy = N1 * b[1, 0] + N2 * b[1, 1]
    kin = (Q[0] + Gx) ** 2 + (Q[1] + Gy) ** 2               # exact |Q+G|^2
    ehalf = np.sqrt(eps)
    inv_ehalf = 1.0 / ehalf
    nd = N * N
    sig = (2 * np.pi * sig_w) ** 2

    def L_apply(x):
        v = (x.reshape(N, N)) * inv_ehalf
        w = np.fft.ifft2(kin * np.fft.fft2(v))
        return ((w * inv_ehalf).ravel())

    L = LinearOperator((nd, nd), matvec=L_apply, dtype=np.complex128)
    # Fourier-diagonal preconditioner for the inner MINRES (positive definite smoother)
    eref = float(np.mean(eps))
    pdiag = 1.0 / (kin / eref + sig)

    def M_apply(x):
        return np.fft.ifft2(pdiag * np.fft.fft2(x.reshape(N, N))).ravel()

    # scipy minres is real-only: use the real isomorphism (Hermitian complex -> symmetric real 2N)
    def to_r(z):
        return np.concatenate([z.real, z.imag])

    def to_c(v):
        return v[:nd] + 1j * v[nd:]

    def A_real(v):
        y = L_apply(to_c(v)) - sig * to_c(v)
        return to_r(y)

    def M_real(v):
        return to_r(M_apply(to_c(v)))

    Ar = LinearOperator((2 * nd, 2 * nd), matvec=A_real, dtype=np.float64)
    Mr = LinearOperator((2 * nd, 2 * nd), matvec=M_real, dtype=np.float64)
    stats = {"n": 0, "it": 0}

    def OPinv_apply(x):
        it = [0]

        def cb(_):
            it[0] += 1
        v, flag = minres(Ar, to_r(np.asarray(x, np.complex128)), M=Mr,
                         rtol=1e-9, maxiter=6000, callback=cb)
        stats["n"] += 1; stats["it"] += it[0]
        if flag != 0:
            print(f"    (minres flag={flag} after {it[0]} it)", flush=True)
        return to_c(v)

    OPinv = LinearOperator((nd, nd), matvec=OPinv_apply, dtype=np.complex128)
    t0 = time.time()
    vals, _ = eigsh(L, k=k, sigma=sig, which="LM", OPinv=OPinv, maxiter=1000, tol=1e-9)
    f = np.sort(np.sqrt(np.maximum(vals, 0)) / (2 * np.pi))
    dt = time.time() - t0
    print(f"  done in {dt:.0f}s ({stats['n']} inner solves, avg {stats['it']/max(stats['n'],1):.0f} MINRES its)")
    print("  spectral-operator ladder:", " ".join(f"{x:.6f}" for x in f))
    gq = f[:4]
    print(f"\n  OPERATOR-CONSISTENT FLOOR (quad mean) = {gq.mean():.6f}   (4-split {gq.max()-gq.min():.2e})")
    print(f"  vs continuum Richardson 0.370907: eps-sampling part  = {gq.mean()-0.370907:+.2e}")
    print(f"  vs res16 FD 0.370047:            FD-stencil part     = {0.370047-gq.mean():+.2e} (of -8.6e-4 total)")
    np.savez(os.path.join(HERE, f"stage_a3_spectral_{cell}_px{px}.npz"),
             freqs=f, cell=cell, px=px, sigma=sig_w)
    print("saved stage_a3_spectral npz")


if __name__ == "__main__":
    main()
