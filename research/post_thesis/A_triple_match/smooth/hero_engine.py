#!/usr/bin/env python3
"""Hero engine: per-commensuration comparison of the raw envelope operator against the
certified full reference on the frozen smooth candidate (candidate_hexM).

One hermitized-collocation discretization family throughout (the Phase-F lesson):
  reference  C = M[rho_sc] diag|k+G|^2 M[rho_sc]  on the supercell PW window,
  EA side    product space (slow torus grid x monolayer PWs), frozen band-1 frame at
             the M2 carrier (a dispersion extremum: no drift dive), lazily projected —
             the full product operator is never materialized.

Per angle: reference manifold ladder (dense eigh below the size cap, else bottom-block
Lanczos with certified residuals), EA ladder from H_P, count check in the isolated
window, sorted per-state deviations. Convergence knobs (gmax_mono, Ns, gmax_sc, fine)
are certified by bump runs.
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402
from lib_v5 import oracles as oc  # noqa: E402
from lib_v5 import raw_projection as rp  # noqa: E402

import candidate_hexM as cand  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LATTICE = cand.LATTICE
DENSE_CAP = 26000        # supercell PW count above which the reference goes iterative


def supercell_reference(m, n, gmax_sc, fine, n_window=40, layers=None):
    """Certified reference ladder near the manifold window at the folded M2 sector."""
    A = lat.supercell_A(LATTICE, m, n)
    Bs = lf.supercell_basis(LATTICE, m, n)
    l1, l2 = layers if layers is not None else cand.layers()
    c_sc = lf.moire_coeffs(LATTICE, m, n, cand.EPS0, l1, l2)
    ks = lat.fold_sector(A, cand.CARRIER_FRAC)
    k_sc = lat.sector_to_cartesian(Bs, ks)
    ns = mp.pw_set(gmax_sc)
    npw = len(ns)
    Brec = 2 * np.pi * np.linalg.inv(Bs).T
    kG = np.array([k_sc + Brec @ np.array(h, float) for h in ns])
    kin = (kG ** 2).sum(1)
    e = mat.sample(c_sc, fine, fine)
    assert e.min() > 0.3, e.min()
    rho = e ** -0.5
    rho_fft = np.fft.fft2(rho) / fine ** 2

    if npw <= DENSE_CAP:
        R = rp._mult_matrix(rho_fft, ns)
        C = R @ (kin[:, None] * R)
        C = 0.5 * (C + C.conj().T)
        w = sla.eigvalsh(C)
        cert = 0.0
        method = "dense"
    else:
        h = np.asarray(ns)
        buf = np.zeros((fine, fine), complex)

        def rho_apply(x):
            buf[:] = 0.0
            buf[h[:, 0] % fine, h[:, 1] % fine] = x
            out = np.fft.fft2(rho * np.fft.ifft2(buf))
            return out[h[:, 0] % fine, h[:, 1] % fine]

        def C_apply(x):
            return rho_apply(kin * rho_apply(x))

        Cop = spla.LinearOperator((npw, npw), matvec=C_apply, dtype=complex)
        ncell = lat.n_cells(A)
        k = min(npw - 2, int(1.6 * ncell) + n_window)
        w, V = spla.eigsh(Cop, k=k, which="SA", tol=1e-11)
        idx = np.argsort(w)
        w, V = w[idx], V[:, idx]
        res = [float(np.linalg.norm(C_apply(V[:, j]) - w[j] * V[:, j]))
               for j in range(len(w) - n_window, len(w))]
        cert = max(res)
        method = f"lanczos(k={k})"
    return dict(w=np.sort(w), kappa_s=ks, method=method, cert=cert,
                npw=npw, A=A, Bs=Bs)


def ea_ladder(m, n, gmax_mono, Ns, fine):
    """Raw-projection EA ladder: frozen band-1 frame at M2, lazily projected."""
    A = lat.supercell_A(LATTICE, m, n)
    A2 = lf.layer2_integer_matrix(LATTICE, m, n)
    W = np.asarray(A2, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(LATTICE, m, n)

    def registry_of_R(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    U, _ = rp.frozen_frame(LATTICE, cand.coeffs, (0.0, 0.0), cand.CARRIER_FRAC,
                           gmax_mono, [cand.BAND], fine=fine)
    H_P = lazy_project(cand.coeffs, cand.CARRIER_FRAC, gmax_mono, Ns,
                       registry_of_R, np.linalg.inv(Bs).T, U, fine)
    herm = np.linalg.norm(H_P - H_P.conj().T) / np.linalg.norm(H_P)
    assert herm < 1e-10, herm
    return np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))


def adapted_frames(coeffs_fn, kappa0, gmax, deltas, band_ids, fine):
    """Registry-adapted frames: eigenvectors of the local hermitized pencil at each
    registry point, phase-aligned to the reference-registry frame (single-band U(1)
    gauge fixed by a real-positive overlap; alignment magnitude asserted > 0.5 so a
    frame flip cannot pass silently)."""
    B0 = lat.monolayer_basis(LATTICE)
    Brec = 2 * np.pi * np.linalg.inv(B0).T
    k0 = Brec @ np.array([float(kappa0[0]), float(kappa0[1])])
    h0, _, _, _ = rp.mono_hermitized(coeffs_fn((0.0, 0.0)), k0, B0, gmax, fine)
    w0, V0 = sla.eigh(h0)
    Uref = V0[:, band_ids]
    frames = []
    for d in deltas:
        h, _, _, _ = rp.mono_hermitized(coeffs_fn(d), k0, B0, gmax, fine)
        w, V = sla.eigh(h)
        U = V[:, band_ids]
        for j in range(U.shape[1]):
            ov = np.vdot(Uref[:, j], U[:, j])
            assert abs(ov) > 0.5, (d, j, abs(ov))
            U[:, j] *= np.conj(ov) / abs(ov)
        frames.append(U)
    return frames


def lazy_project(coeffs_fn, kappa0, gmax, Ns, registry_of_R, slow_to_cart, U, fine):
    """H_P = T^dag L T without materializing L: apply the composed operator terms to
    the trial columns as (slow, pw) tensors. U may be a single frame (frozen model)
    or a list of per-slow-point frames (registry-adapted model — the slow spectral
    derivative then differentiates THROUGH the frame, so every frame-derivative
    contribution enters mechanically). Columns are exactly orthonormal either way
    (disjoint slow support x orthonormal frames)."""
    B0 = lat.monolayer_basis(LATTICE)
    ns = mp.pw_set(gmax)
    npw = len(ns)
    Brec = 2 * np.pi * np.linalg.inv(B0).T
    k0 = Brec @ np.array([float(kappa0[0]), float(kappa0[1])])
    kG = np.array([k0 + Brec @ np.array(h, float) for h in ns])
    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    rho_mats = rp.rho_tables(
        coeffs_fn,
        [registry_of_R(S1.reshape(-1)[j], S2.reshape(-1)[j])
         for j in range(Ns * Ns)],
        gmax, fine=fine)
    rho_arr = np.array(rho_mats).reshape(Ns, Ns, npw, npw)
    per_point = isinstance(U, (list, tuple))
    nb = (U[0] if per_point else U).shape[1]

    # trial tensor X[s1, s2, pw, col], col = (slow position delta, band): the slow
    # factor is represented in the position basis, which keeps the assembly local
    ncol = Ns * Ns * nb
    X = np.zeros((Ns, Ns, npw, ncol), complex)
    col = 0
    for j1 in range(Ns):
        for j2 in range(Ns):
            Uj = U[j1 * Ns + j2] if per_point else U
            for b in range(nb):
                X[j1, j2, :, col] = Uj[:, b]
                col += 1

    ik1 = 2j * np.pi * np.fft.fftfreq(Ns) * Ns
    ik2 = ik1.copy()

    def dslow(T, axis):
        F = np.fft.fft(T, axis=axis)
        shape = [1, 1, 1, 1]
        shape[axis] = Ns
        F *= (ik1 if axis == 0 else ik2).reshape(shape)
        return np.fft.ifft(F, axis=axis)

    def mul_rho(T):
        return np.einsum("abij,abjc->abic", rho_arr, T)

    def dfast(T, i):
        return T * (1j * kG[:, i])[None, None, :, None]

    def DR(T, i):
        return (slow_to_cart[i, 0] * dslow(T, 0)
                + slow_to_cart[i, 1] * dslow(T, 1))

    def L_apply(T):
        Y = mul_rho(T)
        out = np.zeros_like(T)
        for i in (0, 1):
            Zi = dfast(Y, i) + DR(Y, i)
            Zi = mul_rho(dfast(Zi, i) + DR(Zi, i))
            out -= Zi
        return out

    LX = L_apply(X)
    Xf = X.reshape(-1, ncol)
    H_P = Xf.conj().T @ LX.reshape(-1, ncol)
    return H_P


def run_angle(m, n, gmax_mono, Ns, gmax_sc, fine, window_pad=0.004):
    t0 = time.time()
    ref = supercell_reference(m, n, gmax_sc, fine)
    ea = ea_ladder(m, n, gmax_mono, Ns, fine)
    lo, hi = cand.WINDOW[0] - window_pad, cand.WINDOW[1] + window_pad
    wr = ref["w"][(ref["w"] >= lo) & (ref["w"] <= hi)]
    we = ea[(ea >= lo) & (ea <= hi)]
    out = dict(m=m, n=n, ref=wr, ea=we, n_ref=len(wr), n_ea=len(we),
               method=ref["method"], cert=ref["cert"], secs=time.time() - t0)
    if len(wr) == len(we) and len(wr):
        out["dev"] = np.abs(we - wr)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gmax-mono", type=int, default=4)
    ap.add_argument("--ns", type=int, default=17)
    ap.add_argument("--gmax-sc", type=int, default=0, help="0 = auto")
    ap.add_argument("--fine", type=int, default=192)
    args = ap.parse_args()
    if args.gmax_sc == 0:
        A = np.abs(np.asarray(lat.supercell_A(LATTICE, args.m, args.n), float))
        args.gmax_sc = int(args.gmax_mono
                           * max(A[0, 0] + A[1, 0], A[0, 1] + A[1, 1])
                           + args.ns // 2 + 6)
    r = run_angle(args.m, args.n, args.gmax_mono, args.ns, args.gmax_sc, args.fine)
    print(f"(m,n)=({r['m']},{r['n']}) ref[{r['method']}, cert {r['cert']:.1e}, "
          f"npw {args.gmax_sc}] counts ref/ea {r['n_ref']}/{r['n_ea']} "
          f"({r['secs']:.0f}s)")
    print("  ref:", np.array2string(r["ref"], precision=6, separator=","))
    print("  ea :", np.array2string(r["ea"], precision=6, separator=","))
    if "dev" in r:
        print("  dev:", " ".join(f"{d:.3e}" for d in r["dev"]))
