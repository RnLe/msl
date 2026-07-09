#!/usr/bin/env python3
"""Exact photonic moiré continuum model — Galerkin (Rayleigh–Ritz) projection
of the true supercell TM operator onto a reference-Bloch basis at the folded
moiré momenta.

TM generalized eigenproblem on the supercell:  −∇²E = λ ε_bl E, i.e.
    H c = λ S c,   H_αβ = ⟨∇E_α|∇E_β⟩,   S_αβ = ⟨E_α|ε_bl|E_β⟩,
with ε_bl the FULL moiré (supercell) dielectric — the same field FDFD uses.

Basis: reference Bloch E-fields E_{n,p}(r) = e^{i p·r} u_n(r; p; s̄), for
band n ∈ retained set and folded momenta p = X + (j₁/2) g₁ + (j₂/2) g₂
(|j_i| ≤ 2·GCUT+1), from MPB at a single reference registry s̄. All p fold to
the supercell momentum Q_X, so eigh(H,S) directly yields the Q_X spectrum =
the pooled 4-lane X-manifold, comparable to FDFD.

This is variational, converges to FDFD as (bands, GCUT)→∞, and contains the
Bloch-overlap form factors + inter-band coupling exactly (unlike the scalar
FFT-of-Λ heuristic in momentum_kp_moire.py).
"""
import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import map_coordinates

import meep as mp
from meep import mpb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = np.array([np.pi, 0.0])           # carrier k0 = monolayer X (cartesian)


def moire_g(m, n=1):
    theta = 2 * np.arctan2(n, m)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    L1 = np.array([float(m), float(n)]); L2 = np.array([-float(n), float(m)])
    B_super = np.column_stack([L1, L2])
    g = 2 * np.pi * np.linalg.inv(B_super).T   # cols g1, g2  (MOIRÉ recip)
    return g, B_super, theta


def extract_reference_bloch(sbar, momenta_cart, nbands, res=64, r1=0.20, r2=0.10):
    """u_n(r;p) periodic parts (E_z) at the reference registry for a list of
    cartesian momenta. Returns u [n_k, nbands, res, res] and eps_mono [res,res]."""
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                         basis1=mp.Vector3(1, 0, 0), basis2=mp.Vector3(0, 1, 0))
    geom = [mp.Cylinder(radius=r1, center=mp.Vector3(0, 0, 0),
                        material=mp.Medium(epsilon=8.9)),
            mp.Cylinder(radius=r2, center=mp.Vector3(sbar[0], sbar[1], 0),
                        material=mp.Medium(epsilon=8.9))]
    kpts = [mp.Vector3(p[0] / (2 * np.pi), p[1] / (2 * np.pi), 0)
            for p in momenta_cart]
    ms = mpb.ModeSolver(geometry=geom, geometry_lattice=lattice,
                        default_material=mp.Medium(epsilon=1.0),
                        num_bands=nbands, resolution=res, k_points=kpts, mesh_size=3)
    mp.verbosity(0)
    ms.run_tm()
    eps_mono = np.array(ms.get_epsilon())
    nk = len(kpts)
    u = np.zeros((nk, nbands, res, res), dtype=complex)
    freqs = np.array(ms.all_freqs)                 # (nk, nbands)
    # get_efield after run: need per-k; re-solve per k to fetch fields
    for ik, kp in enumerate(kpts):
        ms.solve_kpoint(kp)
        for b in range(nbands):
            e = np.array(ms.get_efield(b + 1, bloch_phase=False))[:, :, 0, 2]
            u[ik, b] = e
    return u, eps_mono, freqs


def _supercell_coords(B_super, Ngrid):
    s1 = np.arange(Ngrid) / Ngrid
    S1, S2 = np.meshgrid(s1, s1, indexing="ij")
    Xc = S1 * B_super[0, 0] + S2 * B_super[0, 1]
    Yc = S1 * B_super[1, 0] + S2 * B_super[1, 1]
    return Xc, Yc


def build_nudft_matrices(res, Xc, Yc):
    """Precompute the separable non-uniform DFT matrices for evaluating any
    monolayer-periodic field (given its res×res Fourier coeffs) at the supercell
    points. Grid-only — reused across all basis fields (the expensive part)."""
    gvals = np.fft.fftfreq(res) * res
    fx = (Xc % 1.0).ravel()
    fy = (Yc % 1.0).ravel()
    M1 = np.exp(2j * np.pi * np.outer(gvals, fx))         # (res, Npts)
    M2 = np.exp(2j * np.pi * np.outer(gvals, fy))         # (res, Npts)
    return M1, M2


def build_field_fourier(coeffs, p_cart, Xc, Yc, M1, M2):
    """E(r)=e^{i p·r} u(r), exact from monolayer Fourier coeffs (no interp
    error, only the res band-limit). Uses precomputed NUDFT matrices M1,M2."""
    res = coeffs.shape[0]
    inner = coeffs @ M2                                    # (res, Npts)
    u_at = np.sum(M1 * inner, axis=0) / (res * res)        # (Npts,)
    E = u_at.reshape(Xc.shape) * np.exp(1j * (p_cart[0] * Xc + p_cart[1] * Yc))
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=7)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--gcut", type=int, default=3)
    ap.add_argument("--nbands", type=int, default=2, help="retained bands 0..nbands-1")
    ap.add_argument("--band-lo", type=int, default=0)
    ap.add_argument("--r1", type=float, default=0.20, help="layer-1 rod radius")
    ap.add_argument("--r2", type=float, default=0.10, help="layer-2 rod radius (weak knob)")
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.23046875, 0.10546875])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--nk-window", type=float, default=None,
                    help="drop basis momenta whose ref band-lo freq > this (speed)")
    ap.add_argument("--s-tol", type=float, default=1e-6,
                    help="canonical-orthogonalization threshold (rel. to S max eig)")
    ap.add_argument("--out", default=os.path.join(HERE, "galerkin_out.npz"))
    args = ap.parse_args()

    g, B_super, theta = moire_g(args.m)
    Ngrid = args.px * round(float(np.sqrt(args.m**2 + 1)))
    # folded momenta: X + (j1/2)g1 + (j2/2)g2, |j|<=2*gcut+1
    J = 2 * args.gcut + 1
    js = np.arange(-J, J + 1)
    momenta, jpairs = [], []
    for j1 in js:
        for j2 in js:
            p = X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1]
            momenta.append(p); jpairs.append((j1, j2))
    momenta = np.array(momenta)

    bands = list(range(args.band_lo, args.band_lo + args.nbands))
    nb_solve = args.band_lo + args.nbands
    u, eps_mono, freqs = extract_reference_bloch(args.sbar, momenta, nb_solve, args.res,
                                                 r1=args.r1, r2=args.r2)

    # optionally prune high-energy basis momenta (ref band_lo freq)
    keep = np.ones(len(momenta), bool)
    if args.nk_window is not None:
        keep = freqs[:, args.band_lo] <= args.nk_window
    idx_keep = np.where(keep)[0]

    # supercell dielectric (FULL moiré) — same as FDFD
    eps_bl, info = build_bilayer_eps_asym(args.m, 1, args.r1, args.r2, 8.9, 8.9, 1.0,
                                          Ngrid, Ngrid, 8, "centered")

    # build basis fields (exact Fourier construction) directly into preallocated
    # complex64 stacks (memory-lean — the 912² real-space fields OOM'd WSL).
    Xc, Yc = _supercell_coords(B_super, Ngrid)
    M1, M2 = build_nudft_matrices(args.res, Xc, Yc)   # precompute once (grid-only)
    coeffs = np.fft.fft2(u, axes=(2, 3))          # monolayer Fourier coeffs
    basis = [(ik, b) for ik in idx_keep for b in bands]
    Nb = len(basis)
    npix = Ngrid * Ngrid
    phaseX = np.exp(-1j * (X[0] * Xc + X[1] * Yc))
    b_sup = 2 * np.pi * np.linalg.inv(B_super).T     # cols b1, b2
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]
    Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()   # |X+G|²  (N²,)
    dA = abs(np.linalg.det(B_super)) / npix
    norm = dA / npix                                 # Parseval
    Psi = np.empty((Nb, npix), np.complex64)          # E (Q_X-Bloch)
    Wh = np.empty((Nb, npix), np.complex64)           # FFT of periodic part w
    for a, (ik, b) in enumerate(basis):
        E = build_field_fourier(coeffs[ik, b], momenta[ik], Xc, Yc, M1, M2)
        Psi[a] = E.ravel().astype(np.complex64)
        Wh[a] = np.fft.fft2(E * phaseX).ravel().astype(np.complex64)
    epsv = eps_bl.ravel().astype(np.float32)
    kinf = kin.astype(np.float32)
    # chunked Gram matrices (cap peak memory): S=<E|ε|E>·dA, H=<ŵ||X+G|²|ŵ>·norm
    H = np.zeros((Nb, Nb), np.complex128)
    S = np.zeros((Nb, Nb), np.complex128)
    CH = 64
    for i0 in range(0, Nb, CH):
        i1 = min(i0 + CH, Nb)
        Sblk = (Psi[i0:i1].conj() * epsv[None, :]) @ Psi.T
        Hblk = (Wh[i0:i1].conj() * kinf[None, :]) @ Wh.T
        S[i0:i1] = Sblk.astype(np.complex128) * dA
        H[i0:i1] = Hblk.astype(np.complex128) * norm
    del Psi, Wh
    H = 0.5 * (H + H.conj().T); S = 0.5 * (S + S.conj().T)
    # Canonical orthogonalization: the reference-Bloch basis is non-orthogonal
    # and near-rank-deficient on the finite grid (redundant folded states).
    # Diagonalize S, keep the subspace with eigenvalue > tol*max, solve there.
    sval, svec = eigh(S)
    tol = args.s_tol * sval.max()
    keepS = sval > tol
    n_kept = int(keepS.sum())
    Vp = svec[:, keepS] / np.sqrt(sval[keepS])          # (Nb, n_kept)
    Hp = Vp.conj().T @ H @ Vp
    Hp = 0.5 * (Hp + Hp.conj().T)
    w = eigh(Hp, eigvals_only=True)
    fvals = np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))
    np.savez(args.out, freqs=fvals, m=args.m, px=args.px, gcut=args.gcut,
             nbands=args.nbands, band_lo=args.band_lo, n_basis=Nb,
             n_kept=n_kept, theta_deg=np.degrees(theta))
    print(f"m={args.m} px={args.px} gcut={args.gcut} bands={bands} "
          f"Nbasis={Nb} (kept {len(idx_keep)}/{len(momenta)} momenta) "
          f"-> S-rank {n_kept}/{Nb} (tol={args.s_tol:.0e})")
    print("Galerkin freqs (first 20):", " ".join(f"{x:.6f}" for x in fvals[:20]))


if __name__ == "__main__":
    main()
