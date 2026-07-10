#!/usr/bin/env python3
"""Stage D (definitive metric) — EA-rung fidelities against the TRUE FDFD ground state.

Ritz-ladder comparisons of small EA subspaces are polluted by under-resolved band-0 tower states
(interlacing pushes them up into the spectral gap). The clean, pollution-immune dossier metric:
project the MEASURED FDFD ground state psi onto each EA subspace S_i and report
    fid_i = ||P_{S_i} psi_w||^2_eps / ||psi_w||^2_eps   (how much of the state the ansatz spans),
    E_i   = RQ(P_{S_i} psi_w)                            (the energy the ansatz assigns),
where psi_w is the true state's component in the exact PW window (its own coverage is reported
too — the §16.3 budget). Projections are in the eps-metric (the pencil's S). Rungs:
    M0 plain EA (frozen Bloch @ X, s̄) -> M1 + local dispersion -> M2 + registry frames -> window.

The window covers the X-valley (q+) stars only; the centered FDFD state also carries the X'
half, which lives in the C4-image window (other sector) — by C4 symmetry its fidelities are
identical, so all statements are per-valley ("plain EA recovers the valley half": structural).

Usage: pwe_ea_fidelity.py [--renv 12] [--gmono 3] [--nref 3] [--band 1]
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import eigh, solve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402
from pwe_ea_ladder import bloch_vector  # noqa: E402

X = np.array([np.pi, 0.0])
M = 57


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--renv", type=float, default=12.0)
    ap.add_argument("--gmono", type=int, default=3)
    ap.add_argument("--nref", type=int, default=3)
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--mono-px", type=int, default=16)
    ap.add_argument("--sector", choices=["even", "odd"], default="odd",
                    help="envelope-lattice parity (centered units); the FDFD X-dominant "
                         "manifold states live on the ODD (offset) lattice")
    args = ap.parse_args()

    # ---- window (primitive frame) --------------------------------------------------------
    N = args.px * round(((M * M + 1) / 2) ** 0.5)
    eps, info = build_bilayer_eps_asym(M, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "primitive")
    B = np.asarray(info["B_super"], float)
    bp = 2 * np.pi * np.linalg.inv(B).T
    Bc = np.array([[float(M), -1.0], [1.0, float(M)]])
    bc = 2 * np.pi * np.linalg.inv(Bc).T
    bnorm = np.linalg.norm(bp[:, 0])
    par = 0 if args.sector == "even" else 1
    epshat = np.fft.fft2(eps) / (N * N)
    dA = abs(np.linalg.det(B))
    gm_ = args.gmono
    gwin = [(k1, k2) for k1 in range(-gm_, gm_ + 1) for k2 in range(-gm_, gm_ + 1)]
    Jm = int(np.ceil(args.renv * bnorm / np.linalg.norm(bc[:, 0]))) + 2
    envs = [(o1, o2) for o1 in range(-Jm, Jm + 1) for o2 in range(-Jm, Jm + 1)
            if (o1 + o2) % 2 == par
            and np.linalg.norm(o1 * bc[:, 0] + o2 * bc[:, 1]) <= args.renv * bnorm + 1e-12]
    lookup, nn = {}, []
    for (k1, k2) in gwin:
        n1m = M * k1 + k2
        n2m = -k1 + M * k2
        for (o1, o2) in envs:
            key = (n1m + o1, n2m + o2)
            if key not in lookup:
                lookup[key] = len(nn); nn.append(key)
    nn = np.array(nn); Nb = len(nn)
    kvec = X[None, :] + nn[:, 0:1] * bc[:, 0][None, :] + nn[:, 1:2] * bc[:, 1][None, :]
    kin = (kvec ** 2).sum(1)
    dn1 = nn[:, 0][:, None] - nn[:, 0][None, :]
    dn2 = nn[:, 1][:, None] - nn[:, 1][None, :]
    S = epshat[((dn1 + dn2) // 2) % N, ((dn2 - dn1) // 2) % N] * dA
    S = 0.5 * (S + S.conj().T)
    print(f"window renv={args.renv} gmono={gm_}: Nb={Nb}", flush=True)

    # ---- true state's PW coefficients on the window ---------------------------------------
    d = np.load(os.path.join(HERE, "stage0b_characters.npz"))
    Nc = int(d["N"]); gf = d["ground_fields"]; epsc = d["eps"]
    s = np.arange(Nc) / Nc
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    xr = S1 * Bc[0, 0] + S2 * Bc[0, 1]; yr = S1 * Bc[1, 0] + S2 * Bc[1, 1]
    phQ = np.exp(-1j * (X[0] * xr + X[1] * yr))
    # centered FFT index of each window k = X + G_prim: n_c = Bc^T G_prim / 2pi (integer)
    nc1 = nn[:, 0] % Nc
    nc2 = nn[:, 1] % Nc
    psis = []
    for i in range(4):
        u = phQ * gf[i].reshape(Nc, Nc) / np.sqrt(epsc)
        uh = np.fft.fft2(u) / (Nc * Nc)
        psis.append(uh[nc1, nc2])
    lam_true = (2 * np.pi * 0.370047) ** 2

    # ---- EA subspaces ----------------------------------------------------------------------
    def col_index(env, g_):
        n1 = M * g_[0] + g_[1] + env[0]
        n2 = -g_[0] + M * g_[1] + env[1]
        return lookup[(n1, n2)]

    Ne = len(envs)
    subspaces = {}
    c0, f0 = bloch_vector(X, args.sbar, gwin, args.mono_px, args.band)
    T0 = np.zeros((Nb, Ne), complex)
    for ie, env in enumerate(envs):
        for ig, g_ in enumerate(gwin):
            T0[col_index(env, g_), ie] = c0[ig]
    subspaces["M0 plain EA (frozen Bloch)"] = T0
    T1 = np.zeros((Nb, Ne), complex)
    for ie, env in enumerate(envs):
        q = X + env[0] * bc[:, 0] + env[1] * bc[:, 1]
        cg, _ = bloch_vector(q, args.sbar, gwin, args.mono_px, args.band)
        for ig, g_ in enumerate(gwin):
            T1[col_index(env, g_), ie] = cg[ig]
    subspaces["M1 + local dispersion"] = T1
    K = args.nref
    sgrid = [(i / K, j / K) for i in range(K) for j in range(K)]
    T2 = np.zeros((Nb, Ne * len(sgrid)), complex)
    for ks, sk in enumerate(sgrid):
        for ie, env in enumerate(envs):
            q = X + env[0] * bc[:, 0] + env[1] * bc[:, 1]
            cg, _ = bloch_vector(q, sk, gwin, args.mono_px, args.band)
            for ig, g_ in enumerate(gwin):
                T2[col_index(env, g_), ks * Ne + ie] = cg[ig]
    subspaces[f"M2 + registry ({K}x{K})"] = T2

    # ---- the WINDOW-EXACT manifold ground (same operator family as the EA subspaces) --------
    # C2- block solve with eigenvectors (the manifold bottom lives there: 0.372078 at r12g3);
    # projecting the FD eigenvector instead inflates energies (FD-vs-spectral symbol mismatch
    # at high k), so the dossier's reference state is the pencil's own ground.
    Sfull = S

    def rq(c):
        num = np.real(np.vdot(c, kin * c)) * dA
        den = np.real(np.vdot(c, Sfull @ c))
        return num / den if den > 1e-300 else np.nan

    nC2 = np.array([-M, 1])
    partner = np.array([lookup.get((int(nC2[0] - a), int(nC2[1] - b)), -1) for a, b in nn])
    keepm = partner >= 0
    reps = np.array([i for i in range(Nb) if keepm[i] and i <= partner[i]])
    nonfix = partner[reps] != reps
    repsm = reps[nonfix]                       # C2- uses non-fixed orbits only
    pa = partner[repsm]
    Pminus = np.zeros((Nb, len(repsm)), complex)
    Pminus[repsm, np.arange(len(repsm))] = 1 / np.sqrt(2)
    Pminus[pa, np.arange(len(repsm))] = -1 / np.sqrt(2)
    Hb = Pminus.conj().T @ (kin[:, None] * Pminus) * dA
    Sb = Pminus.conj().T @ (Sfull @ Pminus)
    Sb = 0.5 * (Sb + Sb.conj().T); Hb = 0.5 * (Hb + Hb.conj().T)
    w, V = eigh(Hb, Sb)
    fb = np.sqrt(np.maximum(w, 0)) / (2 * np.pi)
    iwin = np.where(fb >= 0.3661)[0]
    ig0 = iwin[0]
    cexact = Pminus @ V[:, ig0]
    print(f"\nwindow-exact manifold ground (C2- block): f = {fb[ig0]:.6f}")
    nw2 = np.real(np.vdot(cexact, Sfull @ cexact))

    # FDFD-state coverage (fidelity only; energies referenced to the window-exact state)
    for i in range(2):
        cw = psis[i]
        ov = abs(np.vdot(cexact, Sfull @ cw)) ** 2 / (nw2 * np.real(np.vdot(cw, Sfull @ cw)))
        print(f"  |<window-exact | FDFD state {i}>|^2 (eps-metric, valley-half) = {ov:.4f}")

    print(f"\nEA-rung dossier vs the window-exact ground (f = {fb[ig0]:.6f}):")
    print(f"{'rung':<28} {'fidelity':>9} {'E assigned':>11} {'ΔE':>10}")
    for tag, T in subspaces.items():
        TS = T.conj().T @ (Sfull @ cexact)
        G = T.conj().T @ (Sfull @ T)
        G = 0.5 * (G + G.conj().T)
        sv, sV = eigh(G)
        keep = sv > 1e-9 * sv.max()
        coef = sV[:, keep] @ ((sV[:, keep].conj().T @ TS) / sv[keep])
        proj = T @ coef
        fid = np.real(np.vdot(proj, Sfull @ proj)) / nw2
        fE = np.sqrt(rq(proj)) / (2 * np.pi)
        print(f"  {tag:<27} {fid:>9.4f} {fE:>11.6f} {fE-fb[ig0]:>+10.2e}")
    print("\n(fidelity = eps-metric fraction of the window-exact ground the ansatz spans;")
    print(" E assigned = Rayleigh quotient of the projection. Same operator family throughout.)")


if __name__ == "__main__":
    main()
