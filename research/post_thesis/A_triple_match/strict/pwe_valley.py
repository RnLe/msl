#!/usr/bin/env python3
"""Stage B — valley-windowed PWE: the exact, well-conditioned solver for the moiré manifold.

INSIGHT (§15 + the failed clean fixed-frame ladder): at small twist the manifold's Fourier
support is COMPACT — every manifold state lives on momenta k = X + g_mono + G_env with g_mono a
few monolayer harmonics (the local Bloch character) and G_env a small envelope disk of primitive
reciprocals (the moiré envelope). So the EXACT solver is the plane-wave pencil restricted to that
tensor-product window:
    H = diag |k|^2   (exact spectral kinetic),
    S_{kk'} = eps_hat(k - k')   (analytic Toeplitz from ONE FFT of the sampled eps),
    H c = lambda S c,  variational -> the engine-operator floor (A3: continuum + 2.4e-5) from
    above as (R_env, g_mono window) grow. S is Gram(eps>0) => positive definite, and plane waves
    are orthogonal => NO conditioning wall. C2 (and T_P1, automatic in the primitive frame) act
    as index permutations for exact symmetry resolution.
This replaces the MPB-reference-Bloch basis (gauge headaches, near-dependence, §12 conditioning
wall) with the tensor-product structure the EA itself assumes — the completeness ladder of the EA.

Usage: pwe_valley.py --m 57 --px 16 --renv 3.0 --gmono 2 [--c2-split 1]
  renv: envelope radius in units of |b_prim|; gmono: monolayer-harmonic window |g|_inf <= gmono.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = np.array([np.pi, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--renv", type=float, default=3.0)
    ap.add_argument("--gmono", type=int, default=2)
    ap.add_argument("--renv-shells", type=str, default=None,
                    help="ADAPTIVE window: comma list renv per |g|_inf shell (overrides "
                         "--renv/--gmono), e.g. '15,15,15,8,4' = |g|<=2 at renv15, "
                         "|g|=3 at 8, |g|=4 at 4")
    ap.add_argument("--sector", choices=["even", "odd"], default="even",
                    help="T_P1 sector = envelope-lattice parity in centered units. At 2deg the "
                         "X-star-centered manifold lives on the ODD (offset) lattice: its "
                         "envelope momenta are X_star + odd-sum centered reciprocals (half a "
                         "b_prim cell off the star). 'even' = the integer b_prim lattice "
                         "(the q+ problem; §16.2's slow-converging tail window).")
    ap.add_argument("--c2-split", type=int, default=1)
    ap.add_argument("--window", type=float, nargs=2, default=[0.3661, 0.3785])
    ap.add_argument("--floor", type=float, default=0.370907)
    ap.add_argument("--r1", type=float, default=0.20)
    ap.add_argument("--r2", type=float, default=0.10)
    ap.add_argument("--eps2", type=float, default=8.9, help="layer-2 contrast knob (Stage C)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = args.m
    N = args.px * round(((m * m + 1) / 2) ** 0.5)
    eps, info = build_bilayer_eps_asym(m, 1, args.r1, args.r2, 8.9, args.eps2, 1.0,
                                       N, N, 8, "primitive")
    B = np.asarray(info["B_super"], float)
    bp = 2 * np.pi * np.linalg.inv(B).T          # primitive reciprocals
    Bc = np.array([[float(m), -1.0], [1.0, float(m)]])
    bc = 2 * np.pi * np.linalg.inv(Bc).T         # centered-cell reciprocals (the fine lattice)
    bnorm = np.linalg.norm(bp[:, 0])             # radii quoted in |b_prim| units
    par = 0 if args.sector == "even" else 1
    epshat = np.fft.fft2(eps) / (N * N)          # eps_hat on the primitive reciprocal grid
    t0 = time.time()

    # ---- build the momentum window: k = X + g_mono + G_env (all exact primitive indices)
    if args.renv_shells:
        shell_r = [float(x) for x in args.renv_shells.split(",")]
        gm_ = len(shell_r) - 1
    else:
        gm_ = args.gmono
        shell_r = [args.renv] * (gm_ + 1)
    monos = [(k1, k2) for k1 in range(-gm_, gm_ + 1) for k2 in range(-gm_, gm_ + 1)]

    def env_disk(r):
        # CENTERED-index offsets of the requested parity within radius r*|b_prim|
        Jm = int(np.ceil(r * bnorm / np.linalg.norm(bc[:, 0]))) + 2
        return [(o1, o2) for o1 in range(-Jm, Jm + 1) for o2 in range(-Jm, Jm + 1)
                if (o1 + o2) % 2 == par
                and np.linalg.norm(o1 * bc[:, 0] + o2 * bc[:, 1]) <= r * bnorm + 1e-12]

    disks = {r: env_disk(r) for r in set(shell_r)}
    idx_set = {}
    for (k1, k2) in monos:
        r = shell_r[max(abs(k1), abs(k2))]
        # monolayer g = 2pi(k1,k2) -> CENTERED index (m*k1 + k2, -k1 + m*k2) (even-sum)
        n1m = m * k1 + k2
        n2m = -k1 + m * k2
        for (o1, o2) in disks[r]:
            idx_set[(n1m + o1, n2m + o2)] = True
    nn = np.array(sorted(idx_set.keys()))
    Nb = len(nn)
    assert np.all((nn[:, 0] + nn[:, 1]) % 2 == par), "window parity broken"
    kvec = (X[None, :] + nn[:, 0:1] * bc[:, 0][None, :] + nn[:, 1:2] * bc[:, 1][None, :])
    kin = (kvec ** 2).sum(1)

    def eps_lookup(u, v):
        """eps_hat at k_u - k_v: centered index difference dn (even-sum) -> primitive FFT index
        np = ((dn1+dn2)/2, (dn2-dn1)/2) mod N."""
        dn1 = u[:, 0][:, None] - v[:, 0][None, :]
        dn2 = u[:, 1][:, None] - v[:, 1][None, :]
        p1 = ((dn1 + dn2) // 2) % N
        p2 = ((dn2 - dn1) // 2) % N
        return epshat[p1, p2]
    cfg = (f"shells={shell_r}" if args.renv_shells else f"renv={args.renv}|b| gmono={gm_}")
    print(f"m={m} px={args.px} N={N}: PW window {cfg} -> Nb={Nb}", flush=True)

    # ---- S = eps_hat Toeplitz (vectorized lookup), H = diag kin
    dA = abs(np.linalg.det(B))
    results = {}
    if args.c2_split:
        # C2: k -> -k: in centered indices n -> nC2 - n with nC2 = index(-2X) = (-m, 1)
        # (parity even => preserves the sector). Build the C2 blocks DIRECTLY from the
        # Toeplitz lookup (never forming the full Nb^2 S): for orbit reps a,b (a != C2a):
        #   <a±|S|b±> = eps_hat(k_a-k_b) ± eps_hat(k_a - C2 k_b)   (S Hermitian, C2-invariant),
        # fixed points (a = C2a) only join the + block with weight 1.
        nC2 = np.array([-m, 1])
        lookup = {tuple(v): i for i, v in enumerate(nn)}
        partner = np.array([lookup.get((int(nC2[0] - a), int(nC2[1] - b)), -1) for a, b in nn])
        keep = partner >= 0
        print(f"  C2 closure: {keep.sum()}/{Nb} have partners in-window "
              f"(window symmetrized by intersection)", flush=True)
        reps = np.array([i for i in range(Nb) if keep[i] and i <= partner[i]])
        fixed = partner[reps] == reps
        nA = nn[reps]                                    # rep indices
        nBp = nn[partner[reps]]                          # partner indices (C2 images)
        wgt = np.where(fixed, 1.0, 1.0 / np.sqrt(2))
        for tag, sgn in [("C2+", 1.0), ("C2-", -1.0)]:
            sel = np.ones(len(reps), bool) if sgn > 0 else ~fixed
            ra, rb = nA[sel], nBp[sel]
            wv = wgt[sel]
            fx = fixed[sel]
            m_ = len(ra)
            # exact P†SP, orbit-sum entrywise: B_ab = w_a w_b Σ_{α∈orb(a),β∈orb(b)}
            #   sgn^{[α flip]+[β flip]} eps_hat(k_α − k_β); fixed orbits have a single member.
            Sb = eps_lookup(ra, ra).astype(np.complex128)
            t = eps_lookup(ra, rb); t[:, fx] = 0.0
            Sb += sgn * t
            t = eps_lookup(rb, ra); t[fx, :] = 0.0
            Sb += sgn * t
            t = eps_lookup(rb, rb); t[fx, :] = 0.0; t[:, fx] = 0.0
            Sb += t
            Sb *= (wv[:, None] * wv[None, :]) * dA
            Sb = 0.5 * (Sb + Sb.conj().T)
            kina = (X[None, :] + ra[:, 0:1] * bc[:, 0][None, :]
                    + ra[:, 1:2] * bc[:, 1][None, :])
            kd = (kina ** 2).sum(1)                      # |k_a|^2 = |k_C2a|^2
            Hb = np.diag(kd) * dA
            w = eigh(Hb, Sb, eigvals_only=True)
            f = np.sort(np.sqrt(np.maximum(w, 0)) / (2 * np.pi))
            results[tag] = f
            lo, hi = args.window
            win = f[(f >= lo) & (f <= hi)]
            msg = f"  [{tag}] Nb={m_}  window: {win.size} states"
            if win.size:
                msg += f"  bottom {win[0]:.6f}  Δfloor {win[0]-args.floor:+.2e}"
            print(msg, flush=True)
            if win.size:
                print("    window ladder:", " ".join(f"{x:.6f}" for x in win[:8]), flush=True)
            del Sb, Hb
    else:
        S = eps_lookup(nn, nn) * dA
        S = 0.5 * (S + S.conj().T)
        H = np.diag(kin) * dA
        w = eigh(H, S, eigvals_only=True)
        f = np.sort(np.sqrt(np.maximum(w, 0)) / (2 * np.pi))
        results = {"all": f}
        lo, hi = args.window
        win = f[(f >= lo) & (f <= hi)]
        print(f"  window: {win.size} states" + (f", bottom {win[0]:.6f}" if win.size else ""))
    print(f"  ({time.time()-t0:.1f}s)")
    out = args.out or f"pwe_valley_m{m}_r{args.renv:g}_g{gm_}.npz"
    np.savez(os.path.join(HERE, out),
             **{f"f_{k.replace('+','p').replace('-','m')}": v for k, v in results.items()},
             m=m, renv=args.renv, gmono=gm_, Nb=Nb)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
