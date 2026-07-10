#!/usr/bin/env python3
"""Stage D — the EA trust dossier: every EA variant as a subspace restriction of the exact pencil.

The valley-windowed PWE (§16) is the exact solver on the window k = X + g_mono + G_env. Every
envelope-approximation variant is a RESTRICTION of that pencil to a structured subspace, so the
whole EA hierarchy can be evaluated in ONE framework with NO MPB gauge ambiguity:

  M0 plain EA (single frame, frozen Bloch): per envelope G, the trial vector locks the monolayer
     factor to the X-point Bloch vector of the s̄-frame local crystal:
         v_G = Σ_g û₁(g; X, s̄) |X+G+g⟩            (dim = N_env)
  M1 + exact local dispersion: the Bloch factor follows the envelope momentum:
         v_G = Σ_g û₁(g; X+G, s̄) |X+G+g⟩          (dim = N_env)
  M2 + registry adaptation: multi-frame Bloch factors:
         v_{G,k} = Σ_g û₁(g; X+G, s_k) |X+G+g⟩     (dim = N_env × K²)
  M3 exact: the full window (identity restriction).

û₁ comes from the MONOLAYER pencil at registry s (H=|q+g|² diag, S=ε̂_mono Toeplitz on the same
g-window; ε_mono rasterized with the identical subpixel algorithm as the supercell) — the same
discretization family as the supercell pencil. All models solved as H'=T†HT, S'=T†ST.

Because everything lives at ONE valley momentum q₊ (primitive frame), each model recovers at most
the q₊ (valley) HALF of the centered-cell spectrum — the '1/2, not 1/4' statement is structural;
the dossier measures the ENERGY accuracy of that half at each rung.

Usage: pwe_ea_ladder.py --m 57 --px 16 --renv 12 --gmono 3 --nref 3 [--band 1]
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


def eps_mono_grid(s, r1, r2, eps1, eps2, eps_bg, Npx, Nsub=8):
    """Monolayer (local-crystal) eps at registry s on an Npx x Npx unit-cell grid,
    with the identical subpixel-averaging algorithm as supercell_asym."""
    grid = (np.arange(Npx) + 0.0) / Npx
    Xg, Yg = np.meshgrid(grid, grid, indexing="ij")
    off = (np.arange(Nsub) + 0.5) / Nsub - 0.5
    acc = np.zeros((Npx, Npx))
    for oi in off:
        for oj in off:
            xs = Xg + oi / Npx
            ys = Yg + oj / Npx
            e = np.full((Npx, Npx), eps_bg)
            for (cx, cy, r, er) in [(0.0, 0.0, r1, eps1), (s[0], s[1], r2, eps2)]:
                dx = xs - cx - np.round(xs - cx)
                dy = ys - cy - np.round(ys - cy)
                e[(dx * dx + dy * dy) < r * r] = er
            acc += e
    return acc / (Nsub * Nsub)


def bloch_vector(q, s, gwin, Npx, band, r1=0.20, r2=0.10):
    """û_band(g; q, s): monolayer PW-pencil eigenvector on the g-window (unit S-norm)."""
    em = eps_mono_grid(s, r1, r2, 8.9, 8.9, 1.0, Npx)
    ehat = np.fft.fft2(em) / (Npx * Npx)
    G = np.array(gwin)
    kin = ((q[0] + 2 * np.pi * G[:, 0]) ** 2 + (q[1] + 2 * np.pi * G[:, 1]) ** 2)
    d1 = (G[:, 0][:, None] - G[:, 0][None, :]) % Npx
    d2 = (G[:, 1][:, None] - G[:, 1][None, :]) % Npx
    S = ehat[d1, d2]
    S = 0.5 * (S + S.conj().T)
    H = np.diag(kin)
    w, v = eigh(H, S)
    c = v[:, band]
    c = c / np.sqrt(np.real(c.conj() @ S @ c))
    return c, np.sqrt(max(w[band], 0)) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--renv", type=float, default=12.0)
    ap.add_argument("--gmono", type=int, default=3)
    ap.add_argument("--nref", type=int, default=3)
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--window", type=float, nargs=2, default=[0.3661, 0.3785])
    ap.add_argument("--floor", type=float, default=0.370907)
    ap.add_argument("--mono-px", type=int, default=16)
    ap.add_argument("--models", type=str, default="M0,M1,M2,M3",
                    help="which rungs to run (M3 = full window, redundant with pwe_valley)")
    args = ap.parse_args()
    models = set(args.models.split(","))
    m = args.m
    N = args.px * round(((m * m + 1) / 2) ** 0.5)
    eps, info = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "primitive")
    B = np.asarray(info["B_super"], float)
    bp = 2 * np.pi * np.linalg.inv(B).T
    bnorm = np.linalg.norm(bp[:, 0])
    epshat = np.fft.fft2(eps) / (N * N)
    dA = abs(np.linalg.det(B))
    gm_ = args.gmono
    gwin = [(k1, k2) for k1 in range(-gm_, gm_ + 1) for k2 in range(-gm_, gm_ + 1)]
    Jm = int(np.ceil(args.renv)) + 1
    envs = [(j1, j2) for j1 in range(-Jm, Jm + 1) for j2 in range(-Jm, Jm + 1)
            if np.linalg.norm(j1 * bp[:, 0] + j2 * bp[:, 1]) <= args.renv * bnorm + 1e-12]
    # window index list: order = env-major, mono-minor (so T maps are block-structured)
    lookup = {}
    nn = []
    for (k1, k2) in gwin:
        n1m = int(round(B[0, 0] * k1 + B[1, 0] * k2))
        n2m = int(round(B[0, 1] * k1 + B[1, 1] * k2))
        for (j1, j2) in envs:
            key = (n1m + j1, n2m + j2)
            if key not in lookup:
                lookup[key] = len(nn)
                nn.append(key)
    nn = np.array(nn)
    Nb = len(nn)
    kvec = X[None, :] + nn[:, 0:1] * bp[:, 0][None, :] + nn[:, 1:2] * bp[:, 1][None, :]
    kin = (kvec ** 2).sum(1)
    d1 = (nn[:, 0][:, None] - nn[:, 0][None, :]) % N
    d2 = (nn[:, 1][:, None] - nn[:, 1][None, :]) % N
    S = epshat[d1, d2] * dA
    S = 0.5 * (S + S.conj().T)
    Ne, Ng = len(envs), len(gwin)
    print(f"m={m} window: renv={args.renv} gmono={gm_} -> Nb={Nb} (envs {Ne} x monos {Ng})",
          flush=True)

    def col_index(env, gmono_):
        n1 = int(round(B[0, 0] * gmono_[0] + B[1, 0] * gmono_[1])) + env[0]
        n2 = int(round(B[0, 1] * gmono_[0] + B[1, 1] * gmono_[1])) + env[1]
        return lookup[(n1, n2)]

    def solve_T(T, tag, s_tol=1e-7):
        Hp = T.conj().T @ (kin[:, None] * T) * dA
        Sp = T.conj().T @ (S @ T)
        Sp = 0.5 * (Sp + Sp.conj().T); Hp = 0.5 * (Hp + Hp.conj().T)
        # canonical orthogonalization; multi-frame T is heavily near-dependent, and (§12
        # lesson, reproduced here) an over-tight s_tol admits near-null vectors that emit
        # SPURIOUS sub-floor states (the model span is a subspace of the exact window, so
        # true values must sit above the window's own spectrum). VARIATIONAL GUARD below.
        sv, sV = eigh(Sp)
        keep = sv > s_tol * sv.max()
        Vp = sV[:, keep] / np.sqrt(sv[keep])
        w = eigh(Vp.conj().T @ Hp @ Vp, eigvals_only=True)
        f = np.sort(np.sqrt(np.maximum(w, 0)) / (2 * np.pi))
        lo, hi = args.window
        win = f[(f >= lo) & (f <= hi)]
        nsub = int((f < args.floor - 5e-4).sum() - (f < 0.30).sum())  # sub-floor near-window states
        msg = f"  [{tag}] dim={T.shape[1]} rank={int(keep.sum())} window: {win.size} states"
        if win.size:
            msg += f"  bottom {win[0]:.6f}  Δfloor {win[0]-args.floor:+.2e}"
        if nsub > 0:
            msg += f"  [GUARD: {nsub} states in (0.30, floor-5e-4) — check spuriousness]"
        print(msg, flush=True)
        if win.size:
            print("    ladder:", " ".join(f"{x:.6f}" for x in win[:8]), flush=True)
        # nearest states below the window (diagnosis of under-converged manifolds)
        below = f[(f < lo) & (f > 0.30)]
        if below.size:
            print("    below-window tail:", " ".join(f"{x:.6f}" for x in below[-3:]), flush=True)
        return f

    t0 = time.time()
    results = {}
    if "M0" in models:
        # ---- M0: frozen Bloch factor at X, s̄
        c0, f0 = bloch_vector(X, args.sbar, gwin, args.mono_px, args.band)
        print(f"  monolayer band-{args.band} at X, s̄: f = {f0:.6f}")
        T0 = np.zeros((Nb, Ne), complex)
        for ie, env in enumerate(envs):
            for ig, g in enumerate(gwin):
                T0[col_index(env, g), ie] = c0[ig]
        results["M0 plain EA (frozen Bloch)"] = solve_T(T0, "M0 plain EA (frozen Bloch)")
    if "M1" in models:
        # ---- M1: exact local dispersion (Bloch factor at X+G_env)
        T1 = np.zeros((Nb, Ne), complex)
        for ie, env in enumerate(envs):
            q = X + env[0] * bp[:, 0] + env[1] * bp[:, 1]
            cg, _ = bloch_vector(q, args.sbar, gwin, args.mono_px, args.band)
            for ig, g in enumerate(gwin):
                T1[col_index(env, g), ie] = cg[ig]
        results["M1 + local dispersion"] = solve_T(T1, "M1 + local dispersion")
    if "M2" in models:
        # ---- M2: + registry frames (K x K)
        K = args.nref
        sgrid = [(i / K, j / K) for i in range(K) for j in range(K)]
        T2 = np.zeros((Nb, Ne * len(sgrid)), complex)
        for ks, sk in enumerate(sgrid):
            for ie, env in enumerate(envs):
                q = X + env[0] * bp[:, 0] + env[1] * bp[:, 1]
                cg, _ = bloch_vector(q, sk, gwin, args.mono_px, args.band)
                for ig, g in enumerate(gwin):
                    T2[col_index(env, g), ks * Ne + ie] = cg[ig]
        results[f"M2 + registry ({K}x{K})"] = solve_T(T2, f"M2 + registry ({K}x{K})")
    if "M3" in models:
        results["M3 exact (full window)"] = solve_T(np.eye(Nb, dtype=complex),
                                                    "M3 exact (full window)")
    print(f"  ({time.time()-t0:.0f}s)")
    np.savez(os.path.join(HERE, f"pwe_ea_ladder_m{m}.npz"),
             **{f"f_{i}": v for i, v in enumerate(results.values())},
             labels=np.array(list(results.keys())), m=m)
    print("saved pwe_ea_ladder npz")


if __name__ == "__main__":
    main()
