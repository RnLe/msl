#!/usr/bin/env python3
"""Momentum-space k.p moire model (Bistritzer-MacDonald style) for the ASYM
band-1 manifold.

    H_{GG'}(k) = E_ref(k0 + k + G) delta_{GG'} + Vtilde(G - G')

  * E_ref(q): EXACT local band-1 dispersion (from momentum_kp_ref.py),
    interpolated on the half-moire-g grid. Replaces the parabolic k.p that
    over-binds in the strong-modulation regime.
  * Vtilde(dG): FFT_s of the registry landscape Lambda_1(X;s) (phase-1),
    the moire potential coupling, included to all orders by diagonalisation.

Plane-wave cutoff |n_i| <= NCUT (physical band-limiting). Solves the 4 folded
moire lanes {Gamma, X1, X2, M} of the (m,1)=2x2 tiling and pools them, then
compares to the FDFD asym reference ladder.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import phase2_blaze_v4 as p2  # noqa: E402


def build_vtilde(phase1_npz, band_lo=1):
    d = np.load(phase1_npz, allow_pickle=False)
    ev = d["eigenvalues"]
    nreg = int(round(ev.shape[0] ** 0.5))
    lam_ref = float(np.mean(ev[..., band_lo]))
    Lam1 = ev[..., band_lo].reshape(nreg, nreg) - lam_ref
    Vhat = np.fft.fft2(Lam1) / (nreg * nreg)   # Vhat[dn1,dn2]
    return Vhat, nreg, lam_ref


def solve_lane(Eref, js, Vhat, nreg, ncut, lane_half):
    JMAX = (len(js) - 1) // 2
    ns = [(n1, n2) for n1 in range(-ncut, ncut + 1)
          for n2 in range(-ncut, ncut + 1)]
    NB = len(ns)
    idx = {n: i for i, n in enumerate(ns)}
    H = np.zeros((NB, NB), dtype=complex)
    a1, a2 = lane_half                       # 0 or 1 (half-g offsets)
    for (n1, n2), i in idx.items():
        j1 = JMAX + 2 * n1 + a1
        j2 = JMAX + 2 * n2 + a2
        H[i, i] = Eref[j1, j2]
    for (n1, n2), i in idx.items():
        for (m1, m2), jx in idx.items():
            if i == jx:
                continue
            # AXIS PAIRING: Vhat = fft2(Lam1[sy, sx]) so Vhat axis-0 is the
            # sy-registry harmonic and axis-1 the sx-harmonic. A first-principles
            # FFT of the real-space moiré potential V(r)=Λ(s(r)) shows the strong
            # sx-modulation drives the g1 moiré vector and sy drives g2; hence the
            # g1-difference (n1-m1) must index the sx (axis-1) harmonic and the
            # g2-difference (n2-m2) the sy (axis-0) harmonic — i.e. the indices
            # are SWAPPED relative to the naive [(n1-m1),(n2-m2)] order (which
            # transposes the coupling and swaps the strong/weak harmonics).
            H[i, jx] += Vhat[(n2 - m2) % nreg, (n1 - m1) % nreg]
    H = 0.5 * (H + H.conj().T)
    w = np.linalg.eigvalsh(H)
    return np.sqrt(np.maximum(w, 0)) / (2 * np.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eref", default=os.path.join(HERE, "eref_disp.npz"))
    ap.add_argument("--phase1", default=os.path.join(
        HERE, "phase1_asym_b1_reg128", "square_x_asym_tm_phase1.npz"))
    ap.add_argument("--ncut", type=int, default=6)
    ap.add_argument("--fdfd", default=os.path.join(HERE, "fdfd_asym_x_2deg_res16.npz"))
    ap.add_argument("--window", type=float, nargs=2, default=[0.370, 0.383])
    ap.add_argument("--out", default=os.path.join(HERE, "momentum_kp_m57.npz"))
    args = ap.parse_args()

    ref = np.load(args.eref)
    Eref, js = ref["Eref"], ref["js"]
    Vhat, nreg, lam_ref = build_vtilde(args.phase1)

    LANES = {"Gamma": (0, 0), "X1": (1, 0), "X2": (0, 1), "M": (1, 1)}
    pooled = []
    per_lane = {}
    for name, half in LANES.items():
        f = solve_lane(Eref, js, Vhat, nreg, args.ncut, half)
        per_lane[name] = np.sort(f)
        pooled.append(f)
    pooled = np.sort(np.concatenate(pooled))

    lo, hi = args.window
    ea_win = pooled[(pooled >= lo) & (pooled <= hi)]
    fd = np.sort(np.load(args.fdfd)["freqs"])
    fd_win = fd[(fd >= lo) & (fd <= hi)]

    print(f"NCUT={args.ncut}  basis/lane={(2*args.ncut+1)**2}")
    print(f"EA pooled bottom {pooled[0]:.6f}   FDFD bottom {fd[0]:.6f}  "
          f"(edge d={pooled[0]-fd[0]:+.2e})")
    print(f"in-window [{lo},{hi}]: EA={len(ea_win)}  FDFD={len(fd_win)}  "
          f"(ratio {len(ea_win)/max(len(fd_win),1):.2f})")
    n = min(len(ea_win), len(fd_win))
    if n:
        d = np.abs(ea_win[:n] - fd_win[:n])
        print(f"index-aligned (first {n}): mean|df|={d.mean():.2e} max={d.max():.2e}")
    print("EA first 12:", " ".join(f"{x:.6f}" for x in pooled[:12]))
    print("FDFD first 12:", " ".join(f"{x:.6f}" for x in fd[:12]))
    np.savez(args.out, pooled=pooled, ncut=args.ncut,
             **{f"lane_{k}": v for k, v in per_lane.items()})


if __name__ == "__main__":
    main()
