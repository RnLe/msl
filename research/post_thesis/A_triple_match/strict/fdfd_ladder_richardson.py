#!/usr/bin/env python3
"""Stage B reference: Richardson-extrapolate the full FDFD manifold-window ladder.

The 2 deg FDFD window states come in exact 4-fold quadruplets (two T_P1 doublets, section
15) at every resolution, so cross-resolution matching is by quadruplet index. Each
quadruplet center is extrapolated O(1/px^2) from res16/32/48 (floor_reconciliation.py
recipe, which gives the continuum ground 0.370907 +/- 5.7e-6); the 16/32 vs 32/48
extrapolant difference is the quoted uncertainty. Quadruplets missing at res48 (mode-count
truncation) fall back to the 16/32 extrapolant with a widened uncertainty.

Output: fdfd_ladder_2deg.npz (quadruplet centers f_ext, sigma, per-res values, splits).
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LO, HI = 0.3661, 0.3785


def quads(freqs):
    f = np.sort(np.asarray(freqs).ravel())
    w = f[(f >= LO) & (f <= HI)]
    assert len(w) % 4 == 0, f"window count {len(w)} not a multiple of 4"
    q = w.reshape(-1, 4)
    return q.mean(axis=1), (q.max(axis=1) - q.min(axis=1))


def main():
    centers, splits = {}, {}
    for r in (16, 32, 48):
        d = np.load(os.path.join(HERE, f"fdfd_asym_x_2deg_res{r}.npz"))
        centers[r], splits[r] = quads(d["freqs"])
        print(f"res{r}: {len(centers[r])} quadruplets, max intra-split "
              f"{splits[r].max():.1e}", flush=True)

    n = len(centers[16])
    f_ext = np.zeros(n)
    sig = np.zeros(n)
    used = []
    for i in range(n):
        f16, f32 = centers[16][i], centers[32][i]
        e1632 = (4 * f32 - f16) / 3
        if i < len(centers[48]):
            f48 = centers[48][i]
            e3248 = f48 + (f48 - f32) * (48 ** -2) / (32 ** -2 - 48 ** -2)
            f_ext[i] = e3248
            sig[i] = max(abs(e3248 - e1632), 1e-6)
            used.append("16/32/48")
        else:
            f_ext[i] = e1632
            sig[i] = max(abs(e1632 - f32) * 0.35, 2e-5)  # h^2-tail estimate, widened
            used.append("16/32")
    print(f"\nextrapolated ladder ({n} quadruplets):")
    for i in range(n):
        print(f"  q{i}: {f_ext[i]:.6f} +/- {sig[i]:.1e}  ({used[i]}; "
              f"res16 {centers[16][i]:.6f})", flush=True)
    np.savez(os.path.join(HERE, "fdfd_ladder_2deg.npz"),
             f_ext=f_ext, sigma=sig,
             c16=centers[16], c32=centers[32], c48=centers[48],
             used=np.array(used))
    print("saved fdfd_ladder_2deg.npz")


if __name__ == "__main__":
    main()
