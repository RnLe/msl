#!/usr/bin/env python3
"""Richardson analysis: does the FDFD X-manifold ground state converge TO the
momentum model as px grows? At weak coupling the layer-2 rods are sub-pixel at
px16 (radius r₂·px px), so the FDFD ε is under-resolved. The momentum model uses
MPB res-64 (rods fully resolved) = the continuum/px→∞ reference. If the FDFD
ground state → model as px16→px32→px48, the model is the exact limit and the
px16 residual was FDFD discretization error, not model error.

Ground state = mean of the lowest quadruplet (×4 valley/C4). Reports the model
ground, the FDFD ground per px, the residual, and a 1/px² extrapolation.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ground(freqs, k=4):
    f = np.sort(freqs)
    return float(f[:min(k, len(f))].mean())


def main():
    # (tag, r2, model npz, {px: fdfd npz})
    CANDS = [
        ("070", 0.070, "momentum_kp_r2_070.npz",
         {16: "fdfd_xman_r2_070.npz", 32: "fdfd_xman_r2_070_px32.npz"}),
        ("054", 0.054, "momentum_kp_r2_054.npz",
         {16: "fdfd_xman_r2_054.npz", 32: "fdfd_xman_r2_054_px32.npz",
          48: "fdfd_xman_r2_054_px48.npz"}),
        ("040", 0.040, "momentum_kp_r2_040.npz",
         {16: "fdfd_xman_r2_040.npz", 32: "fdfd_xman_r2_040_px32.npz"}),
    ]
    out = []
    for tag, r2, mf, pxs in CANDS:
        mp = os.path.join(HERE, mf)
        if not os.path.isfile(mp):
            continue
        gm = ground(np.load(mp)["pooled"])
        row = {"r2": r2, "model_ground": gm, "px": {}}
        print(f"\nr2={r2}  model ground (MPB res-64) = {gm:.6f}")
        pts = []
        for px in sorted(pxs):
            fp = os.path.join(HERE, pxs[px])
            if not os.path.isfile(fp):
                print(f"  px{px}: (pending)"); continue
            gf = ground(np.load(fp)["freqs_xmanifold"])
            res = gf - gm
            rodpx = r2 * px
            row["px"][px] = {"fdfd_ground": gf, "residual": res, "rod_px": rodpx}
            pts.append((px, gf))
            print(f"  px{px:2d} (rod {rodpx:.2f}px): FDFD ground {gf:.6f}  "
                  f"residual (FDFD−model) {res:+.2e}")
        # 1/px^2 Richardson extrapolation from the two finest px
        if len(pts) >= 2:
            (p1, g1), (p2, g2) = pts[-2], pts[-1]
            h1, h2 = 1.0 / p1**2, 1.0 / p2**2
            g_inf = g2 + (g2 - g1) * h2 / (h1 - h2)   # linear in 1/px^2 -> h=0
            print(f"  1/px² extrap (px{p1}->{p2}): FDFD(px→∞) ≈ {g_inf:.6f}  "
                  f"residual vs model {g_inf - gm:+.2e}")
            row["fdfd_extrap"] = g_inf
            row["extrap_residual"] = g_inf - gm
        out.append(row)
    np.savez(os.path.join(HERE, "richardson_analysis.npz"),
             data=np.array(out, dtype=object))
    print("\nsaved richardson_analysis.npz")


if __name__ == "__main__":
    main()
