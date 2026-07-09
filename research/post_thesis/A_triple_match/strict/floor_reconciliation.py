#!/usr/bin/env python3
"""Stage 0a — reconcile the spectral-consistent FDFD floor and re-baseline residuals.

THE ISSUE (design review, §13). The variational Galerkin is a SPECTRAL method: it uses the
exact |X+G|^2 kinetic operator (continuum), so its variational floor is the CONTINUUM
ground of -del^2 E = lambda eps_bl E, approached FROM ABOVE. But every historical residual
(and galerkin_recip.py's --fdfd comparison) was quoted against the res16 FINITE-DIFFERENCE
FDFD ground (0.370047), which converges to the continuum FROM BELOW. The two references
straddle the continuum by ~0.86e-3 -- comparable to the whole remaining gap. So the residual
was measured against the wrong operator.

THE FIX. Pin ONE floor consistent with a spectral method: the continuum Richardson limit of
the frozen-candidate (m=57, r1=0.20, r2=0.10, eps=8.9) 2 deg X-manifold ground, from the on-disk
res16/32/48 (px16/32/48) ladder. Report the convergence order, the extrapolation with an
uncertainty, and re-baseline every historical Galerkin bottom against it.

CAVEAT (design review finding 6, noted for Stage 2). The Galerkin's OWN mass matrix samples eps_bl on
the px16 (Ngrid=912, Nsub=8) grid and its reference fields at MPB res=64, so its exact
(complete-basis) floor is the continuum-KINETIC / px16-EPS ground, which may differ from the true
continuum by the eps-sampling error (~1e-4). That refinement only matters when a sub-1e-4 exactness
claim is made (Stage 2); the continuum floor pinned here (+/-~5e-6) is what the current +6e-4-scale
residuals must be measured against.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ground_quad(freqs, k=4):
    f = np.sort(np.asarray(freqs, float))
    q = f[:k]
    return float(q.mean()), float(q.max() - q.min())


def richardson(pxs, gs, order):
    """Extrapolate g(px) -> px=inf assuming g = g_inf + C/px^order, from the two finest px."""
    (p1, g1), (p2, g2) = (pxs[-2], gs[-2]), (pxs[-1], gs[-1])
    h1, h2 = p1 ** (-order), p2 ** (-order)
    g_inf = g2 + (g2 - g1) * h2 / (h1 - h2)
    return g_inf


def main():
    # --- 1. load the frozen-candidate res-ladder, extract the ground quadruplet
    pxs, gs, splits = [], [], []
    print("Frozen-candidate 2deg X-manifold res-ladder (m=57, r1=0.20, r2=0.10, eps=8.9):")
    print(f"  {'px':>4} {'quad mean':>12} {'4-fold split':>14}")
    for r in [16, 32, 48]:
        d = np.load(os.path.join(HERE, f"fdfd_asym_x_2deg_res{r}.npz"))
        assert int(d["m"]) == 57 and float(d["r1"]) == 0.20 and float(d["r2"]) == 0.10
        px = int(d["px"])
        gmean, split = ground_quad(d["freqs"])
        pxs.append(px); gs.append(gmean); splits.append(split)
        print(f"  {px:>4} {gmean:>12.6f} {split:>14.2e}")

    # --- 2. convergence-order check: FD with subpixel smoothing is expected O(1/px^2)
    print("\nConvergence-order check (slope of consecutive points):")
    for label, order in [("1/px  ", 1), ("1/px^2", 2)]:
        h = np.array(pxs, float) ** (-order)
        s01 = (gs[1] - gs[0]) / (h[1] - h[0])
        s12 = (gs[2] - gs[1]) / (h[2] - h[1])
        print(f"  {label}: slope(16->32)={s01:+.5f}  slope(32->48)={s12:+.5f}"
              f"   |ratio-1|={abs(s01/s12 - 1):.3f}   (collinear => correct order)")

    # --- 3. Richardson extrapolation (1/px^2), two independent pairs for an uncertainty
    g_fine = richardson(pxs, gs, 2)                      # px32 -> px48 (most reliable)
    g_coarse = richardson(pxs[:2], gs[:2], 2)            # px16 -> px32
    # crude 3-point: also try treating px16->48
    floor = g_fine
    unc = abs(g_fine - g_coarse)
    print(f"\nContinuum floor (1/px^2 Richardson):")
    print(f"  px16->px32 pair : {g_coarse:.6f}")
    print(f"  px32->px48 pair : {g_fine:.6f}   <-- adopted floor")
    print(f"  FLOOR = {floor:.6f}  +/- {unc:.1e}   (spectral-consistent continuum ground)")
    res16 = gs[0]
    shift = floor - res16
    print(f"  res16 FDFD ground = {res16:.6f}  ->  every residual-vs-res16 is inflated by "
          f"{shift:+.2e}")

    # --- 4. re-baseline the historical two-valley Galerkin bottoms
    #   (documented manifold bottoms from STRONG_COUPLING_ANALYSIS.md §11.3/§12.3)
    print("\nRe-baselined residuals (Galerkin bottom - floor) vs the old (bottom - res16):")
    print(f"  {'result':<34} {'bottom':>9} {'old(vs res16)':>14} {'new(vs cont.)':>14}")
    cases = [
        ("single-valley 9fr (§9)",        0.37680, False),
        ("two-valley 9fr band_lo=0 (mr3)", 0.37154, False),
        ("two-valley 9fr band_lo=1 (b1)",  0.37248, False),
        ("gcut5 spurious sub-floor (g5)",  0.36599, True),
    ]
    for name, bottom, spurious in cases:
        old = bottom - res16
        new = bottom - floor
        flag = "  <-- SUB-FLOOR (variational-violating)" if new < 0 else ""
        print(f"  {name:<34} {bottom:>9.5f} {old:>+14.2e} {new:>+14.2e}{flag}")

    print("\nInterpretation:")
    print(f"  * best CLEAN two-valley 2deg bottom is +{0.37154-floor:.2e} above the continuum floor")
    print(f"    (not +{0.37154-res16:.1e} as quoted vs res16): the target gap is ~1.4x smaller.")
    print("  * the gcut5 state stays clearly sub-floor vs the continuum too -> still spurious,")
    print("    confirming the conditioning wall (not rescued by re-baselining).")
    print("  * ACTION: repoint galerkin_recip.py's --fdfd/sub-floor threshold to this floor.")

    np.savez(os.path.join(HERE, "floor_reconciliation.npz"),
             pxs=np.array(pxs), quad_means=np.array(gs), splits=np.array(splits),
             continuum_floor=floor, floor_uncertainty=unc, res16_ground=res16,
             rebaseline_shift=shift)
    print("\nsaved floor_reconciliation.npz")


if __name__ == "__main__":
    main()
