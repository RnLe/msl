#!/usr/bin/env python3
"""CLEAN weak-coupling crossover via the dielectric-contrast knob eps2, at fixed
r2=0.10 (rods well-resolved at every solver resolution). The residual is pure
MODEL error — no sub-pixel confound. Shows the compact momentum-space continuum
model's deeply-bound states → FDFD as coupling V/E_kin falls.

Per candidate: V/E_kin = ΔΛ / (D·(2πη)²), with ΔΛ from the Λ landscape and the
band-1 curvature D measured from E_ref at X (D changes with eps2, so it is
measured per candidate, not held constant). Ground = lowest-quadruplet mean.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = 57
THETA = 2 * np.arctan2(1, M)
ETA = 2 * np.sin(THETA / 2)
GMOIRE2 = (2 * np.pi * ETA) ** 2          # |g_moire|² = (2πη)²
# Regime-map-consistent kinetic scale: E_kin = <direct_metric>·(2πη)² with the
# phase-1 local mass <direct_metric>=0.585 (the TRUE k·p curvature at X). We do
# NOT use a coarse ½g finite-difference of E_ref (that step is far outside the
# harmonic regime and over-estimates D ~2×). ΔΛ is the directly-measured, robust
# coupling axis; V/E_kin = ΔΛ/E_kin uses the same E_kin as the r₂ sweep so the
# two crossovers share one regime-map axis (ε₂=8.9 ⇒ V/E_kin≈86, the anchor).
DIRECT_METRIC = 0.585
E_KIN = DIRECT_METRIC * GMOIRE2
FLOOR16 = 6e-4

# tag  eps2   fdfd_xman npz
CANDS = [
    ("e89", 8.9, "fdfd_e89.npz", "momentum_e89.npz", "eref_e89.npz", "lambda_e89.npz"),
    ("e50", 5.0, "fdfd_e50.npz", "momentum_e50.npz", "eref_e50.npz", "lambda_e50.npz"),
    ("e35", 3.5, "fdfd_e35.npz", "momentum_e35.npz", "eref_e35.npz", "lambda_e35.npz"),
    ("e25", 2.5, "fdfd_e25.npz", "momentum_e25.npz", "eref_e25.npz", "lambda_e25.npz"),
    ("e20", 2.0, "fdfd_e20.npz", "momentum_e20.npz", "eref_e20.npz", "lambda_e20.npz"),
]


def ground(freqs, k=4):
    f = np.sort(freqs)
    return float(f[:min(k, len(f))].mean())


def main():
    rows = []
    for tag, e2, fdf, eaf, eref, lam in CANDS:
        fp = os.path.join(HERE, fdf); ep = os.path.join(HERE, eaf)
        rf = os.path.join(HERE, eref); lp = os.path.join(HERE, lam)
        if not all(os.path.isfile(x) for x in (ep, rf, lp)):
            print(f"  skip eps2={e2}: missing light outputs"); continue
        lam1 = np.load(lp)["eigenvalues"][:, 1]
        dLam = float(lam1.max() - lam1.min())
        vek = dLam / E_KIN
        gm = ground(np.load(ep)["pooled"])
        rec = dict(eps2=e2, dLam=dLam, E_kin=float(E_KIN),
                   v_over_ekin=float(vek), model_ground=gm)
        if os.path.isfile(fp):
            fd = np.load(fp)["freqs_xmanifold"]
            gf = ground(fd)
            rec.update(n=len(fd), fdfd_ground=gf, ground_res=abs(gf - gm),
                       ground_res_signed=gf - gm)
            print(f"eps2={e2:4.1f} ΔΛ={dLam:.3f} V/E_kin={vek:5.0f} N={len(fd):3d} "
                  f"| model {gm:.6f} FDFD {gf:.6f} res {gf-gm:+.2e}")
        else:
            print(f"eps2={e2:4.1f} ΔΛ={dLam:.3f} V/E_kin={vek:5.0f} | model {gm:.6f} "
                  f"(FDFD pending)")
        rows.append(rec)

    rows.sort(key=lambda r: r["v_over_ekin"])
    json.dump({"protocol": "eps2 clean crossover (r2=0.10 fixed, resolvable rods)",
               "candidates": rows, "fdfd_floor_px16": FLOOR16},
              open(os.path.join(HERE, "eps2_crossover_data.json"), "w"), indent=1)

    have = [r for r in rows if "ground_res" in r]
    if len(have) >= 2:
        V = np.array([r["v_over_ekin"] for r in have])
        res = np.array([r["ground_res"] for r in have])
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.axhspan(0, FLOOR16, color="#eeeeee", alpha=0.9)
        ax.axhline(FLOOR16, color="0.5", ls=":", lw=1)
        ax.text(V.min(), FLOOR16 * 1.1, "FDFD px16 floor", fontsize=8, color="0.4")
        ax.loglog(V, res, "o-", color="#d62728", lw=1.6, ms=8)
        ax.set_xlabel("V/E_kin  (moiré coupling; r₂=0.10 fixed, ε₂ knob)")
        ax.set_ylabel("ground-state residual  |model−FDFD|  [c/a]")
        ax.set_title("Clean weak-coupling crossover (resolvable rods):\n"
                     "model ground → FDFD as coupling weakens", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "fig_eps2_crossover.pdf"))
        fig.savefig(os.path.join(HERE, "fig_eps2_crossover.png"), dpi=200)
        print("saved fig_eps2_crossover.{pdf,png} + eps2_crossover_data.json")


if __name__ == "__main__":
    main()
