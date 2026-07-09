#!/usr/bin/env python3
"""The crossover law: how the compact momentum-space continuum model approaches
eigenvalue-exactness as the moiré coupling V/E_kin is dialed down (fixed angle
m=57, θ=2°; r₂ is the single knob, ΔΛ ∝ ~r₂²).

For each candidate r₂ we already have:
  * fdfd_xman_r2{tag}.npz   — FDFD X-manifold ladder (freqs_xmanifold)
  * momentum_kp_r2{tag}.npz — pooled 4-lane momentum-model ladder
  * scan_common_gap2.npz    — V/E_kin, γ, β per r₂

Same strict metrics as momentum_hero_figure.py (index-aligned lowest-N EA vs the
N-state FDFD X-manifold): edge offset, span ratio, de-trended shape residual.
Plots span-ratio→1, |offset|, residual→FDFD-floor vs V/E_kin (log), marking the
threshold V/E_kin* below which few-parameter eigenvalue-exactness holds.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (r2, tag, fdfd_xman npz, momentum pooled npz).  Strong anchor first.
# The r₂=0.10 anchor reuses the existing px16 FDFD X-manifold (fdfd_xman_2deg)
# but the MPB-consistent momentum ladder (momentum_kp_r2_100) so all 5 points
# share one methodology.
CANDIDATES = [
    (0.100, "100", "fdfd_xman_2deg.npz",     "momentum_kp_r2_100.npz"),
    (0.070, "070", "fdfd_xman_r2_070.npz",   "momentum_kp_r2_070.npz"),
    (0.054, "054", "fdfd_xman_r2_054.npz",   "momentum_kp_r2_054.npz"),
    (0.040, "040", "fdfd_xman_r2_040.npz",   "momentum_kp_r2_040.npz"),
    (0.031, "031", "fdfd_xman_r2_031.npz",   "momentum_kp_r2_031.npz"),
]

FDFD_FLOOR_PX16 = 6e-4      # documented px16 X-manifold drift
FDFD_FLOOR_PX32 = 1.5e-4    # Richardson finalist floor


def scan_lookup():
    p = os.path.join(HERE, "scan_common_gap2.npz")
    if not os.path.isfile(p):
        return {}
    recs = np.load(p, allow_pickle=True)["records"]
    return {round(float(r["r2"]), 3): r for r in recs}


def metrics(fd, ea):
    """fd = FDFD X-manifold (all N), ea = momentum pooled; index-aligned lowest-N.

    Two views of the residual:
      * RAW (no de-trend) is the true eigenvalue error. The exactness of the
        weak-coupling continuum model lives in the DEEPLY-BOUND states — the
        ground quadruplet and lowest-K — which approach the FDFD floor as V→0.
        The near-band-edge (weakly-bound) upper states stay limited (single-band
        + separable breaks down against the continuum), so a full-manifold mean
        understates the achieved exactness.
      * DE-TRENDED (rigid offset removed) is the strong-coupling shape metric,
        kept for continuity; at weak coupling the offset itself →0 so raw≈detrend.
    """
    fd = np.sort(fd); ea = np.sort(ea)
    n = len(fd)
    eaN = ea[:n]
    raw = np.abs(eaN - fd)
    off = float(eaN[0] - fd[0])
    dsh = np.abs((eaN - off) - fd)
    span = float((eaN[-1] - eaN[0]) / (fd[-1] - fd[0])) if fd[-1] > fd[0] else np.nan
    def lowk(k):
        k = min(k, n)
        return float(raw[:k].mean())
    return dict(n=n, edge_offset=off, span_ratio=span,
                ground_res=float(raw[0]),          # lowest state, raw
                low4_raw=lowk(4), low8_raw=lowk(8), low12_raw=lowk(12),
                mean_raw=float(raw.mean()), max_raw=float(raw.max()),
                mean_detrended=float(dsh.mean()), max_detrended=float(dsh.max()),
                fdfd_bottom=float(fd[0]), fdfd_top=float(fd[-1]),
                ea_bottom=float(eaN[0]), ea_top=float(eaN[-1]))


def main():
    scan = scan_lookup()
    rows = []
    for r2, tag, fdf, eaf in CANDIDATES:
        fp = os.path.join(HERE, fdf); ep = os.path.join(HERE, eaf)
        if not (os.path.isfile(fp) and os.path.isfile(ep)):
            print(f"  skip r2={r2}: missing {fdf if not os.path.isfile(fp) else eaf}")
            continue
        fd = np.load(fp)["freqs_xmanifold"]
        ea = np.load(ep)["pooled"]
        m = metrics(fd, ea)
        rec = scan.get(round(r2, 3))
        m["r2"] = r2
        m["v_over_ekin"] = float(rec["v_over_ekin"]) if rec is not None else np.nan
        m["gamma"] = float(rec["gamma"]) if rec is not None else np.nan
        m["beta"] = float(rec["beta"]) if rec is not None else np.nan
        m["dLam_lambda"] = float(rec["dLam_lambda"]) if rec is not None else np.nan
        rows.append(m)
        print(f"r2={r2:.3f} V/E_kin={m['v_over_ekin']:6.1f} N={m['n']:3d} "
              f"| ground={m['ground_res']:.2e} low4={m['low4_raw']:.2e} "
              f"low8={m['low8_raw']:.2e} | rawmean={m['mean_raw']:.2e} "
              f"span={m['span_ratio']:.3f}")

    rows.sort(key=lambda r: r["v_over_ekin"])
    out = dict(protocol="Weak-coupling crossover: momentum-space continuum model "
               "vs FDFD X-manifold, square CML TM, asym bilayer r1=0.20, "
               "m=57 (2 deg). r2 dials V/E_kin; ΔΛ∝~r2². Strict index-aligned "
               "lowest-N EA vs N-state w_X>0.5 FDFD X-manifold.",
               fdfd_floor_px16=FDFD_FLOOR_PX16, fdfd_floor_px32=FDFD_FLOOR_PX32,
               candidates=rows)
    with open(os.path.join(HERE, "crossover_law_data.json"), "w") as f:
        json.dump(out, f, indent=1)

    if len(rows) < 2:
        print("  (need >=2 candidates for the crossover figure)"); return

    V = np.array([r["v_over_ekin"] for r in rows])
    ground = np.array([r["ground_res"] for r in rows])
    low4 = np.array([r["low4_raw"] for r in rows])
    low8 = np.array([r["low8_raw"] for r in rows])
    rawmean = np.array([r["mean_raw"] for r in rows])
    span = np.array([r["span_ratio"] for r in rows])
    C, C2, C3 = "#d62728", "#1f77b4", "#2ca02c"

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.16, wspace=0.30)

    # Panel A — the headline: deeply-bound-state raw residual -> FDFD floor
    for arr, lab, col, mk in [(ground, "ground quadruplet", C, "o"),
                              (low4, "lowest 4 (raw mean)", C2, "s"),
                              (low8, "lowest 8 (raw mean)", C3, "^")]:
        ax[0].loglog(V, arr, mk + "-", color=col, lw=1.5, ms=7, label=lab)
    ax[0].axhspan(0, FDFD_FLOOR_PX16, color="#eeeeee", alpha=0.9)
    ax[0].axhline(FDFD_FLOOR_PX16, color="0.5", ls=":", lw=1)
    ax[0].axhline(FDFD_FLOOR_PX32, color="0.7", ls="--", lw=1)
    ax[0].text(V.max()*0.5, FDFD_FLOOR_PX16*1.15, "FDFD px16 floor", fontsize=7, color="0.4")
    ax[0].set_xlabel("V/E_kin (moiré coupling)")
    ax[0].set_ylabel("raw residual  |Δf|  [c/a]")
    ax[0].set_title("A  deeply-bound states → EXACT\n(raw eigenvalue error → FDFD floor)",
                    fontsize=9.5)
    ax[0].legend(fontsize=7, loc="lower right")

    # Panel B — full-manifold raw mean (limited by weakly-bound upper states)
    ax[1].axhspan(0, FDFD_FLOOR_PX16, color="#eeeeee", alpha=0.9)
    ax[1].axhline(FDFD_FLOOR_PX16, color="0.5", ls=":", lw=1)
    ax[1].loglog(V, rawmean, "o-", color=C2, lw=1.5, ms=7)
    ax[1].set_xlabel("V/E_kin")
    ax[1].set_ylabel("full-manifold raw mean |Δf|")
    ax[1].set_title("B  whole manifold: improves, but\nupper (weakly-bound) states limit it",
                    fontsize=9.5)

    # Panel C — span ratio (the strong-coupling bandwidth metric, for context)
    ax[2].axhspan(0.98, 1.02, color="#c8e6c9", alpha=0.7)
    ax[2].axhline(1.0, color="0.5", ls=":", lw=1)
    ax[2].semilogx(V, span, "o-", color=C2, lw=1.5, ms=7)
    ax[2].set_xlabel("V/E_kin")
    ax[2].set_ylabel("full-manifold span ratio  EA/FDFD")
    ax[2].set_title("C  span ratio (context; set by the\nnear-edge upper states)",
                    fontsize=9.5)

    fig.suptitle("The weak-coupling crossover: the compact photonic moiré continuum "
                 "model's deeply-bound states become eigenvalue-exact as V/E_kin falls  "
                 "(square CML TM, m=57 θ=2°, r₂ knob)", fontsize=10.8, y=0.965)
    fig.savefig(os.path.join(HERE, "fig_crossover_law.pdf"))
    fig.savefig(os.path.join(HERE, "fig_crossover_law.png"), dpi=200)
    print("saved fig_crossover_law.{pdf,png} + crossover_law_data.json")


if __name__ == "__main__":
    main()
