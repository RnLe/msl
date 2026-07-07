#!/usr/bin/env python3
"""THE deliverable: strict MPB / FDFD / EA eigenvalue-stripe comparison.

Triangle protocol (no matching, no assignment optimization anywhere):
  A) MPB <-> FDFD  : same (29,1) supercell, same supercell k=(1/2,0),
                     lowest 20 modes, index-aligned from mode 1.
                     Certifies the FDFD reference against an independent
                     exact method. (MPB has no shift-invert, so it can
                     never reach the EA window at small angles; this leg
                     closes the triangle instead.)
  B) FDFD <-> EA   : 2.01 deg (in-zone, beta=0.075), band-edge window,
                     sorted ladders, edge anchored at the spectral bottom.
  C) FDFD <-> EA   : 3.95 deg (marginal zone, beta=0.147), same window.
  D) Nb ladder     : EA edge error vs retained-band count at 2 deg,
                     against the FDFD discretization drift band.

Regenerates figure + JSON from disk; missing lanes are reported, not faked.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FD = Path("/home/renlephy/msl/research/studies/fdfd_convergence/data_x_tm")
MPB = Path("/home/renlephy/msl/research/studies/fdfd_convergence/data_x_tm_mpb_corrected")

GAMMA_GAP = 0.468            # gap-to-midgap, X, band 0->1 (canonical monolayer)
# 2 deg: true spectral edge at 0.22646 with a gap below; hi = px48 ball limit
EDGE_WINDOW = (0.2262, 0.2275)
# 4 deg: NO spectral edge exists (folded background modes continue down to
# <0.2163, clusters every ~1-2e-3); window = ball-complete region of both
# the EA lanes (>=0.224675) and the FDFD DEEP runs
EDGE_WINDOW_4 = (0.2250, 0.2277)

missing: list[str] = []


def theta_deg(m: int, n: int) -> float:
    return math.degrees(2 * math.atan2(n, m))


def load_freqs(path: Path, key: str = "freqs"):
    if not path.exists():
        missing.append(str(path))
        return None
    return np.sort(np.asarray(np.load(path, allow_pickle=True)[key], dtype=float))


def load_ea(path: Path):
    if not path.exists():
        missing.append(str(path))
        return None, None
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["frequencies"], dtype=float), np.asarray(d["k_labels"])


def stripes(ax, x, freqs, color, half=0.32, lw=1.0, alpha=1.0, cover_to=None,
            window_hi=None, window_lo=None):
    for f in freqs:
        ax.hlines(f, x - half, x + half, color=color, lw=lw, alpha=alpha)
    # shade the not-enumerated region of a capture-ball-limited ladder
    if cover_to is not None and window_hi is not None and cover_to < window_hi:
        ax.fill_betweenx([cover_to, window_hi], x - half, x + half,
                         color="0.88", zorder=0.5)
        if window_lo is not None and \
                (window_hi - cover_to) > 0.08 * (window_hi - window_lo):
            ax.text(x, (cover_to + window_hi) / 2, "not\nenum.", ha="center",
                    va="center", fontsize=6, color="0.45")


# ---------------------------------------------------------------- load data
mpb = np.load(MPB / "mpb_tm_x_4deg_res64_20bands.npz", allow_pickle=True)
mpb_f = np.sort(np.asarray(mpb["freqs_all"], dtype=float))

fd_mpbk = {px: load_freqs(FD / f"fdfd_tm_x_4deg_res{px}_fMPBK.npz")
           for px in (16, 32, 64)}
fd_bot2 = {px: load_freqs(FD / f"fdfd_tm_x_2deg_res{px}_fBOTTOM.npz")
           for px in (16, 32, 48)}
# 4 deg: prefer the DEEP runs (sigma=0.2225, 60 modes — the sigma=0.2270
# 40-mode ball did not enumerate the spectral edge)
fd_bot4 = {}
for px in (16, 32, 48):
    deep = FD / f"fdfd_tm_x_4deg_res{px}_fBOTTOMDEEP.npz"
    fd_bot4[px] = (np.sort(np.load(deep)["freqs"]) if deep.exists()
                   else load_freqs(FD / f"fdfd_tm_x_4deg_res{px}_fBOTTOM.npz"))

EA_RUNGS_2DEG = {
    "Nb2+6rem": "phase2_x_2r6_BOTTOM_m57",
    "Nb4": "phase2_x_4r0_BOTTOM_m57",
    "Nb6": "phase2_x_6r0_BOTTOM_m57",
    "Nb8": "phase2_x_8r0_BOTTOM_m57",
}
ea2 = {}
for rung, d in EA_RUNGS_2DEG.items():
    f, lab = load_ea(HERE / d / "commensurate_m57_n1.npz")
    if f is not None:
        ea2[rung] = {"freqs": np.sort(f), "raw": f, "labels": lab}
ea4_f, ea4_lab = load_ea(HERE / "phase2_x_2r6_BOTTOM_m29" / "commensurate_m29_n1.npz")

# ---------------------------------------------------------------- stats
data: dict = {"protocol": "strict: sorted ladders, index alignment only where "
                          "guaranteed (panel A: from absolute mode 1; panels "
                          "B/C: edge-anchored). No Hungarian/assignment matching.",
              "missing_at_generation": missing}

# Panel A stats
fd_ref_A = fd_mpbk[64] if fd_mpbk[64] is not None else fd_mpbk[32]
panelA = {"m": 29, "n": 1, "theta_deg": theta_deg(29, 1),
          "supercell_k_frac": [0.5, 0.0],
          "note": "March MPB lane ran at supercell k=(1/2,0), which is NOT the "
                  "fold of monolayer X (=(1/2,1/2)); FDFD here matches MPB's k.",
          "mpb_freqs": mpb_f.tolist()}
if fd_ref_A is not None:
    nA = min(len(mpb_f), len(fd_ref_A))
    dA = mpb_f[:nA] - fd_ref_A[:nA]
    panelA.update({
        "fdfd_ref_px": 64 if fd_mpbk[64] is not None else 32,
        "fdfd_ref_freqs": fd_ref_A[:nA].tolist(),
        "resid_mpb_minus_fdfd": dA.tolist(),
        "mean_abs": float(np.mean(np.abs(dA))),
        "max_abs": float(np.max(np.abs(dA))),
        "mean_rel": float(np.mean(np.abs(dA) / fd_ref_A[:nA])),
    })
    if fd_mpbk[32] is not None and fd_mpbk[64] is not None:
        n0 = min(len(fd_mpbk[32]), len(fd_mpbk[64]), nA)
        panelA["fdfd_drift_32_64_mean"] = float(
            np.mean(np.abs(fd_mpbk[32][:n0] - fd_mpbk[64][:n0])))
data["A_mpb_vs_fdfd_4deg"] = panelA


def edge_panel(fd_ladders: dict, ea_rungs: dict, m: int, window) -> dict:
    """Edge-anchored strict comparison in the band-edge window.

    All stats are computed on window-clipped ladders (first-in-window
    anchoring) so that shift-invert capture-ball boundaries cannot fake an
    edge. Richardson h^2 extrapolation from the two finest FDFD rungs.
    """
    out = {"m": m, "n": 1, "theta_deg": theta_deg(m, 1),
           "beta": math.radians(theta_deg(m, 1)) / GAMMA_GAP,
           "window": list(window), "fdfd": {}, "ea": {}}

    def clip(f):
        return f[(f >= window[0]) & (f <= window[1])]

    pxs = sorted(px for px, f in fd_ladders.items() if f is not None)
    ref_px = pxs[-1] if pxs else None
    out["fdfd_ref_px"] = ref_px
    for px in pxs:
        f = fd_ladders[px]
        w = clip(f)
        out["fdfd"][f"res{px}"] = {"freqs": f.tolist(),
                                   "raw_min": float(f[0]),
                                   "edge_in_window": float(w[0]) if len(w) else None,
                                   "n_in_window": int(len(w)),
                                   "coverage_max": float(f.max())}
    if len(pxs) >= 2:
        p1, p2 = pxs[-2], pxs[-1]
        w1, w2 = clip(fd_ladders[p1]), clip(fd_ladders[p2])
        n0 = min(len(w1), len(w2))
        if n0:
            out["fdfd_drift_mean"] = float(np.mean(np.abs(w1[:n0] - w2[:n0])))
            out["fdfd_drift_edge"] = float(abs(w1[0] - w2[0]))
            # Richardson: E(px) ~ px^-2  ->  f_inf = f2 + (f2-f1)*p2^-2/(p1^-2-p2^-2)
            fac = (p2 ** -2) / (p1 ** -2 - p2 ** -2)
            out["fdfd_edge_richardson"] = float(w2[0] + (w2[0] - w1[0]) * fac)
    ref_w = clip(fd_ladders[ref_px]) if ref_px is not None else np.array([])
    for rung, e in ea_rungs.items():
        f = e["freqs"]
        w = clip(f)
        r = {"freqs": f.tolist(),
             "k_labels": [str(x) for x in e["labels"]],
             "freqs_by_lane": {lane: sorted(float(x) for x, l in
                                            zip(e["raw"], e["labels"]) if l == lane)
                               for lane in dict.fromkeys(str(x) for x in e["labels"])},
             "raw_min": float(f[0]),
             "edge_in_window": float(w[0]) if len(w) else None,
             "n_in_window": int(len(w)), "coverage_max": float(f.max())}
        if len(ref_w) and len(w):
            r["edge_minus_fdfd_ref"] = float(w[0] - ref_w[0])
            if "fdfd_edge_richardson" in out:
                r["edge_minus_fdfd_richardson"] = float(
                    w[0] - out["fdfd_edge_richardson"])
            # density ratio only over JOINT coverage — both ladders are
            # capture-ball-limited, so raw in-window counts can be fake.
            # Counting reference = FDFD ladder with the LARGEST coverage
            # (finest px can have the smallest ball).
            cnt_px = max(pxs, key=lambda p: float(fd_ladders[p].max()))
            cnt_full = fd_ladders[cnt_px]
            joint_hi = min(window[1], float(cnt_full.max()), float(f.max()))
            n_fd_j = int(np.sum((cnt_full >= window[0]) & (cnt_full <= joint_hi)))
            n_ea_j = int(np.sum((f >= window[0]) & (f <= joint_hi)))
            r["count_ref_px"] = int(cnt_px)
            r["joint_window"] = [window[0], joint_hi]
            r["n_fdfd_joint"] = n_fd_j
            r["n_ea_joint"] = n_ea_j
            r["density_ratio_ea_over_fdfd"] = (n_ea_j / n_fd_j) if n_fd_j else None
        out["ea"][rung] = r
    return out


data["B_2deg_edge"] = edge_panel(fd_bot2, ea2, 57, EDGE_WINDOW)
data["C_4deg_edge"] = edge_panel(
    fd_bot4, {"Nb2+6rem": {"freqs": np.sort(ea4_f), "raw": ea4_f,
                           "labels": ea4_lab}} if ea4_f is not None else {},
    29, EDGE_WINDOW_4)

# ---------------------------------------------------------------- parameters
data["parameters"] = {
    "crystal": {
        "lattice": "square, lattice constant a", "polarization": "TM (E_z)",
        "rod_radius_over_a": 0.2, "eps_rod": 8.9, "eps_background": 1.0,
        "structure": "two identical rod lattices superimposed in-plane, "
                     "twisted by theta (commensurate (m,n): theta=2*atan(n/m))",
        "centered_cell": "the [(m,n),(-n,m)] supercell is a 2x NON-PRIMITIVE "
                         "centered cell: tau=(L1+L2)/2 is an exact lattice "
                         "vector of BOTH layers (verified numerically to "
                         "machine precision for (29,1) and (57,1)). All exact "
                         "x2 degeneracies in FDFD/MPB = primitive-cell "
                         "folding; at Q_X the two folded primitive momenta "
                         "carry the X and X' valley content respectively. "
                         "EA lane bookkeeping: {Gamma_m, M_m} <-> valley X, "
                         "{X1_m, X2_m} <-> valley X' (C4-paired multisets, "
                         "verified to mean 2.2e-5).",
        "cases": {"2deg": {"m": 57, "n": 1, "theta_deg": theta_deg(57, 1),
                           "L_super_over_a": math.sqrt(57**2 + 1),
                           "N_cells": 57**2 + 1},
                  "4deg": {"m": 29, "n": 1, "theta_deg": theta_deg(29, 1),
                           "L_super_over_a": math.sqrt(29**2 + 1),
                           "N_cells": 29**2 + 1}},
    },
    "ea": {
        "engine": "blaze2d V4 exact-TM operator (tm_operator_model='exact'), "
                  "Berry-free (A=None); registry dependence via direct "
                  "matrix-element fields (direct_metric, direct_b, v_pp, gamma)",
        "expansion_point": "monolayer X = (1/2,0)*2pi/a, target band 0",
        "lambda_ref": 2.355835, "f_ref_c_over_a": 0.244283,
        "registry_grid": "128x128", "envelope_grid_Ns": "128x128",
        "fd_order": 4, "shift_invert_target_f": 0.2270, "n_modes_per_lane": 15,
        "k_lanes": "2x2 CML tiling fold: Gamma_m=(0,0), X1_m=(1/2,0), "
                   "X2_m=(0,1/2), M_m=(1/2,1/2) in moire fractional coords; "
                   "pooled, NOT valley-doubled (doubling retracted, session 4)",
        "rungs": {"Nb2+6rem": "2 retained bands + 6 remote via Lowdin resolvent",
                  "Nb4": "4 retained, 0 remote", "Nb6": "6 retained, 0 remote",
                  "Nb8": "8 retained, 0 remote"},
        "accuracy_zone": "beta = theta_rad / gamma_gap(X,0->1)=0.468; "
                         "in-zone beta <~ 0.1: 2deg->0.075 (in), "
                         "4deg->0.147 (marginal)",
    },
    "fdfd": {
        "operator": "L_TM = eps^{-1/2} (sum g^{ab} D_a^dag D_b) eps^{-1/2}, "
                    "Bloch supercell, 2nd-order stencil",
        "smoothing": "subpixel arithmetic eps-averaging, Nsub=8",
        "eigensolver": "CHOLMOD shift-invert (simplicial) + ARPACK eigsh, "
                       "tol 1e-10",
        "bottom_runs": "Q_X=(pi,0) rad/a == supercell k_frac=(1/2,1/2); "
                       "sigma_omega=0.2270; 40 modes; grid = px * round(L/a)",
        "mpbk_runs": "supercell k_frac=(1/2,0) (matches March MPB lane); "
                     "sigma_omega=0.008; 30 modes",
    },
    "mpb": {
        "file": "mpb_tm_x_4deg_res64_20bands.npz (March 22, thesis sprint)",
        "solver": "MPB ModeSolver.run_tm, 64 px/cell (grid 1856^2), 20 bands",
        "k_point_supercell_frac": [0.5, 0.0],
        "geometry": "identical rods, both layers, radius 0.2/L_super in "
                    "supercell frac units",
    },
}

with open(HERE / "strict_triple_data.json", "w") as fh:
    json.dump(data, fh, indent=1)

# ---------------------------------------------------------------- figure
C_FD, C_FD2, C_MPB, C_EA, C_EA2 = "#222222", "#888888", "#1f77b4", "#d62728", "#ff9955"
fig = plt.figure(figsize=(13.5, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.34, wspace=0.30,
                      left=0.06, right=0.98, top=0.90, bottom=0.175)

# --- Panel A: MPB vs FDFD stripes + residuals
axA = fig.add_subplot(gs[0, 0])
lanes_A = [("MPB\n64 px/cell", mpb_f, C_MPB)]
for px in (32, 64):
    if fd_mpbk[px] is not None:
        lanes_A.append((f"FDFD\n{px} px/cell", fd_mpbk[px][:len(mpb_f)], C_FD if px == 64 else C_FD2))
for i, (lab, f, c) in enumerate(lanes_A):
    stripes(axA, i, f, c)
axA.set_xticks(range(len(lanes_A)), [l for l, _, _ in lanes_A], fontsize=8)
axA.set_ylabel("frequency  f a/c")
axA.set_title(f"A  MPB ↔ FDFD, exact methods\n(29,1) θ={theta_deg(29,1):.3f}°, "
              "supercell k=(½,0), lowest 20", fontsize=9.5)
if "mean_abs" in panelA:
    axA.text(0.03, 0.97, f"mean|Δf| = {panelA['mean_abs']:.1e}\n"
             f"max|Δf| = {panelA['max_abs']:.1e}\n"
             + (f"FDFD drift 32→64: {panelA.get('fdfd_drift_32_64_mean', float('nan')):.1e}"
                if 'fdfd_drift_32_64_mean' in panelA else ""),
             transform=axA.transAxes, va="top", fontsize=8,
             bbox=dict(fc="white", ec="0.7", alpha=0.9))

axAr = fig.add_subplot(gs[1, 0])
if "resid_mpb_minus_fdfd" in panelA:
    r = np.array(panelA["resid_mpb_minus_fdfd"])
    axAr.axhline(0, color="0.6", lw=0.8)
    axAr.plot(np.arange(1, len(r) + 1), r, "o-", ms=4, color=C_MPB, lw=1)
    axAr.set_xlabel("mode index (from absolute bottom)")
    axAr.set_ylabel("f(MPB) − f(FDFD)  [c/a]")
    axAr.set_title("A′  per-mode residual, index-aligned", fontsize=9.5)

# --- Panel B: 2 deg edge stripes
axB = fig.add_subplot(gs[0, 1])
lanes_B = []
for px in (16, 32, 48):
    if fd_bot2[px] is not None:
        lanes_B.append((f"FDFD\n{px}px", fd_bot2[px], C_FD2 if px == 16 else C_FD))
for rung, c in [("Nb2+6rem", C_EA), ("Nb8", C_EA2)]:
    if rung in ea2:
        lanes_B.append((f"EA\n{rung}", ea2[rung]["freqs"], c))
for i, (lab, f, c) in enumerate(lanes_B):
    stripes(axB, i, f[(f >= EDGE_WINDOW[0]) & (f <= EDGE_WINDOW[1])], c,
            cover_to=float(f.max()), window_hi=EDGE_WINDOW[1],
            window_lo=EDGE_WINDOW[0])
axB.set_xticks(range(len(lanes_B)), [l for l, _, _ in lanes_B], fontsize=8)
axB.set_ylim(*EDGE_WINDOW)
axB.set_title(f"B  FDFD ↔ EA, band edge, IN-ZONE\n(57,1) θ={theta_deg(57,1):.3f}°, "
              f"β=0.075, k=Q_X", fontsize=9.5)
axB.set_ylabel("frequency  f a/c")
b = data["B_2deg_edge"]
if "Nb2+6rem" in b["ea"] and b.get("fdfd_ref_px"):
    n_fd_win = b["fdfd"]["res" + str(b["fdfd_ref_px"])]["n_in_window"]
    eb = b["ea"]["Nb2+6rem"]
    axB.text(0.03, 0.97,
             f"edge Δ(Nb2+6rem) vs px{b['fdfd_ref_px']} = "
             f"{eb['edge_minus_fdfd_ref']:+.1e}\n"
             f"vs Richardson f∞: {eb.get('edge_minus_fdfd_richardson', float('nan')):+.1e}\n"
             f"FDFD edge drift (2 finest) = {b.get('fdfd_drift_edge', float('nan')):.1e}\n"
             f"joint-coverage counts: FDFD {eb.get('n_fdfd_joint')}"
             f" / EA {eb.get('n_ea_joint')}"
             f"  (×{eb.get('density_ratio_ea_over_fdfd', float('nan')):.2f})",
             transform=axB.transAxes, va="top", fontsize=8,
             bbox=dict(fc="white", ec="0.7", alpha=0.9))

# --- Panel C: 4 deg edge stripes
axC = fig.add_subplot(gs[0, 2])
lanes_C = []
for px in (16, 32, 48):
    if fd_bot4.get(px) is not None:
        lanes_C.append((f"FDFD\n{px}px", fd_bot4[px], C_FD2 if px == 16 else C_FD))
if ea4_f is not None:
    lanes_C.append(("EA\nNb2+6rem", np.sort(ea4_f), C_EA))
for i, (lab, f, c) in enumerate(lanes_C):
    stripes(axC, i, f[(f >= EDGE_WINDOW_4[0]) & (f <= EDGE_WINDOW_4[1])], c,
            cover_to=float(f.max()), window_hi=EDGE_WINDOW_4[1],
            window_lo=EDGE_WINDOW_4[0])
if lanes_C:
    axC.set_xticks(range(len(lanes_C)), [l for l, _, _ in lanes_C], fontsize=8)
axC.set_ylim(*EDGE_WINDOW_4)
axC.set_title(f"C  FDFD ↔ EA, band edge, MARGINAL\n(29,1) θ={theta_deg(29,1):.3f}°, "
              f"β=0.147, k=Q_X", fontsize=9.5)
c4 = data["C_4deg_edge"]
if c4["ea"].get("Nb2+6rem") and c4.get("fdfd_ref_px"):
    e4 = c4["ea"]["Nb2+6rem"]
    axC.text(0.03, 0.97,
             "no spectral edge at 4° (folded back-\n"
             "ground modes continue below window);\n"
             "cluster shifts ≳ cluster gaps at β=0.147\n"
             f"joint-coverage counts: FDFD {e4.get('n_fdfd_joint')}"
             f" / EA {e4.get('n_ea_joint')}"
             f"  (×{(e4.get('density_ratio_ea_over_fdfd') or float('nan')):.2f})",
             transform=axC.transAxes, va="top", fontsize=7.5,
             bbox=dict(fc="white", ec="0.7", alpha=0.9))

# --- Panel D: Nb ladder edge errors at 2 deg
axD = fig.add_subplot(gs[1, 1])
ref2 = fd_bot2.get(data["B_2deg_edge"].get("fdfd_ref_px"))
if ref2 is not None:
    order = ["Nb2+6rem", "Nb4", "Nb6", "Nb8"]
    xs, ys = [], []
    for i, rung in enumerate(order):
        if rung in ea2:
            xs.append(i)
            ys.append(ea2[rung]["freqs"][0] - ref2[0])
    drift = data["B_2deg_edge"].get("fdfd_drift_edge", 0.0)
    axD.axhspan(-drift, drift, color="0.85", label="FDFD edge drift band")
    axD.axhline(0, color="0.4", lw=0.8)
    axD.plot(xs, ys, "s-", color=C_EA, ms=7)
    axD.set_xticks(range(len(order)), order, fontsize=8)
    axD.set_ylabel("EA edge − FDFD edge  [c/a]")
    axD.set_title("D  Nb ladder at 2°: edge error vs retained bands\n"
                  "(non-monotonic — η²-truncated operator is "
                  "non-variational)", fontsize=9.5)
    axD.legend(fontsize=7, loc="lower left")

# --- Panel E: counting functions N(f) at 2 deg — where the EA excess lives
axE = fig.add_subplot(gs[1, 2])
if ref2 is not None and "Nb2+6rem" in ea2:
    eaf = ea2["Nb2+6rem"]["freqs"]
    hi = min(float(ref2.max()), float(eaf.max()))
    fg = np.linspace(EDGE_WINDOW[0], hi, 600)
    axE.step(fg, [np.sum(ref2 <= g) for g in fg], where="post",
             color=C_FD, label=f"FDFD px{data['B_2deg_edge']['fdfd_ref_px']}")
    axE.step(fg, [np.sum(eaf <= g) for g in fg], where="post",
             color=C_EA, label="EA Nb2+6rem (4-lane pool)")
    axE.set_xlabel("frequency  f a/c")
    axE.set_ylabel("N(f)  levels below f")
    axE.set_title("E  level-counting functions, 2° edge\n"
                  "EA ladder squeezed: levels inside FDFD gaps", fontsize=9.5)
    axE.legend(fontsize=7, loc="upper left")
    axE.set_xlim(EDGE_WINDOW[0], hi)
    data["E_counting_2deg"] = {
        "note": "EA density excess is localized (levels inside FDFD spectral "
                "gaps, e.g. 0.2265-0.2267), not a uniform multiplicity factor "
                "- envelope ladder compression, same eta^2-truncation physics "
                "as the edge softening."}

# --- parameter footer
p = data["parameters"]
footer = (
    "Square lattice (const a), TM (Ez) | rods r/a=0.2, ε=8.9 in ε_bg=1.0 | twisted bilayer, both layers in-plane\n"
    f"(57,1): θ={theta_deg(57,1):.4f}°, L=57.009a, 3250 cells | (29,1): θ={theta_deg(29,1):.4f}°, L=29.017a, 842 cells | "
    "accuracy zone β=θ_rad/γ, γ(X,0→1)=0.468: 2°→β=0.075 (in-zone), 4°→β=0.147 (marginal)\n"
    "EA: blaze2d V4 exact-TM (Berry-free), k₀ = monolayer X = (½,0)·2π/a, band 0, λ_ref=2.35584 (f_ref=0.24428) | "
    "registry 128², envelope N_s=128², fd_order 4 | shift-invert f₀=0.2270, 15 modes/lane\n"
    "EA k-lanes: Γ_m,X1_m,X2_m,M_m (2×2 CML fold), pooled, no valley doubling | "
    "FDFD: L=ε^{-1/2}(Σ g^{ab}D†D)ε^{-1/2}, subpixel ε-avg N_sub=8, CHOLMOD shift-invert, tol 1e-10\n"
    "FDFD bottom runs: Q_X=(π,0) ≡ supercell k=(½,½), σ_ω=0.2270 | "
    "MPB: res 64/cell, 20 bands, run_tm, supercell k=(½,0) — FDFD leg A matches MPB's k exactly\n"
    "STRICT protocol: sorted ladders, no Hungarian/assignment matching; panel A index-aligned from absolute mode 1\n"
    "(m,1) supercell = 2× CENTERED cell: τ=(L1+L2)/2 ∈ both layers' lattices (verified exactly) → FDFD ×2 pairs = primitive-\n"
    "cell folding = X⊕X′ valleys at Q_X; EA lanes {Γ_m,M_m} = valley X, {X1_m,X2_m} = valley X′ (C4-paired, verified to 2e-5)"
)
fig.text(0.06, 0.012, footer, fontsize=6.8, family="monospace", va="bottom")
fig.suptitle("Strict MPB / FDFD / EA eigenvalue comparison — square CML, TM at X "
             "(triangle protocol)", fontsize=12, y=0.965)

fig.savefig(HERE / "fig_strict_triple.pdf")
fig.savefig(HERE / "fig_strict_triple.png", dpi=200)
print("saved fig_strict_triple.{pdf,png} + strict_triple_data.json")
if missing:
    print("PENDING (regenerate when these land):")
    for m in missing:
        print("  -", m)
