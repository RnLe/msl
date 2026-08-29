#!/usr/bin/env python3
"""The geometric explanation of the sector-ladder anatomy.

Left  — momentum space: the monolayer BZ partitioned into the three valley basins,
        the folded momentum lattice of one commensuration, the energy-ceiling contour
        around M2 that bounds the a-priori validity domain of the single-valley
        envelope theory, and the in-domain momenta highlighted.
Right — the mechanism: registry-averaged band-1 dispersion along the M2 -> M3 cut vs
        the M2-frame envelope surface. Inside the basin they coincide; beyond it the
        envelope surface assigns the folded momenta wrong energies — those are
        exactly the extra envelope levels, while the true energies there belong to
        the M3 valley the single-valley model does not carry (the unmatched
        reference levels). Same momenta, two energies.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ladder_wide as lw
import valley_diagnosis as vd
from lib_v5 import lattice as lat
from lib_v5 import lifted as lf
from lib_v5 import micro_pwe as mp
from lib_v5 import raw_projection as rp

HERE = os.path.dirname(os.path.abspath(__file__))
C_TRUE = "#0072B2"
C_EA = "#009E73"
BASIN_TINT = {0: "#D55E00", 1: "#009E73", 2: "#0072B2"}
B0 = vd.B0
BREC0 = vd.BREC0


def ea_surface(kappas):
    """The frozen-M2-frame envelope symbol along a momentum path: h11(k) =
    u1(M2)^H C(k) u1(M2) with C the hermitized collocation operator of the
    registry-averaged material — the surface the single-band model assigns to
    every envelope harmonic."""
    fine = 128
    gmax = vd.GMAX_MONO
    c = vd.avg_coeffs()
    k0 = BREC0 @ np.array(vd.M_FRAC["M2"])
    h0, R, ns, kG = rp.mono_hermitized(c, k0, B0, gmax, fine)
    import scipy.linalg as sla
    w0, V0 = sla.eigh(h0)
    u1 = V0[:, vd.cand.BAND]
    G = kG - k0[None, :]
    out = []
    for kap in kappas:
        kin = ((G + k0[None, :] + kap[None, :]) ** 2).sum(1)
        out.append(float(np.real(u1.conj() @ (R @ (kin[:, None] * R) @ u1))))
    return np.array(out)


def main(mn=(32, 31)):
    m, n = mn
    floor = vd.band1_avg(*vd.M_CART["M2"])
    ceil_e = vd.ceiling()
    Bs = lf.supercell_basis(vd.LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T

    fig, (axk, axd) = plt.subplots(
        1, 2, figsize=(11.6, 5.6), gridspec_kw=dict(width_ratios=[1.0, 1.25]))

    # ---- left: momentum space
    b1, b2 = BREC0[:, 0], BREC0[:, 1]
    span = 0.72 * np.linalg.norm(b1)
    g = np.linspace(-span, span, 240)
    KX, KY = np.meshgrid(vd.M_CART["M2"][0] + g, vd.M_CART["M2"][1] + g,
                         indexing="ij")
    bas, _ = vd.basin_of(KX.reshape(-1), KY.reshape(-1))
    bas = bas.reshape(KX.shape)
    for i in range(3):
        axk.contourf(KX, KY, (bas == i).astype(float), levels=[0.5, 1.5],
                     colors=[BASIN_TINT[i]], alpha=0.10)
        axk.contour(KX, KY, (bas == i).astype(float), levels=[0.5],
                    colors=["0.6"], linewidths=0.7)
    # energy-ceiling contour around M2 (the domain boundary)
    ng = 74
    ge = np.linspace(-0.42 * np.linalg.norm(b1), 0.42 * np.linalg.norm(b1), ng)
    E = np.full((ng, ng), np.nan)
    for i, gx in enumerate(ge):
        for j, gy in enumerate(ge):
            kx, ky = vd.M_CART["M2"][0] + gx, vd.M_CART["M2"][1] + gy
            bb, _ = vd.basin_of(kx, ky)
            if bb[0] == 1:
                E[i, j] = vd.band1_avg(kx, ky)
    axk.contour(vd.M_CART["M2"][0] + ge[:, None] * np.ones(ng)[None, :],
                vd.M_CART["M2"][1] + np.ones(ng)[:, None] * ge[None, :],
                E, levels=[ceil_e], colors=[C_EA], linewidths=2.2, zorder=6)
    # folded momentum lattice
    dom, dom_e, grid = vd.domain_harmonics(m, n)
    n1, n2, gbas, gE = grid["n1"], grid["n2"], grid["basin"], grid["e"]
    kx = vd.M_CART["M2"][0] + Brec[0, 0] * n1 + Brec[0, 1] * n2
    ky = vd.M_CART["M2"][1] + Brec[1, 0] * n1 + Brec[1, 1] * n2
    view = (np.abs(kx - vd.M_CART["M2"][0]) < span) & \
           (np.abs(ky - vd.M_CART["M2"][1]) < span)
    indom = np.zeros(len(n1), bool)
    dset = {tuple(x) for x in dom}
    for i in range(len(n1)):
        indom[i] = (n1[i], n2[i]) in dset
    axk.plot(kx[view & ~indom], ky[view & ~indom], ".", color="0.55", ms=2.2)
    axk.plot(kx[view & indom], ky[view & indom], "o", color=C_EA, ms=4.6,
             mec="white", mew=0.5, zorder=5)
    for nm, xy in vd.M_CART.items():
        # show each M point plus its lattice copies in view
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                p = xy + da * b1 + db * b2
                if (abs(p[0] - vd.M_CART["M2"][0]) < span
                        and abs(p[1] - vd.M_CART["M2"][1]) < span):
                    axk.plot(*p, "x", color="0.15", ms=6, mew=1.6)
                    axk.annotate(nm, p, textcoords="offset points",
                                 xytext=(5, 4), fontsize=9, color="0.15")
    axk.set_aspect("equal")
    axk.set_xlabel(r"$k_x$")
    axk.set_ylabel(r"$k_y$")
    axk.set_title(
        f"The sector's momenta sample every valley\n"
        f"(m,n)=({m},{n}): dots = folded momenta; green = the a-priori domain\n"
        f"(M2 basin at or below the M3 floor, green contour)", fontsize=9.5)

    # ---- right: the dispersion cut along both principal harmonic directions
    b = np.linalg.norm(b1)
    dirs = {"light": Brec[:, 0] / np.linalg.norm(Brec[:, 0]),
            "heavy": Brec[:, 1] / np.linalg.norm(Brec[:, 1])}
    ts = np.linspace(-0.42, 0.42, 240) * b
    for name, dhat in dirs.items():
        ks = vd.M_CART["M2"][None, :] + ts[:, None] * dhat[None, :]
        true_e = np.array([
            mp.solve(vd.avg_coeffs(), k, B0, vd.GMAX_MONO,
                     n_bands=2)[0][vd.cand.BAND] for k in ks])
        ea_e = ea_surface(ts[:, None] * dhat[None, :])
        ls = "-" if name == "light" else (0, (5, 2))
        axd.plot(ts / b, true_e, ls=ls, color=C_TRUE, lw=3.4, alpha=0.9)
        axd.plot(ts / b, ea_e, ls=ls, color=C_EA, lw=1.5)
    axd.axhspan(floor, ceil_e, color="0.5", alpha=0.08, lw=0)
    axd.axhline(ceil_e, color="0.3", lw=1.0, ls=":")
    # the folded harmonics of this commensuration along each direction
    for name, nmax in (("light", 4), ("heavy", 2)):
        dhat = dirs[name]
        step = np.linalg.norm(Brec[:, 0 if name == "light" else 1])
        for k in range(-nmax, nmax + 1):
            kap = k * step * dhat
            e_t = vd.band1_avg(*(vd.M_CART["M2"] + kap))
            if e_t < floor + 0.115:
                axd.plot(k * step / b, e_t, "o", color=C_TRUE, ms=4,
                         mfc="white", mew=1.2, zorder=5)
            e_m = float(ea_surface(kap[None, :])[0])
            if e_m < floor + 0.115:
                axd.plot(k * step / b, e_m, "D", color=C_EA, ms=3.6, zorder=5)
    axd.annotate("true band 1", (0.30, floor + 0.104), fontsize=9,
                 color=C_TRUE)
    axd.annotate("fixed-frame envelope surface\n(nearly isotropic: both directions)",
                 (0.14, floor + 0.014), fontsize=9, color=C_EA)
    axd.annotate("light direction (solid): green rides on blue --\n"
                 "surfaces coincide, these rungs match at $10^{-7}$",
                 (0, floor - 0.0135), ha="center", fontsize=8.5, color="0.25")
    axd.annotate("heavy direction (dashed): the fixed frame misses the\n"
                 "remote-band mass -- wrong energies INSIDE the basin",
                 (0, floor + 0.088), ha="center", fontsize=8.5, color="0.25")
    axd.annotate("domain ceiling (M3 floor)", (-0.41, ceil_e + 0.0015),
                 fontsize=8, color="0.3")
    axd.set_xlabel(r"$\kappa$ along the harmonic directions   ($|b|$ units)")
    axd.set_ylabel(r"eigenvalue $\lambda$")
    axd.set_ylim(floor - 0.02, floor + 0.115)
    axd.set_xlim(-0.44, 0.44)
    axd.set_title("Why some in-basin rungs still fail: the model surface is\n"
                  "exact along one principal direction, flat along the other",
                  fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_valley_geometry.{ext}"), dpi=170,
                    bbox_inches="tight")
    print("saved fig_valley_geometry.{png,pdf}")


if __name__ == "__main__":
    main(tuple(int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (32, 31))
