#!/usr/bin/env python3
"""State-by-state anatomy of the sector spectrum: which valley basin does each
reference state live in, which envelope harmonic carries each EA state, and does the
basin picture account for every unmatched and every extra level?

The sector's folded momenta {M2 + G_moire} sample the whole monolayer BZ. Each
momentum belongs to a valley basin (nearest M point mod the monolayer lattice), and
the registry-averaged band-1 energy at that momentum predicts where its state sits.
The single-M2-band envelope model assigns EVERY envelope harmonic the analytically
continued M2 surface — right inside the M2 basin, wrong outside. Hypothesis: the
extra EA levels and the unmatched reference levels are the same momenta,
mis-energized. This script measures that claim per state.

The a-priori validity domain (declared from monolayer dispersion alone, no reference
data): harmonics whose momentum is (i) nearest to M2 and (ii) at or below the M3
band-1 floor (the energy ceiling above which other-basin states enter the sector).

Commands:
  census    — in-domain state counts per candidate angle (picks the big-run angle)
  diagnose  — full per-state classification at one angle (+ --scaled for the
              uniform asymptotic family), saved to diag_{m}_{n}[s].npz
"""
import argparse
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from sksparse.cholmod import cholesky

import ladder_wide as lw
from ladder_wide import HERE, LATTICE, cand, he, lat, lf, mat  # noqa: F401

from lib_v5 import micro_pwe as mp  # noqa: E402

B0 = lat.monolayer_basis(LATTICE)
BREC0 = 2 * np.pi * np.linalg.inv(B0).T
M_FRAC = {"M1": (0.5, 0.0), "M2": (0.0, 0.5), "M3": (0.5, 0.5)}
M_CART = {k: BREC0 @ np.array(v) for k, v in M_FRAC.items()}
GMAX_MONO = 4

_avg = None
_ecache = {}


def avg_coeffs():
    """Registry-averaged material: the layer-2 star has no (0,0) harmonic, so the
    average over registry is exactly eps0 + layer1."""
    global _avg
    if _avg is None:
        l1, _ = cand.layers()
        _avg = mat.bilayer(cand.EPS0, l1, mat.cosine_layer({}), delta=(0.0, 0.0))
    return _avg


def basin_of(kx, ky):
    """Nearest M point mod the monolayer reciprocal lattice; vectorized.
    Returns index into (M1, M2, M3) and the distance."""
    kx = np.atleast_1d(np.asarray(kx, float))
    ky = np.atleast_1d(np.asarray(ky, float))
    inv = np.linalg.inv(BREC0)
    d = np.full((3, len(kx)), np.inf)
    for i, nm in enumerate(("M1", "M2", "M3")):
        fx = inv[0, 0] * (kx - M_CART[nm][0]) + inv[0, 1] * (ky - M_CART[nm][1])
        fy = inv[1, 0] * (kx - M_CART[nm][0]) + inv[1, 1] * (ky - M_CART[nm][1])
        fx -= np.rint(fx)
        fy -= np.rint(fy)
        dx = BREC0[0, 0] * fx + BREC0[0, 1] * fy
        dy = BREC0[1, 0] * fx + BREC0[1, 1] * fy
        d[i] = np.hypot(dx, dy)
    return np.argmin(d, axis=0), np.min(d, axis=0)


def band1_avg(kx, ky):
    """Registry-averaged band-1 energy at one momentum (cached mod the lattice)."""
    f = np.linalg.inv(BREC0) @ np.array([kx, ky])
    key = (round(float(f[0]) % 1.0, 9), round(float(f[1]) % 1.0, 9))
    if key not in _ecache:
        k = BREC0 @ np.array(key)
        w, _, _, _ = mp.solve(avg_coeffs(), k, B0, GMAX_MONO, n_bands=2)
        _ecache[key] = float(w[cand.BAND])
    return _ecache[key]


def ceiling():
    """The M3 band-1 floor: above it the sector holds other-basin states."""
    return band1_avg(*M_CART["M3"])


_symbol_frame = None


def ea_symbol(kappas):
    """The fixed-M2-frame envelope dispersion surface: h11(kappa) =
    u1(M2)^H C(M2 + kappa) u1(M2) on the registry-averaged material — the energy
    the single-band envelope model assigns to envelope momentum kappa. Its gap to
    band1_avg is the model's a-priori dispersion error (large in the heavy-mass
    direction, where the true mass lives in the remote-band coupling the fixed
    frame cannot carry)."""
    global _symbol_frame
    from lib_v5 import raw_projection as rp
    import scipy.linalg as sla
    if _symbol_frame is None:
        k0 = M_CART["M2"]
        h0, R, ns, kG = rp.mono_hermitized(avg_coeffs(), k0, B0, GMAX_MONO, 128)
        w0, V0 = sla.eigh(h0)
        _symbol_frame = (V0[:, cand.BAND], R, kG - k0[None, :], k0)
    u1, R, G, k0 = _symbol_frame
    out = []
    for kap in np.atleast_2d(kappas):
        kin = ((G + k0[None, :] + np.asarray(kap)[None, :]) ** 2).sum(1)
        out.append(float(np.real(u1.conj() @ (R @ (kin * (R @ u1))))))
    return np.array(out)


def model_domain(m, n, tol, lim=None):
    """The single-band fixed-frame model's a-priori validity domain: basin + ceiling
    (as domain_harmonics) AND |ea_symbol - band1_avg| <= tol at the harmonic's
    momentum. Everything is computed from the monolayer dispersion alone — no
    reference data. Returns (harmonics, true energies, dispersion errors)."""
    dom, dom_e, grid = domain_harmonics(m, n, lim=lim)
    Bs = lf.supercell_basis(LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    kaps = dom @ Brec.T
    de = np.abs(ea_symbol(kaps) - np.array(
        [band1_avg(*(M_CART["M2"] + k)) for k in kaps]))
    keep = de <= tol
    return dom[keep], dom_e[keep], de[keep], dict(all_de=de, grid=grid)


def domain_harmonics(m, n, lim=None):
    """A-priori validity domain: envelope harmonics (integer pairs) whose momentum
    M2 + Brec_sc @ n is in the M2 basin at or below the ceiling. Returns the integer
    pairs, their predicted energies, and the full census grid for context."""
    Bs = lf.supercell_basis(LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    b0 = np.linalg.norm(BREC0[:, 0])
    if lim is None:
        # cover momenta out to 0.45 |b_mono| around M2 (the basin fits well inside)
        lim = int(np.ceil(0.45 * b0 / np.linalg.norm(Brec[:, 0]))) + 2
    ax = np.arange(-lim, lim + 1)
    N1, N2 = np.meshgrid(ax, ax, indexing="ij")
    n1, n2 = N1.reshape(-1), N2.reshape(-1)
    kx = M_CART["M2"][0] + Brec[0, 0] * n1 + Brec[0, 1] * n2
    ky = M_CART["M2"][1] + Brec[1, 0] * n1 + Brec[1, 1] * n2
    bas, _ = basin_of(kx, ky)
    ceil_e = ceiling()
    # energy only needed where the basin test already passed (keeps the cache small)
    e = np.full(len(n1), np.nan)
    sel = np.where(bas == 1)[0]
    for i in sel:
        e[i] = band1_avg(kx[i], ky[i])
    dom = sel[np.array([e[i] <= ceil_e for i in sel])]
    order = dom[np.argsort(e[dom])]
    return (np.stack([n1[order], n2[order]], 1), e[order],
            dict(n1=n1, n2=n2, basin=bas, e=e, lim=lim, ceiling=ceil_e))


def census(angles):
    print(f"ceiling: M3 floor at {ceiling():.6f} "
          f"(+{ceiling() - band1_avg(*M_CART['M2']):.4f} above M2)")
    for m, n in angles:
        t0 = time.time()
        dom, e, _ = domain_harmonics(m, n)
        N = lat.n_cells(lat.supercell_A(LATTICE, m, n))
        print(f"({m},{n})  N_cells={N:6d}  eta={lw.eta(m, n):.5f}  "
              f"in-domain states = {len(dom)}  "
              f"top {e[-1] - e[0]:+.4f} above floor  ({time.time()-t0:.0f}s)",
              flush=True)


def pwe_states(m, n, lo, hi, gcut=4.0, layers=None):
    """Sparse-pencil window eigenpairs + inertia census."""
    P = lw.pencil(m, n, gcut, layers=layers)
    K, S = P["K"], P["S"]
    want = lw.inertia(K, S, hi) - lw.inertia(K, S, lo)
    sigma = 0.5 * (lo + hi)
    fac = cholesky((K - sigma * S).tocsc(), beta=0, mode="simplicial")
    OPinv = spla.LinearOperator(K.shape, matvec=fac, dtype=float)
    k = want + 12
    for _ in range(5):
        w, V = spla.eigsh(K, k=min(k, P["npw"] - 2), M=S, sigma=sigma,
                          which="LM", OPinv=OPinv, tol=1e-11)
        if float(np.max(np.abs(w - sigma))) >= 0.5 * (hi - lo):
            break
        k = int(1.6 * k) + 8
    keep = (w >= lo) & (w <= hi)
    order = np.argsort(w[keep])
    return w[keep][order], V[:, keep][:, order], P, want


def classify_reference(V, P):
    """Per state: basin weights and the dominant momentum coset."""
    kx = P["k_sc"][0] + P["Brec"][0, 0] * P["n1"] + P["Brec"][0, 1] * P["n2"]
    ky = P["k_sc"][1] + P["Brec"][1, 0] * P["n1"] + P["Brec"][1, 1] * P["n2"]
    bas, _ = basin_of(kx, ky)
    W2 = np.abs(V) ** 2
    W2 /= W2.sum(0)
    wb = np.stack([W2[bas == i].sum(0) for i in range(3)])   # (3, nstates)
    # dominant coset: group plane waves by momentum mod the monolayer lattice
    At = np.asarray(P["A"], float).T
    fr = np.linalg.solve(At, np.stack([P["n1"], P["n2"]]).astype(float)) % 1.0
    key = (np.round(fr[0], 6) * 1e6).astype(np.int64) * 10_000_019 \
        + (np.round(fr[1], 6) * 1e6).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    dom_k = np.zeros((V.shape[1], 2))
    dom_e = np.zeros(V.shape[1])
    dom_basin = np.zeros(V.shape[1], int)
    for j in range(V.shape[1]):
        wc = np.bincount(inv, weights=W2[:, j], minlength=len(uk))
        c = np.argmax(wc)
        i = int(np.argmax(np.where(inv == c, W2[:, j], 0.0)))
        dom_k[j] = (kx[i], ky[i])
        dom_e[j] = band1_avg(kx[i], ky[i])
        dom_basin[j] = bas[i]
    return wb, dom_k, dom_e, dom_basin


def ea_states(m, n, lo, hi, Ns, layers=None, fine=192):
    """EA eigenpairs plus per-state envelope-harmonic weights."""
    A = lat.supercell_A(LATTICE, m, n)
    A2i = lf.layer2_integer_matrix(LATTICE, m, n)
    W = np.asarray(A2i, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(LATTICE, m, n)
    l1, l2 = layers if layers is not None else cand.layers()

    def coeffs_fn(d):
        return mat.bilayer(cand.EPS0, l1, l2, delta=d)

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(Ns * Ns)]
    frames = he.adapted_frames(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, deltas,
                               [cand.BAND], fine)
    H_P = he.lazy_project(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, Ns, reg,
                          np.linalg.inv(Bs).T, frames, fine)
    w, V = sla.eigh(0.5 * (H_P + H_P.conj().T))
    keep = (w >= lo) & (w <= hi)
    w, V = w[keep], V[:, keep]
    # envelope momentum content: FFT of the slow position field per state
    env = V.reshape(Ns, Ns, -1)
    F = np.fft.fft2(env, axes=(0, 1)) / Ns
    wgt = np.abs(F) ** 2                      # (Ns, Ns, nstates)
    freqs = np.rint(np.fft.fftfreq(Ns) * Ns).astype(int)
    j1, j2 = np.unravel_index(np.argmax(wgt.reshape(Ns * Ns, -1), axis=0),
                              (Ns, Ns))
    dom_n = np.stack([freqs[j1], freqs[j2]], 1)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    dom_k = M_CART["M2"][None, :] + dom_n @ Brec.T
    dom_share = wgt.reshape(Ns * Ns, -1).max(0)
    return w, dom_n, dom_k, dom_share


def diagnose(m, n, scaled, gcut, Ns):
    t0 = time.time()
    lyr = lw.scaled_layers(m, n)[0] if scaled else None
    tag = f"{m}_{n}" + ("s" if scaled else "")
    dom, dom_e, grid = domain_harmonics(m, n)
    print(f"=== diagnose ({m},{n}){' scaled' if scaled else ''}  "
          f"in-domain harmonics: {len(dom)} ===", flush=True)

    lo = cand.WINDOW[0] - 0.02
    w_r, V_r, P, want = pwe_states(m, n, lo, lo + 0.148, gcut=gcut, layers=lyr)
    hi = float(w_r[0]) + 0.10
    keep = w_r <= hi
    w_r, V_r = w_r[keep], V_r[:, keep]
    wb, rk, re, rbas = classify_reference(V_r, P)
    ceil_e = grid["ceiling"]
    r_in = (rbas == 1) & (re <= ceil_e + 1e-9)
    print(f"  reference: {len(w_r)} states (census {want} in the wide window, "
          f"{time.time()-t0:.0f}s); in-domain {int(r_in.sum())}", flush=True)

    t1 = time.time()
    w_e, en, ek, eshare = ea_states(m, n, lo, hi, Ns, layers=lyr)
    ebas, _ = basin_of(ek[:, 0], ek[:, 1])
    ee = np.array([band1_avg(*k) for k in ek])
    e_in = (ebas == 1) & (ee <= ceil_e + 1e-9)
    print(f"  ea: {len(w_e)} states ({time.time()-t1:.0f}s); "
          f"in-domain {int(e_in.sum())}", flush=True)

    from fig_ladder_wide import match
    pairs, taken = match(w_r, w_e)
    matched_r = np.array([p is not None for p in pairs])
    conv = 8 * np.pi ** 2 * np.sqrt(w_r[0]) / (2 * np.pi)

    n_um = int((~matched_r).sum())
    n_um_out = int(((~matched_r) & (~r_in)).sum())
    n_ex = int((~taken).sum())
    n_ex_out = int(((~taken) & (~e_in)).sum())
    print("\n  --- gate numbers ---")
    print(f"  unmatched reference states: {n_um}; of those out-of-domain: "
          f"{n_um_out}  ({100 * n_um_out / max(n_um, 1):.0f}%)")
    print(f"  extra ea states: {n_ex}; of those out-of-domain: {n_ex_out}  "
          f"({100 * n_ex_out / max(n_ex, 1):.0f}%)")
    n_in_matched = int((matched_r & r_in).sum())
    print(f"  in-domain reference states matched: {n_in_matched}/"
          f"{int(r_in.sum())}")
    dev = np.array([abs(w_e[pairs[i]] - w_r[i]) / conv
                    for i in range(len(w_r)) if pairs[i] is not None and r_in[i]])
    if len(dev):
        print(f"  in-domain matched dev in f: min {dev.min():.1e} "
              f"med {np.median(dev):.1e} max {dev.max():.1e}")

    print("\n  --- per-state table (reference) ---")
    print("   i   lam        wM1  wM2  wM3  basin  E_pred   in  matched")
    for i in range(len(w_r)):
        print(f"  {i:3d}  {w_r[i]:.6f}  {wb[0, i]:.2f} {wb[1, i]:.2f} "
              f"{wb[2, i]:.2f}  {('M1', 'M2', 'M3')[rbas[i]]}   "
              f"{re[i]:+.4f}  {'y' if r_in[i] else '.'}   "
              f"{'y' if matched_r[i] else '.'}")
    print("\n  --- extra ea states ---")
    for j in np.where(~taken)[0]:
        print(f"   ea {w_e[j]:.6f}  n=({en[j, 0]:+d},{en[j, 1]:+d}) "
              f"share {eshare[j]:.2f}  basin {('M1', 'M2', 'M3')[ebas[j]]}  "
              f"E_pred {ee[j]:+.4f}  {'IN-DOMAIN?!' if e_in[j] else 'out'}")

    np.savez(f"{HERE}/diag_{tag}.npz",
             w_r=w_r, wb=wb, r_dom_k=rk, r_e=re, r_basin=rbas, r_in=r_in,
             w_e=w_e, e_dom_n=en, e_dom_k=ek, e_share=eshare, e_basin=ebas,
             e_e=ee, e_in=e_in,
             matched_r=matched_r, ea_taken=taken,
             pairs=np.array([-1 if p is None else p for p in pairs]),
             dom=dom, dom_e=dom_e, ceiling=ceil_e,
             grid_n1=grid["n1"], grid_n2=grid["n2"], grid_basin=grid["basin"],
             grid_e=grid["e"], mn=np.array([m, n]))
    print(f"\n  saved diag_{tag}.npz  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["census", "diagnose"])
    ap.add_argument("--m", type=int, default=18)
    ap.add_argument("--n", type=int, default=17)
    ap.add_argument("--scaled", action="store_true")
    ap.add_argument("--gcut", type=float, default=4.0)
    ap.add_argument("--ns", type=int, default=21)
    args = ap.parse_args()
    if args.cmd == "census":
        census([(18, 17), (32, 31), (49, 48), (55, 54), (60, 59), (65, 64)])
    else:
        diagnose(args.m, args.n, args.scaled, args.gcut, args.ns)
