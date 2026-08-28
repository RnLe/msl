#!/usr/bin/env python3
"""Adapted-frame hero family: the registry-adapted single-band raw projection across
the commensurate family, against the certified PWE reference (hero_family.npz, or
recomputed if absent) and the FDFD leg. Includes an Ns-bump certification at the
calibration angle and the eta-scaling fit.
"""
import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))
import candidate_hexM as cand  # noqa: E402
import hero_engine as he  # noqa: E402
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GMAX_MONO = 4
FINE = 192


def adapted_value(m, n, Ns):
    A = lat.supercell_A(cand.LATTICE, m, n)
    A2 = lf.layer2_integer_matrix(cand.LATTICE, m, n)
    W = np.asarray(A2, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(cand.LATTICE, m, n)

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(Ns * Ns)]
    frames = he.adapted_frames(cand.coeffs, cand.CARRIER_FRAC, GMAX_MONO, deltas,
                               [cand.BAND], FINE)
    H_P = he.lazy_project(cand.coeffs, cand.CARRIER_FRAC, GMAX_MONO, Ns, reg,
                          np.linalg.inv(Bs).T, frames, FINE)
    w = np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))
    lo, hi = cand.WINDOW[0] - 0.004, cand.WINDOW[1] + 0.004
    win = w[(w >= lo) & (w <= hi)]
    return win


def main():
    fam = np.load(os.path.join(HERE, "hero_family.npz"), allow_pickle=True)
    fd = np.load(os.path.join(HERE, "fdfd_leg_ladders.npz"), allow_pickle=True)
    family = [tuple(x) for x in fam["family"]]

    # Ns certification at the calibration angle
    v17 = adapted_value(5, 4, 17)
    v21 = adapted_value(5, 4, 21)
    print(f"Ns certification (5,4): Ns17 {v17[0]:.7f}  Ns21 {v21[0]:.7f}  "
          f"drift {abs(v17[0] - v21[0]):.2e}", flush=True)

    rows = []
    for m, n in family:
        t0 = time.time()
        win = adapted_value(m, n, 17)
        ref = np.atleast_1d(fam[f"ref_{m}_{n}"])
        lam_fd = float(np.atleast_1d(fd[f"{m}_{n}_extrap"])[0])
        unc_fd = float(np.atleast_1d(fd[f"{m}_{n}_unc"])[0])
        eta = float(np.atleast_1d(fam[f"eta_{m}_{n}"])[0])
        dev = abs(win[0] - ref[0]) if len(win) == 1 and len(ref) == 1 else np.nan
        dev_fd = abs(win[0] - lam_fd) if len(win) == 1 else np.nan
        f0 = np.sqrt(ref[0]) / (2 * np.pi)
        rows.append(dict(m=m, n=n, eta=eta, ea=win, ref=ref, lam_fd=lam_fd,
                         unc_fd=unc_fd, dev=dev, dev_fd=dev_fd))
        print(f"({m},{n}) eta={eta:.4f} count {len(win)} ea {win[0]:.7f} "
              f"ref {ref[0]:.7f} fdfd {lam_fd:.7f}(±{unc_fd:.0e}) "
              f"|ea-ref| {dev:.3e} (f: {dev/(8*np.pi**2*f0):.2e}) "
              f"|ea-fdfd| {dev_fd:.3e} ({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in rows if np.isfinite(r["dev"])]
    if len(ok) >= 3:
        etas = np.array([r["eta"] for r in ok])
        devs = np.array([r["dev"] for r in ok])
        p = np.polyfit(np.log(etas), np.log(devs), 1)[0]
        print(f"\nadapted-frame residual exponent: eta^{p:.2f}", flush=True)
    np.savez(os.path.join(HERE, "hero_adapted.npz"),
             **{f"{k}_{r['m']}_{r['n']}": r[k] for r in rows
                for k in ("ea", "ref", "lam_fd", "unc_fd", "eta", "dev", "dev_fd")},
             family=np.array(family), ns=17, gmax_mono=GMAX_MONO,
             ns_cert_drift=abs(v17[0] - v21[0]))
    print("saved hero_adapted.npz")


if __name__ == "__main__":
    main()
