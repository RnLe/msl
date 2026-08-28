#!/usr/bin/env python3
"""The hero family run: EA (raw projection) vs certified PWE reference vs the FDFD leg
across the commensurate angle family of the frozen candidate, with the eta-scaling fit.

Reads fdfd_leg_ladders.npz (independent solver leg). Writes hero_family.npz and prints
the per-angle table + the fitted residual exponent.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))
import candidate_hexM as cand  # noqa: E402
import hero_engine as he  # noqa: E402
from lib_v5 import lattice as lat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY = [(4, 3), (5, 4), (6, 5), (7, 6), (9, 8)]
GMAX_MONO = 4
NS = 17


def main():
    fd = np.load(os.path.join(HERE, "fdfd_leg_ladders.npz"), allow_pickle=True)
    rows = []
    for m, n in FAMILY:
        A = np.abs(np.asarray(lat.supercell_A(cand.LATTICE, m, n), float))
        gmax_sc = int(GMAX_MONO * max(A[0, 0] + A[1, 0], A[0, 1] + A[1, 1])
                      + NS // 2 + 6)
        fine = max(256, 3 * gmax_sc)
        # round fine to a friendly FFT size
        fine = int(2 ** np.ceil(np.log2(fine)))
        t0 = time.time()
        r = he.run_angle(m, n, GMAX_MONO, NS, gmax_sc, fine)
        eta = 2 * np.sin(lat.twist_angle(cand.LATTICE, m, n) / 2)
        key = f"{m}_{n}_extrap"
        lam_fd = float(np.atleast_1d(fd[key])[0]) if key in fd else np.nan
        unc_fd = float(np.atleast_1d(fd[f"{m}_{n}_unc"])[0]) \
            if f"{m}_{n}_unc" in fd else np.nan
        row = dict(m=m, n=n, eta=eta, ref=r["ref"], ea=r["ea"],
                   n_ref=r["n_ref"], n_ea=r["n_ea"], method=r["method"],
                   cert=r["cert"], lam_fd=lam_fd, unc_fd=unc_fd,
                   gmax_sc=gmax_sc, fine=fine, secs=time.time() - t0)
        rows.append(row)
        dev = (abs(r["ea"][0] - r["ref"][0])
               if r["n_ref"] == 1 and r["n_ea"] == 1 else np.nan)
        fddev = (abs(r["ref"][0] - lam_fd)
                 if r["n_ref"] == 1 and np.isfinite(lam_fd) else np.nan)
        print(f"({m},{n}) eta={eta:.4f} counts {r['n_ref']}/{r['n_ea']} "
              f"ref {r['ref'][0] if r['n_ref'] else np.nan:.7f} "
              f"ea {r['ea'][0] if r['n_ea'] else np.nan:.7f} "
              f"|ea-ref| {dev:.3e} |ref-fdfd| {fddev:.3e} "
              f"[{r['method']}, cert {r['cert']:.1e}, {row['secs']:.0f}s]",
              flush=True)

    ok = [r for r in rows if r["n_ref"] == 1 and r["n_ea"] == 1]
    if len(ok) >= 3:
        etas = np.array([r["eta"] for r in ok])
        devs = np.array([abs(r["ea"][0] - r["ref"][0]) for r in ok])
        p = np.polyfit(np.log(etas), np.log(devs), 1)[0]
        print(f"\nfitted EA-vs-reference residual exponent: eta^{p:.2f}")
    np.savez(os.path.join(HERE, "hero_family.npz"),
             **{f"{k}_{r['m']}_{r['n']}": r[k] for r in rows
                for k in ("ref", "ea", "eta", "lam_fd", "unc_fd", "cert")},
             family=np.array(FAMILY), gmax_mono=GMAX_MONO, ns=NS)
    print("saved hero_family.npz")


if __name__ == "__main__":
    main()
