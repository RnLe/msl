#!/usr/bin/env python3
"""The scaled asymptotic family: layer-2 amplitude a2 ∝ eta^2 keeps V/E_kin fixed
(the uniform asymptotic branch), registry-adapted single-band raw projection vs the
certified PWE reference per scaled material. The measured residual exponent on THIS
family is the asymptotic-order statement; the small-angle member is the e-6 landing.
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
from lib_v5 import materials as mat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY = [(5, 4), (6, 5), (7, 6), (9, 8)]
GMAX_MONO = 4
NS = 17
FINE = 192
A2_REF = cand.LAYER2_AMP          # 0.12 at the (5,4) anchor
ETA_REF = 2 * np.sin(lat.twist_angle(cand.LATTICE, 5, 4) / 2)
WIN = (1.60, 1.68)


def scaled_layers(a2):
    return (mat.cosine_layer(cand.LAYER1_AMPS),
            mat.cosine_layer({h: a2 for h in cand.STAR}))


def run(m, n):
    eta = 2 * np.sin(lat.twist_angle(cand.LATTICE, m, n) / 2)
    a2 = A2_REF * (eta / ETA_REF) ** 2
    l1, l2 = scaled_layers(a2)

    def coeffs_fn(d):
        return mat.bilayer(cand.EPS0, l1, l2, delta=d)

    A = lat.supercell_A(cand.LATTICE, m, n)
    A2i = lf.layer2_integer_matrix(cand.LATTICE, m, n)
    W = np.asarray(A2i, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(cand.LATTICE, m, n)
    Aabs = np.abs(np.asarray(A, float))
    gmax_sc = int(GMAX_MONO * max(Aabs[0, 0] + Aabs[1, 0], Aabs[0, 1] + Aabs[1, 1])
                  + NS // 2 + 6)
    fine_sc = int(2 ** np.ceil(np.log2(max(256, 3 * gmax_sc))))

    t0 = time.time()
    ref = he.supercell_reference(m, n, gmax_sc, fine_sc, layers=(l1, l2))
    wr = ref["w"][(ref["w"] >= WIN[0]) & (ref["w"] <= WIN[1])]

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(NS) / NS
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(NS * NS)]
    frames = he.adapted_frames(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, deltas,
                               [cand.BAND], FINE)
    H_P = he.lazy_project(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, NS, reg,
                          np.linalg.inv(Bs).T, frames, FINE)
    w = np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))
    we = w[(w >= WIN[0]) & (w <= WIN[1])]
    dev = abs(we[0] - wr[0]) if len(we) >= 1 and len(wr) >= 1 else np.nan
    f0 = np.sqrt(wr[0]) / (2 * np.pi) if len(wr) else np.nan
    print(f"({m},{n}) eta={eta:.4f} a2={a2:.4f} counts {len(wr)}/{len(we)} "
          f"ref {wr[0] if len(wr) else np.nan:.7f} ea {we[0] if len(we) else np.nan:.7f} "
          f"|ea-ref| {dev:.3e} (f: {dev/(8*np.pi**2*f0):.2e}) "
          f"[{ref['method']}, cert {ref['cert']:.1e}, {time.time()-t0:.0f}s]",
          flush=True)
    return dict(m=m, n=n, eta=eta, a2=a2, ref=wr, ea=we, dev=dev)


def main():
    rows = [run(m, n) for m, n in FAMILY]
    ok = [r for r in rows if np.isfinite(r["dev"])]
    if len(ok) >= 3:
        etas = np.array([r["eta"] for r in ok])
        devs = np.array([r["dev"] for r in ok])
        p = np.polyfit(np.log(etas), np.log(devs), 1)[0]
        print(f"\nscaled-family residual exponent: eta^{p:.2f}", flush=True)
    np.savez(os.path.join(HERE, "hero_scaled.npz"),
             **{f"{k}_{r['m']}_{r['n']}": r[k] for r in rows
                for k in ("ref", "ea", "eta", "a2", "dev")},
             family=np.array(FAMILY))
    print("saved hero_scaled.npz")


if __name__ == "__main__":
    main()
