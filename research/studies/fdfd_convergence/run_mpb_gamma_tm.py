#!/usr/bin/env python3
"""
MPB TM Γ-point eigensolve for moiré supercells.
Runs at res=64 per unit cell for 8° and 4° angles.
Saves one .npz per (angle, freq_target) into data_gamma_tm_mpb/.
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data_gamma_tm_mpb')
os.makedirs(DATA_DIR, exist_ok=True)

R_OVER_A = 0.2
EPS_ROD  = 8.9
EPS_BG   = 1.0
RES      = 64        # per unit cell
N_MODES  = 20

ANGLES = [
    {'m': 14, 'n': 1, 'label': '8deg'},
    {'m': 29, 'n': 1, 'label': '4deg'},
]
TARGET_FREQS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]


def freq_tag(f):
    return f'f{f * 100:03.0f}'


def build_geometry(m, n):
    """Build MPB lattice + geometry for a bilayer moiré supercell."""
    import meep as mp
    from meep import mpb

    L1 = np.array([m, n], dtype=float)
    L2 = np.array([-n, m], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_rad = 2 * np.arctan2(n, m)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    B_super = np.column_stack([L1, L2])
    B_inv = np.linalg.inv(B_super)
    r_mpb = R_OVER_A / L_super

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0),
        basis2=mp.Vector3(L2[0], L2[1], 0))

    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m - 2, m + n + 2):
            for i2 in range(-n - 2, m + n + 2):
                pos = i1 * a1 + i2 * a2
                frac = B_inv @ pos
                f1, f2 = frac[0] % 1.0, frac[1] % 1.0
                if f1 >= 0.5: f1 -= 1.0
                if f2 >= 0.5: f2 -= 1.0
                geometry.append(mp.Cylinder(
                    radius=r_mpb,
                    center=mp.Vector3(f1, f2, 0),
                    material=mp.Medium(epsilon=EPS_ROD)))

    return lattice, geometry, L_super


def run_mpb(m, n, label, sigma_omega):
    """Run MPB at Gamma, TM, for one frequency target."""
    import meep as mp
    from meep import mpb

    dest = os.path.join(DATA_DIR,
                        f'mpb_tm_gamma_{label}_res{RES}_{freq_tag(sigma_omega)}.npz')
    if os.path.isfile(dest):
        print(f'  [skip] {os.path.basename(dest)}')
        return True

    lattice, geometry, L_super = build_geometry(m, n)
    print(f'  {label}  res={RES}  σ_ω={sigma_omega}  rods={len(geometry)}  '
          f'L={L_super:.1f}')
    sys.stdout.flush()

    # Find how many bands we need: target_freq * L_super gives MPB-scale freq.
    # We want 20 eigenvalues near sigma_omega in a/2πc.
    # MPB gives ALL bands from 0 upward. We need enough bands.
    # Estimate: for freq f in a/2πc, MPB freq = f * L_super.
    # Request enough bands to cover target + margin.
    mpb_target = sigma_omega * L_super
    n_bands = max(N_MODES, int(mpb_target * L_super * 4) + 40)
    # Cap to avoid absurd memory use
    n_bands = min(n_bands, 800)

    mp.verbosity(0)
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=n_bands,
        resolution=RES,
        k_points=[mp.Vector3(0, 0, 0)])

    # Suppress MPB output
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    try:
        t0 = time.time()
        ms.run_tm()
        t_solve = time.time() - t0
    finally:
        os.dup2(o1, 1); os.dup2(o2, 2)
        os.close(fd); os.close(o1); os.close(o2)

    freqs_mpb_raw = np.array(ms.all_freqs)[0]
    freqs = freqs_mpb_raw / L_super  # convert to a/2πc

    # Select 20 nearest to target
    dists = np.abs(freqs - sigma_omega)
    idx = np.argsort(dists)[:N_MODES]
    idx = np.sort(idx)
    freqs_sel = freqs[idx]

    np.savez(dest,
             freqs=freqs_sel,
             freqs_all=freqs,
             n_bands_total=n_bands,
             sigma_omega=sigma_omega,
             m=m, n=n, resolution=RES,
             t_solve=t_solve)

    print(f'  ✓ {t_solve:.1f}s  {n_bands} bands  '
          f'selected freq=[{freqs_sel[0]:.6f}, {freqs_sel[-1]:.6f}]')
    return True


def main():
    total = len(ANGLES) * len(TARGET_FREQS)
    ok = 0
    t_start = time.time()
    print(f'MPB TM Γ-point — {total} runs (res={RES}/cell)')

    for a in ANGLES:
        for tf in TARGET_FREQS:
            print(f'\n[{ok+1}/{total}]  {a["label"]}  f={tf}')
            if run_mpb(a['m'], a['n'], a['label'], tf):
                ok += 1

    print(f'\nDone.  {ok}/{total} succeeded, {time.time()-t_start:.0f}s total')


if __name__ == '__main__':
    main()
