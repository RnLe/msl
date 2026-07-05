#!/usr/bin/env python
"""Get accurate omega0 for square lattice, TM band 3 at M, at res=64."""
import numpy as np, os
import meep as mp
from meep import mpb

a = 1.0
r_over_a = 0.2
eps_rod = 11.56
eps_bg = 1.0

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
geometry = [mp.Cylinder(
    radius=r_over_a,
    center=mp.Vector3(0, 0, 0),
    material=mp.Medium(epsilon=eps_rod),
)]

for res in [32, 64, 128]:
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=eps_bg),
        num_bands=10,
        resolution=res,
        k_points=[mp.Vector3(0.5, 0.5, 0)],
    )
    mp.verbosity(0)
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    ms.run_tm()
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)

    freqs = np.array(ms.all_freqs[0])
    print(f"res={res:4d}: bands 0-9 = {freqs}")
    print(f"         band3 = {freqs[3]:.8f}")
    if res > 32:
        print(f"         gap_below = {freqs[3]-freqs[2]:.6f}, gap_above = {freqs[4]-freqs[3]:.6f}")
