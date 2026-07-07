"""Asymmetric-bilayer supercell dielectric builder (square lattice).

Layer 1: rods radius r1 (fixed lattice); layer 2: rods radius r2, rotated by
the commensurate twist. Faithful clone of
T_direct_validation.supercell_geometry.build_supercell_eps conventions
(arithmetic subpixel averaging on the fractional grid), extended with:
  - per-layer radii (r1, r2) and dielectric (eps1, eps2)
  - cell='centered'  -> L1=(m,n), L2=(-n,m)      [the usual CML cell]
    cell='primitive' -> P1=((m-n)/2,(m+n)/2), P2=(-(m+n)/2,(m-n)/2)
    (primitive requires m,n both odd; area = (m^2+n^2)/2)
"""
import numpy as np


def build_bilayer_eps_asym(m, n, r1, r2, eps1, eps2, eps_bg,
                           Nx, Ny, smoothing_Nsub=8, cell='centered'):
    theta_rad = 2.0 * np.arctan2(n, m)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]])

    if cell == 'centered':
        L1 = np.array([float(m), float(n)])
        L2 = np.array([float(-n), float(m)])
    elif cell == 'primitive':
        if (m - n) % 2 != 0:
            raise ValueError("primitive cell requires m, n of equal parity "
                             "(both odd for coprime m,n)")
        L1 = np.array([(m - n) / 2.0, (m + n) / 2.0])
        L2 = np.array([-(m + n) / 2.0, (m - n) / 2.0])
    else:
        raise ValueError(cell)
    B_super = np.column_stack([L1, L2])

    B1 = np.eye(2)                # layer-1 basis (square, a=1)
    B2 = R @ B1                   # layer-2 basis
    B1_inv = np.linalg.inv(B1)
    B2_inv = np.linalg.inv(B2)

    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]

    layers = [(B1, B1_inv, r1, eps1), (B2, B2_inv, r2, eps2)]

    def _fill_binary(Xp, Yp):
        eps = np.full((Nx, Ny), eps_bg, dtype=np.float64)
        XY = np.stack([Xp, Yp], axis=0)
        for B, B_inv, r, eps_rod in layers:
            frac = np.einsum('ij,jkl->ikl', B_inv, XY)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B, f_near)
            eps[(disp[0] ** 2 + disp[1] ** 2) < r ** 2] = eps_rod
        return eps

    Nsub = smoothing_Nsub
    sub_offsets = (np.arange(Nsub) + 0.5) / Nsub - 0.5
    eps_grid = np.zeros((Nx, Ny), dtype=np.float64)
    ds1, ds2 = 1.0 / Nx, 1.0 / Ny
    for si in sub_offsets:
        for sj in sub_offsets:
            Xs = X + si * ds1 * L1[0] + sj * ds2 * L2[0]
            Ys = Y + si * ds1 * L1[1] + sj * ds2 * L2[1]
            eps_grid += _fill_binary(Xs, Ys)
    eps_grid /= Nsub * Nsub

    g11, g12, g22 = L1 @ L1, L1 @ L2, L2 @ L2
    info = {
        'lattice_type': 'square', 'm': m, 'n': n, 'a': 1.0,
        'theta_rad': theta_rad, 'theta_deg': np.degrees(theta_rad),
        'r1': r1, 'r2': r2, 'eps1': eps1, 'eps2': eps2, 'eps_bg': eps_bg,
        'cell': cell,
        'B_super': B_super,
        'metric': np.array([[g11, g12], [g12, g22]]),
        'N_cells': int(round(abs(np.cross(L1, L2)))),
        'Nx': Nx, 'Ny': Ny,
        'subpixel_smoothing': True, 'smoothing_Nsub': Nsub,
    }
    return eps_grid, info
