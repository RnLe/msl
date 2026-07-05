"""
Stencil Interpolation Module: K-dependent band data from k-stencil

Given the raw ω(k₀ + δk) stencil data computed at each registry point in
Phase 1, this module provides functions to:

1. Evaluate ω, vg, M_inv at an *arbitrary* K-shift within the stencil patch
   (i.e. at k₀ + K, not just at k₀).
2. Use a 2D polynomial least-squares fit on the stencil grid (overdetermined
   for 7×7 = 49 points fitting 6 polynomial coefficients) to get robust
   interpolated values.
3. Support the dynamic H(K) assembly in Phase 3, where each moiré K-point
   needs its own ω_n(K), vg_n(K), M_inv_n(K).

THEORY
------
The stencil stores ω_n(k₀ + offset_i · dk, k₀ + offset_j · dk) on an
n×n grid centered at k₀. A 2D quadratic polynomial:

    ω(δkx, δky) ≈ c₀ + c₁ δkx + c₂ δky + c₃ δkx² + c₄ δky² + c₅ δkx δky

is fit to the n² stencil points via least squares. From the coefficients:
    ω(K) = polynomial evaluated at (Kx, Ky)
    vg(K) = [∂ω/∂kx, ∂ω/∂ky] = [c₁ + 2c₃ Kx + c₅ Ky, c₂ + 2c₄ Ky + c₅ Kx]
    M_inv(K) = [[∂²ω/∂kx², ∂²ω/∂kx∂ky], [∂²ω/∂kx∂ky, ∂²ω/∂ky²]]
             = [[2c₃, c₅], [c₅, 2c₄]]

For a quadratic fit, M_inv is constant across K (curvature is constant),
but ω and vg vary with K. This is the leading-order correction.

For higher accuracy, a 4th-order polynomial can be used (15 coefficients
for 49 points, still overdetermined), giving K-dependent M_inv as well.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def fit_quadratic_2d(stencil_omega_band, offsets, dk):
    """
    Fit a 2D quadratic polynomial to stencil data for a single band.
    
    Model: ω(δkx, δky) = c0 + c1·δkx + c2·δky + c3·δkx² + c4·δky² + c5·δkx·δky
    
    Args:
        stencil_omega_band: (n_stencil, n_stencil) frequency data for one band
        offsets: list of integer offsets, e.g. [-3,-2,-1,0,1,2,3]
        dk: k-space step size
        
    Returns:
        coeffs: (6,) array [c0, c1, c2, c3, c4, c5]
        residual: RMS residual of the fit
    """
    offsets = np.asarray(offsets)
    n = len(offsets)
    
    # Build design matrix for all n² stencil points
    rows = []
    omega_vals = []
    for ix, ox in enumerate(offsets):
        for iy, oy in enumerate(offsets):
            dkx = ox * dk
            dky = oy * dk
            rows.append([1.0, dkx, dky, dkx**2, dky**2, dkx * dky])
            omega_vals.append(stencil_omega_band[ix, iy])
    
    A = np.array(rows)
    b = np.array(omega_vals)
    
    # Filter out NaN values
    valid = ~np.isnan(b)
    if np.sum(valid) < 6:
        return np.full(6, np.nan), np.nan
    
    A_valid = A[valid]
    b_valid = b[valid]
    
    # Least-squares solve
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_valid, b_valid, rcond=None)
    
    # RMS residual
    fitted = A_valid @ coeffs
    rms = np.sqrt(np.mean((b_valid - fitted)**2))
    
    return coeffs, rms


def fit_quartic_2d(stencil_omega_band, offsets, dk):
    """
    Fit a 2D quartic polynomial to stencil data for a single band.
    
    Model: ω(δkx, δky) = c0 + c1·δkx + c2·δky + c3·δkx² + c4·δky²
           + c5·δkx·δky + c6·δkx³ + c7·δky³ + c8·δkx²·δky + c9·δkx·δky²
           + c10·δkx⁴ + c11·δky⁴ + c12·δkx³·δky + c13·δkx·δky³ + c14·δkx²·δky²
    
    15 coefficients fit on 49 points (7×7 stencil) — well overdetermined.
    
    Args:
        stencil_omega_band: (n_stencil, n_stencil) frequency data for one band
        offsets: list of integer offsets
        dk: k-space step size
        
    Returns:
        coeffs: (15,) array
        residual: RMS residual of the fit
    """
    offsets = np.asarray(offsets)
    
    rows = []
    omega_vals = []
    for ix, ox in enumerate(offsets):
        for iy, oy in enumerate(offsets):
            dkx = ox * dk
            dky = oy * dk
            rows.append([
                1.0, dkx, dky, dkx**2, dky**2, dkx*dky,           # order 0-2
                dkx**3, dky**3, dkx**2*dky, dkx*dky**2,            # order 3
                dkx**4, dky**4, dkx**3*dky, dkx*dky**3, dkx**2*dky**2  # order 4
            ])
            omega_vals.append(stencil_omega_band[ix, iy])
    
    A = np.array(rows)
    b = np.array(omega_vals)
    
    valid = ~np.isnan(b)
    if np.sum(valid) < 15:
        return np.full(15, np.nan), np.nan
    
    A_valid = A[valid]
    b_valid = b[valid]
    
    coeffs, residuals, rank, sv = np.linalg.lstsq(A_valid, b_valid, rcond=None)
    
    fitted = A_valid @ coeffs
    rms = np.sqrt(np.mean((b_valid - fitted)**2))
    
    return coeffs, rms


def evaluate_quadratic(coeffs, Kx, Ky):
    """
    Evaluate quadratic polynomial and its derivatives at (Kx, Ky).
    
    Args:
        coeffs: (6,) array [c0, c1, c2, c3, c4, c5]
        Kx, Ky: k-shift from k₀ (can be arrays for vectorized evaluation)
        
    Returns:
        omega: frequency at k₀ + K
        vg: (2,) or (N, 2) group velocity [dω/dkx, dω/dky]
        M_inv: (2, 2) inverse mass tensor (constant for quadratic)
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    
    omega = c0 + c1*Kx + c2*Ky + c3*Kx**2 + c4*Ky**2 + c5*Kx*Ky
    
    vg_x = c1 + 2*c3*Kx + c5*Ky
    vg_y = c2 + 2*c4*Ky + c5*Kx
    
    M_inv = np.array([[2*c3, c5], [c5, 2*c4]])
    
    if np.ndim(Kx) == 0:
        vg = np.array([vg_x, vg_y])
    else:
        vg = np.stack([vg_x, vg_y], axis=-1)
    
    return omega, vg, M_inv


def evaluate_quartic(coeffs, Kx, Ky):
    """
    Evaluate quartic polynomial and its derivatives at (Kx, Ky).
    
    Args:
        coeffs: (15,) array
        Kx, Ky: k-shift from k₀
        
    Returns:
        omega: frequency at k₀ + K
        vg: (2,) group velocity
        M_inv: (2, 2) inverse mass tensor (K-dependent for quartic)
    """
    c = coeffs
    
    omega = (c[0] + c[1]*Kx + c[2]*Ky + c[3]*Kx**2 + c[4]*Ky**2 + c[5]*Kx*Ky
             + c[6]*Kx**3 + c[7]*Ky**3 + c[8]*Kx**2*Ky + c[9]*Kx*Ky**2
             + c[10]*Kx**4 + c[11]*Ky**4 + c[12]*Kx**3*Ky + c[13]*Kx*Ky**3 + c[14]*Kx**2*Ky**2)
    
    # dω/dkx
    vg_x = (c[1] + 2*c[3]*Kx + c[5]*Ky
            + 3*c[6]*Kx**2 + 2*c[8]*Kx*Ky + c[9]*Ky**2
            + 4*c[10]*Kx**3 + 3*c[12]*Kx**2*Ky + c[13]*Ky**3 + 2*c[14]*Kx*Ky**2)
    
    # dω/dky
    vg_y = (c[2] + 2*c[4]*Ky + c[5]*Kx
            + 3*c[7]*Ky**2 + c[8]*Kx**2 + 2*c[9]*Kx*Ky
            + 4*c[11]*Ky**3 + c[12]*Kx**3 + 3*c[13]*Kx*Ky**2 + 2*c[14]*Kx**2*Ky)
    
    # d²ω/dkx²
    d2_xx = (2*c[3] + 6*c[6]*Kx + 2*c[8]*Ky
             + 12*c[10]*Kx**2 + 6*c[12]*Kx*Ky + 2*c[14]*Ky**2)
    
    # d²ω/dky²
    d2_yy = (2*c[4] + 6*c[7]*Ky + 2*c[9]*Kx
             + 12*c[11]*Ky**2 + 6*c[13]*Kx*Ky + 2*c[14]*Kx**2)
    
    # d²ω/dkxdky
    d2_xy = (c[5] + 2*c[8]*Kx + 2*c[9]*Ky
             + 3*c[12]*Kx**2 + 3*c[13]*Ky**2 + 4*c[14]*Kx*Ky)
    
    if np.ndim(Kx) == 0:
        vg = np.array([vg_x, vg_y])
        M_inv = np.array([[d2_xx, d2_xy], [d2_xy, d2_yy]])
    else:
        vg = np.stack([vg_x, vg_y], axis=-1)
        M_inv = np.stack([
            np.stack([d2_xx, d2_xy], axis=-1),
            np.stack([d2_xy, d2_yy], axis=-1),
        ], axis=-2)
    
    return omega, vg, M_inv


def fit_stencil_polynomials(stencil_omega, offsets, dk, fit_order='quadratic'):
    """
    Fit polynomial surfaces to all stencil data across registry and bands.
    
    Args:
        stencil_omega: (n_registry, n_registry, N_bands, n_stencil, n_stencil)
        offsets: list of integer offsets (e.g. [-3,...,3])
        dk: k-space step size
        fit_order: 'quadratic' (6 coefficients) or 'quartic' (15 coefficients)
        
    Returns:
        poly_coeffs: (n_registry, n_registry, N_bands, n_coeffs) polynomial coefficients
        rms_residuals: (n_registry, n_registry, N_bands) RMS fit residuals
    """
    n_reg1, n_reg2, n_bands = stencil_omega.shape[:3]
    
    fit_func = fit_quartic_2d if fit_order == 'quartic' else fit_quadratic_2d
    n_coeffs = 15 if fit_order == 'quartic' else 6
    
    poly_coeffs = np.full((n_reg1, n_reg2, n_bands, n_coeffs), np.nan)
    rms_residuals = np.full((n_reg1, n_reg2, n_bands), np.nan)
    
    for i in range(n_reg1):
        for j in range(n_reg2):
            for b in range(n_bands):
                coeffs, rms = fit_func(stencil_omega[i, j, b], offsets, dk)
                poly_coeffs[i, j, b] = coeffs
                rms_residuals[i, j, b] = rms
    
    return poly_coeffs, rms_residuals


def interpolate_band_data_at_K(
    stencil_omega,
    registry_omega0,
    registry_vg,
    registry_M_inv,
    offsets, dk, n_registry,
    delta_frac_grid,
    K_moire,
    all_bands, subspace_bands,
    fit_order='quadratic',
    poly_coeffs=None,
):
    """
    Interpolate band data (ω, vg, M_inv) to the moiré grid at a given moiré K-point.
    
    This generalizes extract_multiband_data_from_mpb_v3 to support K ≠ 0.
    At K=0, this returns the same result as the original function (within
    polynomial fit accuracy).
    
    Args:
        stencil_omega: (n_registry, n_registry, N_all, n_stencil, n_stencil) raw stencil data
        registry_omega0: (n_registry, n_registry, N_all) center frequencies (for K=0 fallback)
        registry_vg: (n_registry, n_registry, N_all, 2) center velocities (for K=0 fallback)
        registry_M_inv: (n_registry, n_registry, N_all, 2, 2) center mass tensors (for K=0 fallback)
        offsets: stencil offset array
        dk: k-space step
        n_registry: number of registry samples per direction
        delta_frac_grid: (Ns1, Ns2, 2) fractional registry coordinates
        K_moire: (2,) moiré K-point in units of 2π/a (MPB reciprocal units)
        all_bands: list of all band indices
        subspace_bands: list of subspace band indices
        fit_order: 'quadratic' or 'quartic'
        poly_coeffs: (n_registry, n_registry, N_all, n_coeffs) precomputed polynomial coefficients
                     If None, fits are computed on the fly (use precomputed for K-loops)
        
    Returns:
        omega_grid: (Ns1, Ns2, N_subspace) interpolated frequencies at K
        vg_grid: (Ns1, Ns2, N_subspace, 2) interpolated group velocities at K
        M_inv_grid: (Ns1, Ns2, N_subspace, 2, 2) interpolated mass tensors at K
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    
    Ns1, Ns2 = delta_frac_grid.shape[:2]
    N_subspace = len(subspace_bands)
    subspace_to_all = [all_bands.index(b) for b in subspace_bands]
    
    Kx, Ky = K_moire
    
    evaluate_func = evaluate_quartic if fit_order == 'quartic' else evaluate_quadratic
    
    # Fit polynomials if not provided
    if poly_coeffs is None:
        poly_coeffs, _ = fit_stencil_polynomials(
            stencil_omega, offsets, dk, fit_order=fit_order
        )
    
    # Evaluate polynomial at K for each registry point and band
    n_coeffs = poly_coeffs.shape[-1]
    N_all = len(all_bands)
    
    reg_omega_K = np.zeros((n_registry, n_registry, N_all))
    reg_vg_K = np.zeros((n_registry, n_registry, N_all, 2))
    reg_M_inv_K = np.zeros((n_registry, n_registry, N_all, 2, 2))
    
    for i in range(n_registry):
        for j in range(n_registry):
            for b in range(N_all):
                c = poly_coeffs[i, j, b]
                if np.any(np.isnan(c)):
                    reg_omega_K[i, j, b] = np.nan
                    reg_vg_K[i, j, b] = np.nan
                    reg_M_inv_K[i, j, b] = np.nan
                    continue
                omega_K, vg_K, M_inv_K = evaluate_func(c, Kx, Ky)
                reg_omega_K[i, j, b] = omega_K
                reg_vg_K[i, j, b] = vg_K
                reg_M_inv_K[i, j, b] = M_inv_K
    
    # NaN-fill via nearest neighbor (same as extract_multiband_data_from_mpb_v3)
    for grid in [reg_omega_K, reg_vg_K, reg_M_inv_K]:
        _fill_nans_nd(grid)
    
    # Periodic interpolation to moiré grid (same approach as Phase 1)
    step = 1.0 / n_registry
    x_coords = np.linspace(0, 1 - step, n_registry)
    y_coords = np.linspace(0, 1 - step, n_registry)
    
    def make_periodic_interp(grid_2d):
        extended = np.zeros((n_registry + 1, n_registry + 1))
        extended[:n_registry, :n_registry] = grid_2d
        extended[n_registry, :n_registry] = grid_2d[0, :]
        extended[:n_registry, n_registry] = grid_2d[:, 0]
        extended[n_registry, n_registry] = grid_2d[0, 0]
        x_ext = np.append(x_coords, 1.0)
        y_ext = np.append(y_coords, 1.0)
        return RegularGridInterpolator(
            (x_ext, y_ext), extended,
            method='linear', bounds_error=False, fill_value=None
        )
    
    # Query points from delta_frac_grid
    delta_frac_x = delta_frac_grid[:, :, 0]
    delta_frac_y = delta_frac_grid[:, :, 1]
    query_x = np.mod(delta_frac_x + 0.5, 1.0)
    query_y = np.mod(delta_frac_y + 0.5, 1.0)
    query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
    
    omega_grid = np.zeros((Ns1, Ns2, N_subspace))
    vg_grid = np.zeros((Ns1, Ns2, N_subspace, 2))
    M_inv_grid = np.zeros((Ns1, Ns2, N_subspace, 2, 2))
    
    for sub_idx, all_idx in enumerate(subspace_to_all):
        interp_omega = make_periodic_interp(reg_omega_K[:, :, all_idx])
        omega_grid[:, :, sub_idx] = interp_omega(query_points).reshape(Ns1, Ns2)
        
        for comp in range(2):
            interp_vg = make_periodic_interp(reg_vg_K[:, :, all_idx, comp])
            vg_grid[:, :, sub_idx, comp] = interp_vg(query_points).reshape(Ns1, Ns2)
        
        for ii in range(2):
            for jj in range(2):
                interp_M = make_periodic_interp(reg_M_inv_K[:, :, all_idx, ii, jj])
                M_inv_grid[:, :, sub_idx, ii, jj] = interp_M(query_points).reshape(Ns1, Ns2)
    
    # Regularize mass tensors
    min_abs_eig = 1e-6
    for i in range(Ns1):
        for j in range(Ns2):
            for n in range(N_subspace):
                M = M_inv_grid[i, j, n]
                eigvals, eigvecs = np.linalg.eigh(M)
                mask = np.abs(eigvals) < min_abs_eig
                eigvals = np.where(mask, np.sign(eigvals) * min_abs_eig, eigvals)
                eigvals = np.where(eigvals == 0, min_abs_eig, eigvals)
                M_inv_grid[i, j, n] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return omega_grid, vg_grid, M_inv_grid


def _fill_nans_nd(grid):
    """Fill NaN values in a multi-dimensional grid using nearest-neighbor."""
    from scipy.ndimage import distance_transform_edt
    
    shape = grid.shape
    if len(shape) == 3:
        for k in range(shape[2]):
            sl = grid[:, :, k]
            mask = np.isnan(sl)
            if np.any(mask) and not np.all(mask):
                _, indices = distance_transform_edt(mask, return_indices=True)
                sl[mask] = sl[tuple(indices[:, mask])]
    elif len(shape) == 4:
        for k in range(shape[2]):
            for c in range(shape[3]):
                sl = grid[:, :, k, c]
                mask = np.isnan(sl)
                if np.any(mask) and not np.all(mask):
                    _, indices = distance_transform_edt(mask, return_indices=True)
                    sl[mask] = sl[tuple(indices[:, mask])]
    elif len(shape) == 5:
        for k in range(shape[2]):
            for i in range(shape[3]):
                for j in range(shape[4]):
                    sl = grid[:, :, k, i, j]
                    mask = np.isnan(sl)
                    if np.any(mask) and not np.all(mask):
                        _, indices = distance_transform_edt(mask, return_indices=True)
                        sl[mask] = sl[tuple(indices[:, mask])]
