"""
2D FDFD eigensolver for moiré photonic crystal supercells.

Supports both polarizations:

  TM (Ez modes): -∇²Ez = (ω/c)² ε Ez   (generalized eigenproblem)
  TE (Hz modes): -∇·(ε⁻¹ ∇Hz) = (ω/c)² Hz  (standard eigenproblem)

on a moiré supercell with oblique lattice vectors L1, L2.

The grid uses fractional supercell coordinates (s1, s2) ∈ [0,1)², with
physical coordinates r = s1·L1 + s2·L2. Derivatives in Cartesian are
expressed via the chain rule:
    ∂/∂x_i = Σ_α (B⁻¹)_{αi} ∂/∂s_α

where B = [L1 | L2] is the supercell basis matrix.

Bloch boundary conditions: f(s + ê_α) = exp(i q·L_α) f(s)
are implemented by modifying the FD stencil with phase factors.

The operators are discretized in manifestly Hermitian PSD form:
  Laplacian: A = Σ_{αβ} g^{αβ} (D_α⁺)† D_β⁺
  TE:        L = Σ_{αβ} g^{αβ} (D_α⁺)† E⁻¹ D_β⁺
  TM:        L_TM = E^{-1/2} A E^{-1/2}  (standard form of generalized problem)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator
from typing import Dict, Tuple, Optional, List
import time


def _build_bloch_forward_diff_1d(N: int, ds: float, phase: complex,
                                  dtype=complex) -> sp.csr_matrix:
    """
    Build 1D forward-difference matrix with Bloch-periodic BC.

    D⁺[i] = (f[i+1] - f[i]) / ds
    Wrap-around: f[N] = phase * f[0]

    Args:
        N: number of grid points
        ds: grid spacing (= 1/N in fractional coords)
        phase: Bloch phase factor exp(i q·L_α)
        dtype: np.float64 for Gamma-point (real), complex otherwise

    Returns:
        (N, N) sparse CSR matrix
    """
    inv_ds = 1.0 / ds

    # D⁺[i,i] = -1/ds, D⁺[i,i+1] = +1/ds
    diag_main = np.full(N, -inv_ds, dtype=dtype)
    diag_plus = np.full(N - 1, inv_ds, dtype=dtype)

    D = sp.diags([diag_main, diag_plus], [0, 1], shape=(N, N),
                 format='lil', dtype=dtype)

    # Bloch wrap: D⁺[N-1, 0] = phase / ds
    wrap_val = inv_ds if dtype == np.float64 else phase * inv_ds
    D[N - 1, 0] = wrap_val

    return D.tocsr()


def build_fdfd_operator(
    eps_grid: np.ndarray,
    info: Dict,
    q_vec: np.ndarray = None,
    polarization: str = 'tm',
) -> sp.csr_matrix:
    """
    Assemble the 2D FDFD operator on an oblique supercell grid with
    Bloch boundary conditions.

    For TM polarization (Ez modes):
        -nabla^2 Ez = (w/c)^2 eps Ez
        Returned operator: L_TM = E^{-1/2} A E^{-1/2}
        where A = sum g^{ab} D_a^dag D_b is the Bloch Laplacian.
        This transforms the generalized eigenproblem to standard form.

    For TE polarization (Hz modes):
        -div(eps^{-1} grad Hz) = (w/c)^2 Hz
        Returned operator: L_TE = sum g^{ab} D_a^dag E^{-1} D_b

    Both forms are manifestly Hermitian positive-semi-definite.

    Args:
        eps_grid: (Nx, Ny) dielectric function on the supercell grid
        info: geometry info dict from build_supercell_eps
        q_vec: (2,) Bloch wavevector in Cartesian k-space (rad/length).
               None or [0,0] for Gamma-point.
        polarization: 'tm' (Ez, default) or 'te' (Hz)

    Returns:
        L: (Nx*Ny, Nx*Ny) sparse Hermitian PSD matrix
    """
    polarization = polarization.lower()
    if polarization not in ('tm', 'te'):
        raise ValueError(f"polarization must be 'tm' or 'te', got '{polarization}'")

    Nx, Ny = eps_grid.shape

    B_super = info['B_super']  # (2, 2), columns are L1, L2
    L1 = B_super[:, 0]
    L2 = B_super[:, 1]

    # Contravariant metric tensor g^{ab} = (B^{-1} B^{-T})_{ab}
    B_inv = np.linalg.inv(B_super)
    g_contra = B_inv @ B_inv.T  # (2, 2)

    # Grid spacings in fractional coords
    ds1 = 1.0 / Nx
    ds2 = 1.0 / Ny

    # Bloch phases: phase_a = exp(i q . L_a)
    if q_vec is None:
        q_vec = np.zeros(2)

    # Detect Gamma point: if q=0, all phases are 1 and operator is real
    is_gamma = np.allclose(q_vec, 0)
    dtype = np.float64 if is_gamma else np.complex128

    phase1 = 1.0 if is_gamma else np.exp(1j * np.dot(q_vec, L1))
    phase2 = 1.0 if is_gamma else np.exp(1j * np.dot(q_vec, L2))

    # Build 1D forward-difference matrices with Bloch phases
    D1_1d = _build_bloch_forward_diff_1d(Nx, ds1, phase1, dtype=dtype)
    D2_1d = _build_bloch_forward_diff_1d(Ny, ds2, phase2, dtype=dtype)

    I1 = sp.eye(Nx, format='csr', dtype=dtype)
    I2 = sp.eye(Ny, format='csr', dtype=dtype)

    # 2D forward-difference operators via Kronecker products
    D1 = sp.kron(D1_1d, I2, format='csr')  # d+/ds1
    D2 = sp.kron(I1, D2_1d, format='csr')  # d+/ds2

    D1h = D1.T if is_gamma else D1.conj().T
    D2h = D2.T if is_gamma else D2.conj().T

    if polarization == 'te':
        # TE (Hz): L = sum g^{ab} D_a^dag eps^{-1} D_b
        eps_inv = 1.0 / eps_grid.ravel()
        E_inv = sp.diags(eps_inv, format='csr')

        L_op = g_contra[0, 0] * (D1h @ E_inv @ D1) + \
               g_contra[1, 1] * (D2h @ E_inv @ D2) + \
               g_contra[0, 1] * (D1h @ E_inv @ D2 + D2h @ E_inv @ D1)
    else:
        # TM (Ez): -nabla^2 Ez = lambda eps Ez
        # Generalized: A x = lambda B x, A = Laplacian, B = diag(eps)
        # Standard form: L_TM = B^{-1/2} A B^{-1/2}, eigenvalues preserved
        A = g_contra[0, 0] * (D1h @ D1) + \
            g_contra[1, 1] * (D2h @ D2) + \
            g_contra[0, 1] * (D1h @ D2 + D2h @ D1)

        eps_inv_sqrt = 1.0 / np.sqrt(eps_grid.ravel())
        S = sp.diags(eps_inv_sqrt, format='csr')
        L_op = S @ A @ S

    # Enforce exact Hermiticity (symmetry for real case)
    if is_gamma:
        L_op = 0.5 * (L_op + L_op.T)
    else:
        L_op = 0.5 * (L_op + L_op.conj().T)

    return L_op


def build_fdfd_generalized_tm(
    eps_grid: np.ndarray,
    info: Dict,
    q_vec: np.ndarray = None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict]:
    """
    Build the generalized eigenproblem A x = λ B x for TM polarization.

    A = Bloch Laplacian  (Hermitian PSD, sparse)
    B = diag(ε)          (positive diagonal, sparse)

    The eigenvalues λ = (ω/c)² are identical to those of the standard-form
    operator L_TM = ε^{-1/2} A ε^{-1/2} returned by build_fdfd_operator().

    Returns:
        A:    (N, N) sparse Bloch Laplacian
        B:    (N, N) sparse diag(ε) mass matrix
        meta: dict with grid/metric info for building FFT preconditioner
    """
    Nx, Ny = eps_grid.shape
    B_super = info['B_super']
    L1 = B_super[:, 0]
    L2 = B_super[:, 1]

    B_inv = np.linalg.inv(B_super)
    g_contra = B_inv @ B_inv.T

    ds1 = 1.0 / Nx
    ds2 = 1.0 / Ny

    if q_vec is None:
        q_vec = np.zeros(2)

    is_gamma = np.allclose(q_vec, 0)
    dtype = np.float64 if is_gamma else np.complex128

    phase1 = 1.0 if is_gamma else np.exp(1j * np.dot(q_vec, L1))
    phase2 = 1.0 if is_gamma else np.exp(1j * np.dot(q_vec, L2))

    D1_1d = _build_bloch_forward_diff_1d(Nx, ds1, phase1, dtype=dtype)
    D2_1d = _build_bloch_forward_diff_1d(Ny, ds2, phase2, dtype=dtype)

    I1 = sp.eye(Nx, format='csr', dtype=dtype)
    I2 = sp.eye(Ny, format='csr', dtype=dtype)

    D1 = sp.kron(D1_1d, I2, format='csr')
    D2 = sp.kron(I1, D2_1d, format='csr')

    D1h = D1.T if is_gamma else D1.conj().T
    D2h = D2.T if is_gamma else D2.conj().T

    A = (g_contra[0, 0] * (D1h @ D1) +
         g_contra[1, 1] * (D2h @ D2) +
         g_contra[0, 1] * (D1h @ D2 + D2h @ D1))

    if is_gamma:
        A = 0.5 * (A + A.T)
    else:
        A = 0.5 * (A + A.conj().T)

    B_mat = sp.diags(eps_grid.ravel(), format='csr')

    meta = {
        'g_contra': g_contra,
        'Nx': Nx, 'Ny': Ny,
        'ds1': ds1, 'ds2': ds2,
        'is_gamma': is_gamma,
    }
    return A, B_mat, meta


def _make_fft_precond(Nx: int, Ny: int, g_contra: np.ndarray) -> LinearOperator:
    """
    FFT-based preconditioner: exact inverse of the Bloch Laplacian at Γ-point.

    For Fourier mode (m1, m2) the discrete Laplacian eigenvalue is:
        λ_A = g^{11}|μ₁|² + g^{22}|μ₂|² + 2g^{12} Re(μ₁*μ₂)
    where μ_α = (exp(i2πm_α/N_α) - 1) / ds_α  (forward-difference eigenvalue).

    The preconditioner applies:  v ↦ IFFT2( FFT2(v) / λ_A ).
    The (0,0) mode (null space of the Laplacian) is regularized.

    Returns:
        LinearOperator of shape (Nx*Ny, Nx*Ny)
    """
    ds1 = 1.0 / Nx
    ds2 = 1.0 / Ny

    m1 = np.arange(Nx)
    m2 = np.arange(Ny)
    M1, M2 = np.meshgrid(m1, m2, indexing='ij')

    # Forward-difference eigenvalues
    mu1 = (np.exp(2j * np.pi * M1 / Nx) - 1) / ds1
    mu2 = (np.exp(2j * np.pi * M2 / Ny) - 1) / ds2

    lam_A = (g_contra[0, 0] * np.abs(mu1)**2 +
             g_contra[1, 1] * np.abs(mu2)**2 +
             g_contra[0, 1] * 2 * np.real(mu1.conj() * mu2))

    # Regularize the (0,0) null mode
    lam_A[0, 0] = 1.0

    inv_lam = (1.0 / lam_A).astype(np.float64)  # purely real at Γ
    N = Nx * Ny

    def matvec(x):
        X = x.reshape(Nx, Ny)
        return np.real(np.fft.ifft2(np.fft.fft2(X) * inv_lam)).ravel()

    def matmat(X):
        k = X.shape[1]
        Xr = X.reshape(Nx, Ny, k)
        Xf = np.fft.fft2(Xr, axes=(0, 1))
        Xf *= inv_lam[:, :, np.newaxis]
        return np.real(np.fft.ifft2(Xf, axes=(0, 1))).reshape(N, k)

    return LinearOperator((N, N), matvec=matvec, matmat=matmat,
                          dtype=np.float64)


def solve_fdfd_lobpcg(
    eps_grid: np.ndarray,
    info: Dict,
    q_vec: np.ndarray = None,
    n_modes: int = 20,
    tol: float = 1e-8,
    maxiter: int = 500,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Solve the TM eigenproblem via LOBPCG with FFT preconditioner.

        A x = λ ε x,    A = Bloch Laplacian,  ε = diag(eps_grid)

    RAM usage: O(N·k) — no matrix factorization.
    The FFT preconditioner is the *exact* inverse of A at the Γ-point,
    so LOBPCG convergence is governed only by the dielectric contrast
    ratio ε_max/ε_min (typically < 10 → fast convergence).

    Args:
        eps_grid: (Nx, Ny) dielectric function on the supercell grid
        info:     geometry dict from build_supercell_eps
        q_vec:    (2,) Bloch wavevector (None or [0,0] for Γ-point)
        n_modes:  number of lowest eigenvalues to compute
        tol:      convergence tolerance for LOBPCG residual norms
        maxiter:  maximum LOBPCG iterations
        verbose:  print progress

    Returns:
        eigenvalues:  (n_modes,) array of λ = (ω/c)², sorted ascending
        eigenvectors: (Nx*Ny, n_modes) array of eigenmodes
        timing:       dict with 'assembly', 'precond', 'solve', 'total' times
    """
    timings = {}
    t_total = time.time()

    # ── Build generalized eigenproblem ──────────────────────────
    t0 = time.time()
    A, B, meta = build_fdfd_generalized_tm(eps_grid, info, q_vec)
    timings['assembly'] = time.time() - t0

    Nx, Ny = meta['Nx'], meta['Ny']
    N = Nx * Ny

    if verbose:
        print(f"  LOBPCG: {N:,} DOF ({Nx}×{Ny}), nnz(A)={A.nnz:,}, "
              f"k={n_modes}, assembly={timings['assembly']:.1f}s")

    # ── FFT preconditioner ──────────────────────────────────────
    t0 = time.time()
    M = _make_fft_precond(Nx, Ny, meta['g_contra'])
    timings['precond'] = time.time() - t0

    # ── Initial guess: random (reproducible) ────────────────────
    rng = np.random.default_rng(42)
    X0 = rng.standard_normal((N, n_modes))

    # ── Solve ───────────────────────────────────────────────────
    t0 = time.time()
    eigenvalues, eigenvectors = lobpcg(
        A, X0, B=B, M=M,
        tol=tol, maxiter=maxiter,
        largest=False,
        verbosityLevel=1 if verbose else 0,
    )
    timings['solve'] = time.time() - t0

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    timings['total'] = time.time() - t_total

    if verbose:
        omega = np.sqrt(np.maximum(eigenvalues, 0))
        print(f"  LOBPCG done: {n_modes} modes in {timings['solve']:.1f}s "
              f"(total {timings['total']:.1f}s)")
        print(f"  ω range: [{omega[0]:.6f}, {omega[-1]:.6f}]")

    return eigenvalues, eigenvectors, timings


def _make_shifted_fft_precond(
    Nx: int, Ny: int, g_contra: np.ndarray, sigma: float,
    eps_mean: float = 1.0, abs_value: bool = False,
) -> LinearOperator:
    """
    FFT preconditioner for the shifted generalized system (A - σ·ε̄·I).

    Approximates (A - σ·B)⁻¹ where B = diag(ε) by replacing ε with
    its spatial mean ε̄.  In Fourier space the Bloch Laplacian A is
    diagonal with eigenvalues λ_A(G), so:

        M⁻¹ v = IFFT2( FFT2(v) / (λ_A - σ·ε̄) )

    If abs_value=True, uses |λ_A - σ·ε̄|⁻¹ which is SPD — required
    for use with MINRES.  If False, signed version for GMRES.
    Regularized where |λ_A - σ·ε̄| is tiny.
    """
    ds1 = 1.0 / Nx
    ds2 = 1.0 / Ny

    m1 = np.arange(Nx)
    m2 = np.arange(Ny)
    M1, M2 = np.meshgrid(m1, m2, indexing='ij')

    mu1 = (np.exp(2j * np.pi * M1 / Nx) - 1) / ds1
    mu2 = (np.exp(2j * np.pi * M2 / Ny) - 1) / ds2

    lam_A = (g_contra[0, 0] * np.abs(mu1)**2 +
             g_contra[1, 1] * np.abs(mu2)**2 +
             g_contra[0, 1] * 2 * np.real(mu1.conj() * mu2))

    shifted = lam_A - sigma * eps_mean
    min_abs = np.max(np.abs(shifted)) * 1e-14

    if abs_value:
        # SPD version: |λ_A - σε̄|⁻¹  — safe for MINRES
        inv_shifted = (1.0 / np.maximum(np.abs(shifted), min_abs)).astype(np.float64)
    else:
        # Signed version for GMRES
        shifted = np.where(np.abs(shifted) < min_abs,
                           np.sign(shifted) * min_abs, shifted)
        shifted[0, 0] = max(abs(shifted[0, 0]), min_abs)
        inv_shifted = (1.0 / shifted).astype(np.float64)

    N = Nx * Ny

    def matvec(x):
        X = x.reshape(Nx, Ny)
        return np.real(np.fft.ifft2(np.fft.fft2(X) * inv_shifted)).ravel()

    return LinearOperator((N, N), matvec=matvec, dtype=np.float64)


def solve_fdfd_hybrid(
    eps_grid: np.ndarray,
    info: Dict,
    q_vec: np.ndarray = None,
    n_modes: int = 20,
    sigma_omega: float = 0.01,
    tol_eigsh: float = 1e-10,
    tol_inner: float = 1e-8,
    maxiter_eigsh: int = 20000,
    maxiter_inner: int = 200,
    inner_solver: str = 'minres',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Hybrid ARPACK + iterative inner solve with FFT preconditioner.

    Uses the **generalized** TM eigenproblem:
        A x = λ B x,    A = Bloch Laplacian,  B = diag(ε)

    Shift-invert: eigsh solves (A - σB)⁻¹ B x = ν x.
    The inner linear system (A - σB) x = b is solved iteratively
    via MINRES (default) or GMRES + FFT preconditioner.

    MINRES (default): uses |λ_A - σε̄|⁻¹ SPD preconditioner.
      - Short 3-term recurrence → O(N) memory per iteration
      - No restart stalling
    GMRES: uses signed (λ_A - σε̄)⁻¹ preconditioner.
      - Stores restart vectors → O(N × restart) memory

    RAM: O(N·k) — no matrix factorization stored.

    Args:
        eps_grid:      (Nx, Ny) dielectric function
        info:          geometry dict from build_supercell_eps
        q_vec:         (2,) Bloch wavevector, None for Γ-point
        n_modes:       number of eigenvalues to compute
        sigma_omega:   shift-invert target in ω units [a/2πc]
        tol_eigsh:     ARPACK convergence tolerance
        tol_inner:     inner solver convergence tolerance
        maxiter_eigsh: max ARPACK iterations
        maxiter_inner: max inner solver iterations per call
        inner_solver:  'minres' (default, recommended) or 'gmres'
        verbose:       print progress

    Returns:
        eigenvalues:  (n_modes,) sorted ascending  [λ = (ω/c)²]
        eigenvectors: (N, n_modes)
        timings:      dict
    """
    timings = {}
    t_total = time.time()

    # ── Build generalized eigenproblem A x = λ B x ──────────────
    t0 = time.time()
    A_op, B_mat, meta = build_fdfd_generalized_tm(eps_grid, info, q_vec)
    timings['assembly'] = time.time() - t0

    Nx, Ny = meta['Nx'], meta['Ny']
    N = Nx * Ny
    sigma = (2 * np.pi * sigma_omega) ** 2
    eps_mean = float(eps_grid.mean())

    # ── FFT preconditioner ──────────────────────────────────────
    g_contra = meta['g_contra']
    use_minres = (inner_solver == 'minres')
    M_precond = _make_shifted_fft_precond(
        Nx, Ny, g_contra, sigma,
        eps_mean=eps_mean, abs_value=use_minres)

    # ── Matrix-free shifted operator (A - σ·B) ─────────────────
    eps_flat = eps_grid.ravel()

    def shifted_matvec(x):
        return A_op @ x - sigma * (eps_flat * x)

    A_shifted_op = LinearOperator((N, N), matvec=shifted_matvec,
                                  dtype=A_op.dtype)

    # ── Inner solve ─────────────────────────────────────────────
    inner_calls = [0]
    inner_failures = [0]

    if use_minres:
        from scipy.sparse.linalg import minres

        def op_inv_matvec(b):
            inner_calls[0] += 1
            x, info_code = minres(A_shifted_op, b, M=M_precond,
                                  rtol=tol_inner, maxiter=maxiter_inner)
            if info_code != 0:
                inner_failures[0] += 1
                if verbose and inner_failures[0] <= 5:
                    print(f"    MINRES: info={info_code} "
                          f"(call #{inner_calls[0]})", flush=True)
            return x
    else:
        from scipy.sparse.linalg import gmres

        def op_inv_matvec(b):
            inner_calls[0] += 1
            x, info_code = gmres(A_shifted_op, b, M=M_precond,
                                 atol=0, rtol=tol_inner,
                                 maxiter=maxiter_inner, restart=50)
            if info_code != 0:
                inner_failures[0] += 1
                if verbose and inner_failures[0] <= 5:
                    print(f"    GMRES: info={info_code} "
                          f"(call #{inner_calls[0]})", flush=True)
            return x

    OPinv = LinearOperator((N, N), matvec=op_inv_matvec, dtype=A_op.dtype)

    if verbose:
        print(f"  Hybrid ({inner_solver.upper()}): {N:,} DOF, "
              f"nnz(A)={A_op.nnz:,}, k={n_modes}, "
              f"σ_ω={sigma_omega}, ε̄={eps_mean:.2f}")
        print(f"  Assembly: {timings['assembly']:.1f}s")

    # ── ARPACK eigsh — generalized shift-invert ─────────────────
    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(
        A_op, k=n_modes, M=B_mat, sigma=sigma, which='LM',
        OPinv=OPinv, maxiter=maxiter_eigsh, tol=tol_eigsh,
    )
    timings['solve'] = time.time() - t0
    timings['inner_calls'] = inner_calls[0]
    timings['inner_failures'] = inner_failures[0]
    timings['inner_solver'] = inner_solver

    del OPinv, M_precond, A_shifted_op

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    timings['total'] = time.time() - t_total

    if verbose:
        omega = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
        print(f"  Hybrid done: {n_modes} modes in {timings['solve']:.1f}s "
              f"(total {timings['total']:.1f}s)")
        print(f"  Inner {inner_solver.upper()} calls: {inner_calls[0]} "
              f"({inner_failures[0]} failures)")
        print(f"  ω range: [{omega[0]:.6f}, {omega[-1]:.6f}]")

    return eigenvalues, eigenvectors, timings


def solve_fdfd_supercell(
    eps_grid: np.ndarray,
    info: Dict,
    q_vec: np.ndarray = None,
    n_modes: int = 20,
    sigma: float = None,
    which: str = 'SM',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the FDFD eigenproblem on a moiré supercell.

    Finds eigenvalues λ = ω²/c² and eigenmodes Hz(s1, s2) of
    L Hz = λ Hz, where L = -∇·(ε⁻¹ ∇).

    Args:
        eps_grid: (Nx, Ny) dielectric function
        info: geometry dict from build_supercell_eps
        q_vec: (2,) Bloch wavevector in Cartesian k-space, or None for Γ
        n_modes: number of eigenvalues/vectors to compute
        sigma: shift for shift-invert mode. If None, auto-selected.
        which: 'SM' for smallest magnitude, 'LM' for shift-invert

    Returns:
        eigenvalues: (n_modes,) array of ω²/c² values, sorted ascending
        eigenvectors: (Nx*Ny, n_modes) array of eigenmodes
    """
    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, info, q_vec)
    t_assemble = time.time() - t0

    N_dof = L_op.shape[0]
    print(f"  FDFD operator: {N_dof} DOF, nnz={L_op.nnz}, "
          f"assembly time={t_assemble:.1f}s")

    # Always use shift-invert for reliable convergence
    t0 = time.time()
    if sigma is None:
        # Default: target the lowest positive eigenvalues
        # Use a small positive shift to avoid the zero mode
        sigma = 0.01

    # Prefer CHOLMOD LDLᵀ factorization over scipy's SuperLU:
    # ~2-5× less memory and 2-10× faster for 2D FDFD Laplacians.
    try:
        from sksparse.cholmod import cholesky
        from scipy.sparse.linalg import LinearOperator

        L_shifted = L_op - sigma * sp.eye(N_dof, format='csc')
        factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
        OPinv = LinearOperator((N_dof, N_dof),
                               matvec=lambda b: factor(b),
                               dtype=L_op.dtype)
        eigenvalues, eigenvectors = eigsh(
            L_op, k=n_modes, sigma=sigma, which='LM',
            OPinv=OPinv, maxiter=10000, tol=1e-10,
        )
        del factor, OPinv, L_shifted
    except ImportError:
        eigenvalues, eigenvectors = eigsh(
            L_op, k=n_modes, sigma=sigma, which='LM',
            maxiter=10000, tol=1e-10,
        )
    t_solve = time.time() - t0

    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert to frequencies: ω = sqrt(λ) (in units of 2πc/a if a is used)
    omega = np.sqrt(np.maximum(eigenvalues, 0))

    print(f"  Eigensolver: {n_modes} modes in {t_solve:.1f}s")
    print(f"  ω range: [{omega[0]:.6f}, {omega[-1]:.6f}]")

    return eigenvalues, eigenvectors


def eigenvectors_to_fields(
    eigenvectors: np.ndarray,
    eps_grid: np.ndarray,
    Nx: int,
    Ny: int,
    polarization: str = 'tm',
    solver: str = 'standard',
) -> np.ndarray:
    """
    Convert raw FDFD eigenvectors to physical field profiles on the grid.

    For TM standard-form solver (build_fdfd_operator):
        L_TM = S A S  where S = diag(ε^{-1/2}).
        Eigenvector y satisfies L_TM y = λ y.
        Physical field: Ez = ε^{1/2} · y.

    For TM generalized solver (solve_fdfd_hybrid / solve_fdfd_lobpcg):
        A x = λ ε x.  The eigenvector x is already the physical Ez.

    For TE (both solvers):
        The eigenvector is directly the physical Hz field.

    Args:
        eigenvectors: (N, n_modes) raw eigenvectors from the solver
        eps_grid:     (Nx, Ny) dielectric function used in the solve
        Nx, Ny:       grid dimensions
        polarization: 'tm' or 'te'
        solver:       'standard' (build_fdfd_operator / solve_fdfd_supercell)
                      or 'generalized' (solve_fdfd_hybrid / solve_fdfd_lobpcg)

    Returns:
        fields: (n_modes, Nx, Ny) array of physical field profiles
    """
    n_modes = eigenvectors.shape[1]
    fields = np.empty((n_modes, Nx, Ny), dtype=eigenvectors.dtype)

    for i in range(n_modes):
        v = eigenvectors[:, i]
        if polarization.lower() == 'tm' and solver == 'standard':
            # Back-transform: Ez = ε^{1/2} · y
            v = np.sqrt(eps_grid.ravel()) * v
        fields[i] = v.reshape(Nx, Ny)

    return fields


def solve_band_structure(
    eps_grid: np.ndarray,
    info: Dict,
    q_path: np.ndarray,
    n_modes: int = 20,
    sigma: float = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute band structure along a q-path through the moiré BZ.

    Args:
        eps_grid: (Nx, Ny) dielectric function
        info: geometry dict from build_supercell_eps
        q_path: (n_q, 2) array of q-points in Cartesian k-space
        n_modes: number of bands to compute
        sigma: shift for eigsh (target frequency squared)
        verbose: print progress

    Returns:
        bands: (n_q, n_modes) array of eigenvalues ω²/c²
    """
    n_q = len(q_path)
    bands = np.zeros((n_q, n_modes))

    for iq, q in enumerate(q_path):
        if verbose and iq % 10 == 0:
            print(f"  q-point {iq+1}/{n_q}: q=({q[0]:.4f}, {q[1]:.4f})")

        eigenvalues, _ = solve_fdfd_supercell(
            eps_grid, info, q_vec=q, n_modes=n_modes, sigma=sigma,
        )
        bands[iq, :] = eigenvalues

    return bands


if __name__ == '__main__':
    """Quick test: solve a 1×1 unit cell and compare with known values."""
    from .supercell_geometry import build_supercell_eps

    print("=" * 60)
    print("FDFD SOLVER TEST: 1×1 hexagonal unit cell")
    print("=" * 60)

    # Build a simple 1×1 hexagonal cell (m=1, n=0 → θ=60°, N=1)
    # For a proper unit test, use (1,0) which gives N=1
    eps, info = build_supercell_eps(
        'hex', m=1, n=0, a=1.0, r_over_a=0.2,
        eps_rod=11.56, eps_bg=1.0, Nx=64, Ny=64,
    )
    print(f"ε grid: {eps.shape}, rod fraction: {(eps > 1.5).mean():.3f}")
    print(f"θ = {info['theta_deg']:.2f}°, N_cells = {info['N_cells']}")

    # Solve at Γ
    eigenvalues, _ = solve_fdfd_supercell(
        eps, info, q_vec=np.zeros(2), n_modes=10,
    )
    omega = np.sqrt(np.maximum(eigenvalues, 0))
    print(f"\nΓ-point frequencies (ω·a/2πc):")
    for i, w in enumerate(omega):
        print(f"  mode {i}: ω = {w:.6f}")
