"""
3-way benchmark: MPB vs FDFD (SuperLU) vs FDFD (CHOLMOD)
for a honeycomb monolayer at the Dirac K' point.

Tests accuracy and performance of CHOLMOD-accelerated shift-invert
before deploying it on the large (30,29) moiré supercell.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import time
import tracemalloc
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import (
    build_supercell_eps, build_monolayer_basis,
)
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle

# ── MPB reference values ──
# Honeycomb TM, eps_rod=11.56, eps_bg=1.0, r/a=0.2
# K' = (2/3, 1/3) in reciprocal lattice coords
# From prior MPB runs:
#   band 1 = band 2 = 0.274409  (Dirac degeneracy)
MPB_DIRAC_FREQ = 0.274409


def get_K_prime_cartesian(B_mono):
    """Compute K' = (2/3, 1/3) in Cartesian k-space."""
    # Reciprocal lattice: G = 2π B^{-T}
    G = 2 * np.pi * np.linalg.inv(B_mono).T
    g1 = G[:, 0]
    g2 = G[:, 1]
    # K' = (2/3) g1 + (1/3) g2
    K_prime = (2.0 / 3.0) * g1 + (1.0 / 3.0) * g2
    return K_prime


def solve_eigsh_cholmod(L_op, n_modes, sigma):
    """eigsh with CHOLMOD LDL^T factorization for shift-invert.

    Uses simplicial mode which supports symmetric INDEFINITE matrices
    (the shifted matrix L - sigma*I is indefinite when sigma is inside
    the spectrum). Supernodal mode requires strict positive definiteness.
    """
    from sksparse.cholmod import cholesky

    N = L_op.shape[0]
    L_shifted = L_op - sigma * sp.eye(N, format='csc')
    L_shifted_csc = L_shifted.tocsc()

    tracemalloc.start()
    t_factor_start = time.time()
    # mode='simplicial' → LDL^T decomposition (handles indefinite)
    # mode='supernodal' → LL^T (requires SPD, faster for large SPD)
    factor = cholesky(L_shifted_csc, beta=0, mode='simplicial')
    t_factor = time.time() - t_factor_start

    def solve_shifted(b):
        return factor(b)

    OPinv = LinearOperator((N, N), matvec=solve_shifted, dtype=L_op.dtype)

    t0 = time.time()
    evals, evecs = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                         OPinv=OPinv, maxiter=5000, tol=1e-10)
    t_solve = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    idx = np.argsort(evals)
    return evals[idx], t_factor, t_solve, peak_mem


def solve_eigsh_splu(L_op, n_modes, sigma):
    """eigsh with explicit scipy splu (SuperLU) — for timing breakdown."""
    from scipy.sparse.linalg import splu

    N = L_op.shape[0]
    L_shifted = L_op - sigma * sp.eye(N, format='csc')
    L_shifted_csc = L_shifted.tocsc()

    tracemalloc.start()
    t_factor_start = time.time()
    lu = splu(L_shifted_csc)
    t_factor = time.time() - t_factor_start

    def solve_shifted(b):
        return lu.solve(b)

    OPinv = LinearOperator((N, N), matvec=solve_shifted, dtype=L_op.dtype)

    t0 = time.time()
    evals, evecs = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                         OPinv=OPinv, maxiter=5000, tol=1e-10)
    t_solve = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    idx = np.argsort(evals)
    return evals[idx], t_factor, t_solve, peak_mem


def run_benchmark():
    print("=" * 70)
    print("3-WAY BENCHMARK: MPB vs FDFD(SuperLU) vs FDFD(CHOLMOD)")
    print("Honeycomb TM monolayer, K' = (2/3, 1/3), 16 pts/cell")
    print("=" * 70)
    print()

    # ── Build monolayer geometry ──
    # (m=1, n=0) gives the primitive cell with N_cells=1, theta=60°
    Nx = 16
    eps, info = build_supercell_eps(
        'honeycomb', m=1, n=0, a=1.0,
        r_over_a=0.2, eps_rod=11.56, eps_bg=1.0,
        Nx=Nx, Ny=Nx,
    )
    N_dof = Nx * Nx
    print(f"Geometry: honeycomb, N_cells={info['N_cells']}, "
          f"Nx={Nx}, DOF={N_dof}")
    print(f"Fill fraction: {(eps > 1.5).mean():.3f}")
    print(f"B_super columns:")
    print(f"  L1 = {info['L1']}")
    print(f"  L2 = {info['L2']}")

    # ── K' point in Cartesian k-space ──
    B_mono = build_monolayer_basis('honeycomb', a=1.0)
    K_prime = get_K_prime_cartesian(B_mono)
    print(f"K' (Cartesian) = ({K_prime[0]:.6f}, {K_prime[1]:.6f})")
    print()

    # ── Build FDFD operator at K' ──
    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=K_prime, polarization='tm')
    t_build = time.time() - t0
    print(f"Operator: shape={L.shape}, nnz={L.nnz}, build={t_build:.3f}s")
    print(f"Hermiticity check: ||L - L†|| = {sp.linalg.norm(L - L.conj().T):.2e}")
    print()

    # ── Target sigma near the Dirac frequency ──
    omega_target = MPB_DIRAC_FREQ
    sigma = (2 * np.pi * omega_target) ** 2
    n_modes = 10
    print(f"Target: omega={omega_target:.6f}, sigma={sigma:.4f}")
    print(f"Requesting {n_modes} modes")
    print()

    # ── Method 1: MPB reference ──
    print("─" * 50)
    print("Method 1: MPB (reference)")
    print("─" * 50)
    print(f"  Dirac frequency at K': {MPB_DIRAC_FREQ:.6f}")
    print(f"  (from prior MPB run with high resolution)")
    print()

    # ── Method 2: FDFD + SuperLU (current, with timing breakdown) ──
    print("─" * 50)
    print("Method 2: FDFD + eigsh/SuperLU (explicit splu)")
    print("─" * 50)
    evals_su, t_factor_su, t_solve_su, mem_su = solve_eigsh_splu(L, n_modes, sigma)
    freqs_su = np.sqrt(np.maximum(evals_su, 0)) / (2 * np.pi)
    print(f"  Factorization: {t_factor_su:.4f}s")
    print(f"  Lanczos solve: {t_solve_su:.4f}s")
    print(f"  Total: {t_factor_su + t_solve_su:.4f}s")
    print(f"  Peak memory (traced): {mem_su / 1e6:.1f} MB")
    print(f"  Frequencies:")
    for i, f in enumerate(freqs_su):
        diff = abs(f - MPB_DIRAC_FREQ)
        print(f"    mode {i}: ω = {f:.6f}  (Δ from MPB = {diff:.6f})")
    print()

    # ── Method 3: FDFD + CHOLMOD ──
    print("─" * 50)
    print("Method 3: FDFD + eigsh/CHOLMOD")
    print("─" * 50)
    try:
        evals_ch, t_factor_ch, t_solve_ch, mem_ch = solve_eigsh_cholmod(
            L, n_modes, sigma)
        freqs_ch = np.sqrt(np.maximum(evals_ch, 0)) / (2 * np.pi)
        print(f"  Factorization: {t_factor_ch:.3f}s")
        print(f"  Solve (Lanczos): {t_solve_ch:.3f}s")
        print(f"  Total: {t_factor_ch + t_solve_ch:.3f}s")
        print(f"  Peak memory (traced): {mem_ch / 1e6:.1f} MB")
        print(f"  Frequencies:")
        for i, f in enumerate(freqs_ch):
            diff = abs(f - MPB_DIRAC_FREQ)
            print(f"    mode {i}: ω = {f:.6f}  (Δ from MPB = {diff:.6f})")
        print()

        # ── Comparison ──
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Method':<25} {'Factor(s)':<10} {'Solve(s)':<10} {'Total(s)':<10} {'Mem(MB)':<10} {'ω Dirac':<12} {'Δ MPB':<10}")
        print("-" * 87)
        print(f"{'MPB (reference)':<25} {'--':<10} {'--':<10} {'--':<10} {'--':<10} {MPB_DIRAC_FREQ:<12.6f} {'0':<10}")

        # Find the two frequencies closest to Dirac for each method
        dirac_idx_su = np.argsort(np.abs(freqs_su - MPB_DIRAC_FREQ))[:2]
        dirac_idx_ch = np.argsort(np.abs(freqs_ch - MPB_DIRAC_FREQ))[:2]

        for idx in sorted(dirac_idx_su):
            f = freqs_su[idx]
            d = abs(f - MPB_DIRAC_FREQ)
            print(f"{'SuperLU b' + str(idx):<25} {t_factor_su:<10.4f} {t_solve_su:<10.4f} {t_factor_su+t_solve_su:<10.4f} {mem_su/1e6:<10.1f} {f:<12.6f} {d:<10.6f}")

        for idx in sorted(dirac_idx_ch):
            f = freqs_ch[idx]
            d = abs(f - MPB_DIRAC_FREQ)
            print(f"{'CHOLMOD b' + str(idx):<25} {t_factor_ch:<10.4f} {t_solve_ch:<10.4f} {t_factor_ch+t_solve_ch:<10.4f} {mem_ch/1e6:<10.1f} {f:<12.6f} {d:<10.6f}")

        print()
        print(f"Eigenvalue agreement (SuperLU vs CHOLMOD):")
        max_diff = np.max(np.abs(evals_su - evals_ch))
        print(f"  Max |eval_SU - eval_CH| = {max_diff:.2e}")
        print(f"  All eigenvalues match: {max_diff < 1e-8}")
        print()
        t_total_su = t_factor_su + t_solve_su
        t_total_ch = t_factor_ch + t_solve_ch
        speedup = t_total_su / t_total_ch if t_total_ch > 0 else float('inf')
        print(f"Speedup (CHOLMOD vs SuperLU): {speedup:.2f}x")
        print(f"  Factor speedup: {t_factor_su / max(t_factor_ch, 1e-9):.2f}x")
        print(f"  Solve speedup:  {t_solve_su / max(t_solve_ch, 1e-9):.2f}x")
        print(f"Memory ratio (CHOLMOD / SuperLU): {mem_ch / max(mem_su, 1):.2f}x")

    except Exception as e:
        print(f"  CHOLMOD FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Falling back to SuperLU-only results.")

    # ── Also test at higher resolution for scaling ──
    print()
    print("=" * 70)
    print("SCALING TEST: 32 pts/cell and 64 pts/cell")
    print("=" * 70)

    for Nx_test in [32, 64]:
        print(f"\n--- {Nx_test} pts/cell (DOF={Nx_test**2}) ---")
        eps_t, info_t = build_supercell_eps(
            'honeycomb', m=1, n=0, a=1.0,
            r_over_a=0.2, eps_rod=11.56, eps_bg=1.0,
            Nx=Nx_test, Ny=Nx_test,
        )
        L_t = build_fdfd_operator(eps_t, info_t, q_vec=K_prime, polarization='tm')
        print(f"  nnz = {L_t.nnz}")

        evals_su_t, tf_su_t, ts_su_t, mem_su_t = solve_eigsh_splu(
            L_t, n_modes, sigma)
        freqs_su_t = np.sqrt(np.maximum(evals_su_t, 0)) / (2 * np.pi)
        dirac_f_su = freqs_su_t[np.argmin(np.abs(freqs_su_t - MPB_DIRAC_FREQ))]
        print(f"  SuperLU: factor={tf_su_t:.4f}s, solve={ts_su_t:.4f}s, "
              f"total={tf_su_t+ts_su_t:.4f}s, mem={mem_su_t/1e6:.1f}MB, "
              f"ω_Dirac={dirac_f_su:.6f} (Δ={abs(dirac_f_su-MPB_DIRAC_FREQ):.6f})")

        try:
            evals_ch_t, tf_ch_t, ts_ch_t, mem_ch_t = solve_eigsh_cholmod(
                L_t, n_modes, sigma)
            freqs_ch_t = np.sqrt(np.maximum(evals_ch_t, 0)) / (2 * np.pi)
            dirac_f_ch = freqs_ch_t[np.argmin(np.abs(freqs_ch_t - MPB_DIRAC_FREQ))]
            t_total_su_t = tf_su_t + ts_su_t
            t_total_ch_t = tf_ch_t + ts_ch_t
            speedup_t = t_total_su_t / t_total_ch_t if t_total_ch_t > 0 else float('inf')
            print(f"  CHOLMOD: factor={tf_ch_t:.4f}s, solve={ts_ch_t:.4f}s, "
                  f"total={t_total_ch_t:.4f}s, mem={mem_ch_t/1e6:.1f}MB, "
                  f"ω_Dirac={dirac_f_ch:.6f} (Δ={abs(dirac_f_ch-MPB_DIRAC_FREQ):.6f})")
            print(f"  Speedup: {speedup_t:.2f}x, |Δeval|_max={np.max(np.abs(evals_su_t - evals_ch_t)):.2e}")
        except Exception as e:
            print(f"  CHOLMOD FAILED: {e}")

    # ── Extrapolation to moiré supercell ──
    print()
    print("=" * 70)
    print("EXTRAPOLATION TO (30,29) MOIRÉ SUPERCELL (from 64-pt data)")
    print("=" * 70)
    # Use the last successful timing for extrapolation
    Nx_ref = 64
    try:
        t_ref_su = tf_su_t + ts_su_t  # from last loop iteration
        t_ref_ch = tf_ch_t + ts_ch_t
        mem_ref_su = mem_su_t
        mem_ref_ch = mem_ch_t
    except:
        t_ref_su = tf_su_t + ts_su_t
        t_ref_ch = None
        mem_ref_su = mem_su_t
        mem_ref_ch = None

    N_moire = 2611
    for res_label, Nx_m in [("res=12", int(round(np.sqrt(N_moire)*12))),
                            ("res=16", int(round(np.sqrt(N_moire)*16))),
                            ("res=20", int(round(np.sqrt(N_moire)*20)))]:
        dof_m = Nx_m * Nx_m
        scale = (Nx_m / Nx_ref) ** 3  # LU/Cholesky scales as ~Nx^3
        t_est_su = t_ref_su * scale
        mem_scale = (Nx_m / Nx_ref) ** 2.5  # memory scales as ~Nx^2.5
        mem_est_su = mem_ref_su * mem_scale
        print(f"{res_label}: Nx={Nx_m}, DOF={dof_m:,}")
        print(f"  SuperLU est.: {t_est_su:.0f}s ({t_est_su/60:.1f} min), {mem_est_su/1e9:.1f} GB")
        if t_ref_ch is not None:
            t_est_ch = t_ref_ch * scale
            mem_est_ch = mem_ref_ch * mem_scale
            print(f"  CHOLMOD est.: {t_est_ch:.0f}s ({t_est_ch/60:.1f} min), {mem_est_ch/1e9:.1f} GB")


if __name__ == '__main__':
    run_benchmark()
