"""
Compute moiré band structure along Γ_m → M_m → K_m → Γ_m.

Uses the (8,7) commensurate angle (θ=4.41°, N=169) with 20 pts/cell
resolution and shift-invert targeting the Dirac manifold.
"""
import numpy as np
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps, build_moire_bz_path
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from scipy.sparse.linalg import eigsh

# ── Parameters ──
m, n = 8, 7
EPS_BG, EPS_HOLE, R_OVER_A = 1.0, 11.56, 0.2
omega_ref = 0.182693
sigma = (2 * np.pi * omega_ref)**2
res = 20          # pts per monolayer lattice constant
n_modes = 30     # modes near omega_ref
n_per_seg = 8    # k-points per BZ segment (3 segments → 22 total with endpoints)

N_cells = m*m + m*n + n*n
theta = np.degrees(commensurate_twist_angle('honeycomb', m, n))
Nx = int(round(np.sqrt(N_cells) * res))

print(f"Moiré band structure: (m,n)=({m},{n}), θ={theta:.3f}°, N_cells={N_cells}")
print(f"Resolution: {res} pts/cell, Nx={Nx}, DOF={Nx*Nx:,}")
print(f"Targeting σ={sigma:.4f} (ω_ref={omega_ref}), {n_modes} modes")
print()

# ── Build supercell ──
t0 = time.time()
eps, info = build_supercell_eps('honeycomb', m=m, n=n, a=1.0,
                                r_over_a=R_OVER_A, eps_rod=EPS_HOLE,
                                eps_bg=EPS_BG, Nx=Nx, Ny=Nx)
print(f"Supercell built: {time.time()-t0:.1f}s, fill={np.mean(eps > 1.5):.3f}")

# ── BZ path ──
q_path, q_dist, labels = build_moire_bz_path(info['B_super'], 'honeycomb',
                                               n_points=n_per_seg)
n_q = len(q_path)
print(f"BZ path: {n_q} k-points, segments: " +
      " → ".join(f"{lbl}" for _, lbl in labels))
print()

# ── Solve at each k-point ──
all_evals = np.zeros((n_q, n_modes))
all_freqs = np.zeros((n_q, n_modes))
t_total = time.time()

for iq in range(n_q):
    q = q_path[iq]
    # Check if this is a labeled point
    label_str = ""
    for idx, lbl in labels:
        if idx == iq:
            label_str = f" ({lbl})"
            break

    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=q, polarization='tm')
    evals, evecs = eigsh(L, k=n_modes, sigma=sigma, which='LM',
                         maxiter=5000, tol=1e-8)
    t_solve = time.time() - t0

    idx_sort = np.argsort(evals)
    evals = evals[idx_sort]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

    all_evals[iq] = evals
    all_freqs[iq] = freqs

    print(f"  q[{iq:>2}/{n_q}]{label_str}: "
          f"|q|={np.linalg.norm(q):.4f}, "
          f"f=[{freqs[0]:.6f}, {freqs[-1]:.6f}], "
          f"{t_solve:.1f}s")

t_total = time.time() - t_total
print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")

# ── Save results ──
out_dir = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(out_dir, f"moire_bands_m{m}_n{n}_res{res}.npz")
np.savez(out_file,
    freqs=all_freqs,
    evals=all_evals,
    q_path=q_path,
    q_dist=q_dist,
    labels=np.array([(idx, lbl) for idx, lbl in labels], dtype=object),
    omega_ref=omega_ref,
    theta_deg=theta,
    m=m, n=n, N_cells=N_cells,
    res=res, Nx=Nx, n_modes=n_modes,
)
print(f"Saved to {out_file}")

# ── Quick summary ──
print(f"\n{'='*60}")
print("BAND STRUCTURE SUMMARY")
print(f"{'='*60}")
# Identify clusters: modes within 2% of omega_ref
mask = np.abs(all_freqs - omega_ref) < 0.02 * omega_ref
f_near = all_freqs[mask] if mask.any() else all_freqs.ravel()
print(f"Frequency range: [{all_freqs.min():.6f}, {all_freqs.max():.6f}]")
print(f"Modes within 2% of ω_ref: {mask.sum()} out of {all_freqs.size}")
if mask.any():
    print(f"  Bandwidth of near-Dirac modes: {f_near.max() - f_near.min():.6f}")

# Print band extrema for first 30 modes
print(f"\nBand extrema (min, max) for each mode:")
for i in range(n_modes):
    bmin, bmax = all_freqs[:, i].min(), all_freqs[:, i].max()
    bw = bmax - bmin
    print(f"  band {i:>2}: [{bmin:.6f}, {bmax:.6f}]  BW={bw:.6f}")
