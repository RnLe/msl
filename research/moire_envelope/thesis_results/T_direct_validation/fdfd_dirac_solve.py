"""
FDFD solve targeting the Dirac manifold (K' frequency window) for comparison
with the envelope approximation.

The envelope pipeline targets k0 = K' = (2/3, 1/3) where bands 1-2 form a
Dirac cone at omega_D ≈ 0.274. The envelope eigenvalues (physical frequencies)
span [0.221, 0.259] at theta=4.408°.

We run the FDFD at Gamma of the moiré BZ, targeting sigma near omega_D.
"""
import numpy as np
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from scipy.sparse.linalg import eigsh

# Parameters
m, n = 8, 7
EPS_BG, EPS_HOLE, R_OVER_A = 1.0, 11.56, 0.2
res = 20

# The envelope approximation center
omega_env_center = 0.234  # omega_ref from envelope
omega_D = 0.274409        # Dirac frequency at K'

N_cells = m*m + m*n + n*n
theta = np.degrees(commensurate_twist_angle('honeycomb', m, n))
Nx = int(round(np.sqrt(N_cells) * res))

print(f"FDFD Dirac manifold solve: (m,n)=({m},{n}), θ={theta:.3f}°, N_cells={N_cells}")
print(f"Resolution: {res} pts/cell, Nx={Nx}, DOF={Nx*Nx:,}")
print()

# Build supercell
eps, info = build_supercell_eps('honeycomb', m=m, n=n, a=1.0,
                                r_over_a=R_OVER_A, eps_rod=EPS_HOLE,
                                eps_bg=EPS_BG, Nx=Nx, Ny=Nx)
print(f"Supercell built, fill={np.mean(eps > 1.5):.3f}")

# We need to find enough modes to cover the envelope frequency range [0.221, 0.259]
# Target sigma near the envelope center (~0.234)
sigma_center = (2 * np.pi * omega_env_center)**2
n_modes = 80  # need many modes to span the wide envelope window

print(f"Targeting sigma = {sigma_center:.4f} (ω = {omega_env_center})")
print(f"Requesting {n_modes} modes")
print()

# Solve at Gamma
t0 = time.time()
L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
print(f"Operator built: {time.time()-t0:.1f}s, DOF={L.shape[0]:,}")

t0 = time.time()
evals, evecs = eigsh(L, k=n_modes, sigma=sigma_center, which='LM',
                     maxiter=5000, tol=1e-8)
t_solve = time.time() - t0
print(f"Eigensolver: {t_solve:.1f}s for {n_modes} modes")

idx = np.argsort(evals)
evals = evals[idx]
freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

print(f"\nFDFD frequencies at Gamma (moiré BZ):")
for i, f in enumerate(freqs):
    in_env = "  <-- in envelope window" if 0.218 < f < 0.262 else ""
    print(f"  mode {i:>3}: f = {f:.6f}{in_env}")

# Report modes in envelope window
mask_env = (freqs > 0.218) & (freqs < 0.262)
n_in_env = mask_env.sum()
print(f"\nModes in envelope window [0.218, 0.262]: {n_in_env}")
if n_in_env > 0:
    f_env = freqs[mask_env]
    print(f"  Freq range: [{f_env.min():.6f}, {f_env.max():.6f}]")
    print(f"  Bandwidth: {f_env.max()-f_env.min():.6f}")

# Save results
out_dir = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(out_dir, f"fdfd_dirac_m{m}_n{n}_res{res}.npz")
np.savez(out_file,
    freqs=freqs,
    evals=evals,
    m=m, n=n, N_cells=N_cells,
    res=res, Nx=Nx, n_modes=n_modes,
    omega_env_center=omega_env_center,
    omega_D=omega_D,
    theta_deg=theta,
)
print(f"\nSaved to {out_file}")
