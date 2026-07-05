#!/usr/bin/env python
"""
Lightweight planning script for square-lattice FDFD-vs-EA validation.

1. Find commensurate angles near 2° for square lattice
2. Compute M-point TM band structure for square lattice (r/a, eps choices)
3. Identify a good candidate: band extremum at M with spectral gaps
4. Estimate grid sizes and RAM for FDFD

NO heavy computation — just parameter exploration.
"""
import numpy as np
import math
import sys, os

# ── 1. Commensurate angles near 2° for square lattice ──
# For square: θ = 2·arctan(n/m), coprime (m,n), m > n > 0
# N_cells = m² + n²

print("=" * 70)
print("  COMMENSURATE ANGLES NEAR 2° (SQUARE LATTICE)")
print("=" * 70)
print(f"  {'(m,n)':>8s}  {'θ (deg)':>10s}  {'N_cells':>8s}  {'|C1|/a':>8s}  {'n_rods':>8s}")

candidates = []
for m in range(1, 200):
    for n in range(1, m):
        if math.gcd(m, n) != 1:
            continue
        theta = 2 * math.atan(n / m)
        theta_deg = math.degrees(theta)
        if 1.0 < theta_deg < 4.0:
            N = m*m + n*n
            C1_len = math.sqrt(N)
            n_rods = N  # 1 sublattice × 2 layers = 2N, but for single layer = N
            candidates.append((m, n, theta_deg, N, C1_len))

# Sort by closeness to 2.0°
candidates.sort(key=lambda x: abs(x[2] - 2.0))
for m, n, theta_deg, N, C1_len in candidates[:20]:
    n_rods_bilayer = 2 * N  # 2 layers, 1 sublattice each
    print(f"  ({m},{n}){'':<4s}  {theta_deg:10.4f}  {N:8d}  {C1_len:8.2f}  {n_rods_bilayer:8d}")

print()
# Pick the best few for detailed analysis
best = candidates[:5]
print("Top 5 closest to 2°:")
for m, n, theta_deg, N, C1_len in best:
    print(f"  ({m},{n}): θ={theta_deg:.4f}°, N_cells={N}, |C1|={C1_len:.2f}a")

    # Resolution analysis
    rod_diam = 0.4  # 2 * r/a for r/a=0.2 (in units of a)
    for res in [32, 40, 64, 80, 128]:
        Nx = int(round(C1_len * res))
        dof = Nx * Nx
        ram_gb = dof * 16 * 100 / 1e9  # rough: 100 non-zeros per row, 16 bytes each
        pix_per_rod = res * rod_diam
        print(f"    res={res:4d}: Nx={Nx:6d}, DOF={dof:12,}, ~{ram_gb:.1f} GB, "
              f"{pix_per_rod:.0f} pix/rod_diam")
    print()

# ── 2. Square lattice M-point band structure (quick MPB) ──
print("=" * 70)
print("  SQUARE LATTICE M-POINT BAND STRUCTURE")
print("  (Will run lightweight MPB at low resolution to find candidates)")
print("=" * 70)

# We'll compute this with MPB at res=32 (lightweight)
try:
    import meep as mp
    from meep import mpb

    a = 1.0
    # Try a few (r/a, eps) combos
    combos = [
        (0.2, 11.56, 1.0),   # standard: dielectric rods in air
        (0.2, 8.9, 1.0),     # slightly lower contrast
        (0.3, 11.56, 1.0),   # bigger rods
        (0.15, 11.56, 1.8),  # like existing square_M_b3 candidate
    ]

    N_BANDS = 20

    for r_over_a, eps_rod, eps_bg in combos:
        print(f"\n--- r/a={r_over_a}, ε_rod={eps_rod}, ε_bg={eps_bg} ---")

        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))  # square: default basis
        geometry = [
            mp.Cylinder(
                radius=r_over_a,
                center=mp.Vector3(0, 0, 0),
                material=mp.Medium(epsilon=eps_rod),
            )
        ]
        ms = mpb.ModeSolver(
            geometry=geometry,
            geometry_lattice=lattice,
            default_material=mp.Medium(epsilon=eps_bg),
            num_bands=N_BANDS,
            resolution=32,
            k_points=[mp.Vector3(0.5, 0.5, 0)],  # M point
        )

        # Suppress output
        mp.verbosity(0)
        fd = os.open(os.devnull, os.O_WRONLY)
        o1, o2 = os.dup(1), os.dup(2)
        os.dup2(fd, 1); os.dup2(fd, 2)
        ms.run_tm()
        os.dup2(o1, 1); os.dup2(o2, 2)
        os.close(fd); os.close(o1); os.close(o2)

        freqs = np.array(ms.all_freqs[0])
        print(f"  TM bands at M: {freqs}")

        # Find gaps
        print(f"  {'band':>5s}  {'ω':>10s}  {'gap_below':>10s}  {'gap_above':>10s}  {'is_extremum':>12s}")
        for i in range(N_BANDS):
            gap_below = freqs[i] - freqs[i-1] if i > 0 else 0
            gap_above = freqs[i+1] - freqs[i] if i < N_BANDS-1 else 0
            is_good = "✓ GOOD" if (gap_below > 0.01 and gap_above > 0.01) else ""
            print(f"  {i:5d}  {freqs[i]:10.6f}  {gap_below:10.6f}  {gap_above:10.6f}  {is_good:>12s}")

except ImportError:
    print("  MPB not available — skipping band structure")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ── 3. RAM estimate for FDFD ──
print("\n" + "=" * 70)
print("  RAM ESTIMATE FOR FDFD EIGENSOLVE")
print("=" * 70)
print("  System: 64 GB total (assumed)")
print("  Sparse matrix: ~5 non-zeros per row (5-point stencil)")
print("  ARPACK needs ~10-20 Lanczos vectors × DOF × 16 bytes")
print()

for label, N_cells in [("(29,1) θ≈1.97°", 842), ("(23,1) θ≈2.49°", 530)]:
    C1_len = math.sqrt(N_cells)
    print(f"  {label}: N_cells={N_cells}, |C1|={C1_len:.2f}a")
    for res in [32, 40, 64, 80, 128]:
        Nx = int(round(C1_len * res))
        dof = Nx * Nx
        # Sparse matrix: CSR, ~5 entries/row → 5*DOF*(8+4) bytes ≈ 60*DOF bytes
        sparse_bytes = 5 * dof * 12
        # ARPACK: k=200 eigenvalues, ~20 Lanczos vectors → 220*DOF*16 bytes
        arpack_bytes = 220 * dof * 16
        total_gb = (sparse_bytes + arpack_bytes) / 1e9
        print(f"    res={res:4d}: Nx={Nx:6d}, DOF={dof:12,}, est. RAM ≈ {total_gb:.1f} GB")
    print()
