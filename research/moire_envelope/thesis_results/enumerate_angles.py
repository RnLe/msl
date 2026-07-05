import numpy as np
from math import gcd

results = []
for m in range(1, 40):
    for n in range(1, m):
        if gcd(m, n) != 1:
            continue
        theta = 2 * np.degrees(np.arctan2(n, m))
        N_cells = m**2 + n**2
        L_super = np.sqrt(N_cells)
        grid32 = 32 * round(L_super)
        grid64 = 64 * round(L_super)
        dof32 = grid32**2
        dof64 = grid64**2
        if theta < 25:
            results.append((theta, m, n, N_cells, L_super, grid32, grid64, dof32, dof64))

results.sort(key=lambda x: x[0])
print(f"{'(m,n)':>10s}  {'θ':>8s}  {'N_cells':>8s}  {'L_sup':>7s}  {'grid32':>7s}  {'grid64':>7s}  {'DOF@32':>12s}  {'DOF@64':>12s}  {'RAM@64(GB)':>10s}")
print('-' * 100)
for theta, m, n, N, L, g32, g64, d32, d64 in results:
    # RAM estimate: eigsh needs ~10 * DOF * 16 bytes (complex128) for Lanczos + sparse LU
    # Sparse LU is the dominant factor: ~50-100 * nnz * 16 bytes for fill-in
    # nnz ~ 7 * DOF, fill-in factor ~10-30x for 2D Laplacian
    # Conservative: ~200 * DOF * 8 bytes for total RAM
    ram_gb_64 = d64 * 200 * 8 / 1e9
    ram_gb_32 = d32 * 200 * 8 / 1e9
    marker = ''
    if abs(theta - 3) < 1: marker = ' ***3'
    elif abs(theta - 5) < 1: marker = ' ***5'
    elif abs(theta - 8) < 1: marker = ' ***8'
    elif abs(theta - 10) < 0.5: marker = ' **10'
    elif abs(theta - 15) < 1: marker = ' **15'
    print(f"({m:2d},{n:2d})  {theta:7.2f}°  {N:8d}  {L:7.2f}  {g32:7d}  {g64:7d}  {d32:12,}  {d64:12,}  {ram_gb_64:9.1f}{marker}")
