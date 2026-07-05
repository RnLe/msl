#!/usr/bin/env python
"""Compute FDFD resolution per rod for various supercells."""
import math

a = 1.0
r_over_a = 0.2
r_rod = r_over_a * a   # 0.2a
rod_diameter = 2 * r_rod  # 0.4a

print("=== Rod geometry ===")
print(f"  a = {a}, r/a = {r_over_a}, rod diameter = {rod_diameter}a")
print()

# What does res=N mean in our FDFD?
# Need to check the actual code. Two possible conventions:
#  A) res = pixels per monolayer lattice constant a  →  Nx = res * |C1|
#  B) res = Nx (total grid points per supercell side) →  pix/a = res / |C1|

for m, n in [(4, 3), (30, 29)]:
    N_cells = m*m + m*n + n*n
    C1_len = math.sqrt(N_cells) * a
    n_rods_bilayer = 4 * N_cells   # 2 layers × 2 sublattices
    theta = math.degrees(2 * math.atan(math.sqrt(3) * n / (2*m + n)))

    print(f"=== ({m},{n}) supercell  θ = {theta:.4f}° ===")
    print(f"  N_cells = {N_cells}, |C1| = {C1_len:.3f}a")
    print(f"  n_rods (bilayer) = {n_rods_bilayer}")
    print()

    print(f"  {'res':>5s}  {'Interp':>10s}  {'Nx':>6s}  {'pix/a':>7s}  {'pix/diam':>9s}  {'pix²/rod':>10s}")
    for res in [16, 32, 40, 64, 128]:
        # Convention A: res = pix/a
        Nx_A = int(round(res * C1_len))
        ppd_A = res * rod_diameter
        ppa_A = math.pi * (res * r_rod)**2

        # Convention B: res = Nx total
        ppa_B = res / C1_len
        ppd_B = ppa_B * rod_diameter
        pparea_B = math.pi * (ppa_B * r_rod)**2

        print(f"  {res:5d}  {'pix/a':>10s}  {Nx_A:6d}  {res:7.1f}  {ppd_A:9.1f}  {ppa_A:10.1f}")
        print(f"  {'':5s}  {'Nx total':>10s}  {res:6d}  {ppa_B:7.2f}  {ppd_B:9.2f}  {pparea_B:10.2f}")
    print()

# Moiré scale context
theta_11 = 1.1
L_moire = a / (2 * math.sin(math.radians(theta_11) / 2))
print(f"=== Moiré context at θ = {theta_11}° ===")
print(f"  L_moiré ≈ {L_moire:.1f}a")
print(f"  (30,29): |C1| = {math.sqrt(30**2+30*29+29**2):.2f}a, θ = {math.degrees(2*math.atan(math.sqrt(3)*29/(2*30+29))):.4f}°")
