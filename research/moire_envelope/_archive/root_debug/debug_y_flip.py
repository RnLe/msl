"""Test to understand the y-flip issue between BLAZE and MPB"""
import numpy as np
import math

# The registry formula: delta_physical = (R(θ) - I) @ R / η + τ

# For a simple test case at R = (0, 1) with θ = 5°

theta_deg = 5.0
theta_rad = math.radians(theta_deg)
eta = 1.0
tau = np.array([0.0, 0.0])

R_vec = np.array([0.0, 1.0])

c = np.cos(theta_rad)
s = np.sin(theta_rad)
R_rot = np.array([[c, -s], [s, c]])
I = np.eye(2)

# Physical shift
delta_physical = (R_rot - I) @ R_vec / eta + tau
print(f"R = {R_vec}")
print(f"θ = {theta_deg}°")
print(f"(R(θ) - I) @ R = {(R_rot - I) @ R_vec}")
print(f"delta_physical = {delta_physical}")

# The rotation R(θ) rotates counterclockwise.
# For R = (0, 1), after rotation by +5°, we get approximately (−0.087, 0.996)
# So (R(θ) - I) @ R = (−0.087, −0.004)
# This means for positive y, the delta has a NEGATIVE x component.

# Now check both conventions:
# MPB 60°: a1 = [1, 0], a2 = [0.5, sqrt(3)/2]
# BLAZE 120°: a1 = [1, 0], a2 = [-0.5, sqrt(3)/2]

a1_mpb = np.array([1.0, 0.0])
a2_mpb = np.array([0.5, math.sqrt(3)/2])

a1_blaze = np.array([1.0, 0.0])
a2_blaze = np.array([-0.5, math.sqrt(3)/2])

lattice_mat_mpb = np.column_stack([a1_mpb, a2_mpb])
lattice_mat_blaze = np.column_stack([a1_blaze, a2_blaze])

delta_frac_mpb = np.linalg.inv(lattice_mat_mpb) @ delta_physical
delta_frac_blaze = np.linalg.inv(lattice_mat_blaze) @ delta_physical

print(f"\nMPB 60° lattice matrix:\n{lattice_mat_mpb}")
print(f"delta_frac (MPB convention) = {delta_frac_mpb}")

print(f"\nBLAZE 120° lattice matrix:\n{lattice_mat_blaze}")
print(f"delta_frac (BLAZE convention) = {delta_frac_blaze}")

# The transform T = [[1,1],[0,1]] should convert MPB frac to BLAZE frac
T = np.array([[1, 1], [0, 1]])
delta_frac_converted = T @ delta_frac_mpb
print(f"\nT @ delta_frac_mpb = {delta_frac_converted}")
print(f"Matches BLAZE? {np.allclose(delta_frac_converted, delta_frac_blaze)}")

# Now check the opposite: R = (0, -1)
R_vec_neg = np.array([0.0, -1.0])
delta_physical_neg = (R_rot - I) @ R_vec_neg / eta + tau
print(f"\n---")
print(f"R = {R_vec_neg}")
print(f"delta_physical = {delta_physical_neg}")
delta_frac_mpb_neg = np.linalg.inv(lattice_mat_mpb) @ delta_physical_neg
print(f"delta_frac (MPB) = {delta_frac_mpb_neg}")

# Key: if we flip y in R_vec, does delta_y flip sign?
print(f"\nSymmetry check:")
print(f"R = (0,+1) -> delta_frac = {delta_frac_mpb}")
print(f"R = (0,-1) -> delta_frac = {delta_frac_mpb_neg}")
print(f"Note: delta_y flips sign but delta_x does too!")

# Now the central question: is compute_registry_map using the correct lattice convention?
# It uses a1, a2 from create_twisted_bilayer, which is:
#   a2 = [a * 0.5, a * sqrt(3) / 2, 0.0]  <- MPB 60° convention
print("\n=== KEY INSIGHT ===")
print("compute_registry_map uses MPB 60° convention for the lattice matrix.")
print("So delta_grid is in MPB fractional coordinates.")
print("The transform T = [[1,1],[0,1]] converts this to BLAZE coordinates.")
print("This should be correct...")

# Wait, let me check the specific issue with the plot.
# The issue is that MPB fixed and BLAZE are y-mirrored.
# Let me think about where this could come from...

# Hypothesis: Maybe the sign of θ is different?
print("\n=== Testing θ sign ===")
theta_neg = -theta_rad
c_neg = np.cos(theta_neg)
s_neg = np.sin(theta_neg)
R_rot_neg = np.array([[c_neg, -s_neg], [s_neg, c_neg]])
delta_physical_theta_neg = (R_rot_neg - I) @ R_vec / eta + tau
delta_frac_theta_neg = np.linalg.inv(lattice_mat_mpb) @ delta_physical_theta_neg
print(f"With θ = +5°: delta_frac = {delta_frac_mpb}")
print(f"With θ = -5°: delta_frac = {delta_frac_theta_neg}")

# Check if flipping y in R is equivalent to flipping θ sign
print(f"\nR=(0,+1), θ=+5°: {delta_frac_mpb}")
print(f"R=(0,-1), θ=+5°: {delta_frac_mpb_neg}")
print(f"R=(0,+1), θ=-5°: {delta_frac_theta_neg}")
print("Flipping y in R is NOT the same as flipping θ!")

# Actually the correct relationship:
# Under y → -y: R_vec → [1, 0; 0, -1] @ R_vec
# Under θ → -θ: R(θ) → R(-θ) = R^T(θ)
# These are NOT equivalent operations.
