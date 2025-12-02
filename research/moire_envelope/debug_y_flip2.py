"""Check the actual ordering of BLAZE registry grid"""
import numpy as np
import h5py

# Load the BLAZE data
blaze_path = 'runs/phase0_blaze_20251201_140949/candidate_0000/phase1_band_data.h5'
mpb_path = 'runs/phase0_blaze_20251201_140949_mpb_fixed/candidate_0000/phase1_band_data.h5'

with h5py.File(blaze_path, 'r') as hf:
    V_blaze = hf['V'][:]
    R_blaze = hf['R_grid'][:]
    delta_blaze = hf['delta_grid'][:]

with h5py.File(mpb_path, 'r') as hf:
    V_mpb = hf['V'][:]
    R_mpb = hf['R_grid'][:]
    delta_mpb = hf['delta_grid'][:]

# Check delta_grid - these should be identical since they come from the same formula
print("Delta grid comparison:")
print(f"BLAZE delta_grid shape: {delta_blaze.shape}")
print(f"MPB delta_grid shape: {delta_mpb.shape}")
print(f"Delta grids match: {np.allclose(delta_blaze, delta_mpb)}")

# Check specific points
print("\nSample points:")
print(f"R[0,32] = {R_blaze[0, 32]}")  # y = 0
print(f"R[32,0] = {R_blaze[32, 0]}")  # x = 0
print(f"R[32,32] = {R_blaze[32, 32]}")  # center

print(f"\ndelta_blaze[0,32] = {delta_blaze[0, 32]}")
print(f"delta_blaze[32,0] = {delta_blaze[32, 0]}")
print(f"delta_blaze[32,32] = {delta_blaze[32, 32]}")

print(f"\nV_blaze[32,32] (center) = {V_blaze[32, 32]:.5f}")
print(f"V_mpb[32,32] (center) = {V_mpb[32, 32]:.5f}")

# Check symmetry
print("\nSymmetry check:")
print(f"V_blaze[0, 32] = {V_blaze[0, 32]:.5f}")
print(f"V_blaze[63, 32] = {V_blaze[63, 32]:.5f}")
print(f"V_blaze[32, 0] = {V_blaze[32, 0]:.5f}")
print(f"V_blaze[32, 63] = {V_blaze[32, 63]:.5f}")

print(f"\nV_mpb[0, 32] = {V_mpb[0, 32]:.5f}")
print(f"V_mpb[63, 32] = {V_mpb[63, 32]:.5f}")
print(f"V_mpb[32, 0] = {V_mpb[32, 0]:.5f}")
print(f"V_mpb[32, 63] = {V_mpb[32, 63]:.5f}")

# Test y-flip relationship
print("\n=== Y-flip test ===")
# If MPB and BLAZE are y-flipped, then V_blaze[i, j] ≈ V_mpb[i, 63-j]
for i in [0, 16, 32, 48, 63]:
    for j in [0, 16, 32, 48, 63]:
        v_blaze = V_blaze[i, j]
        v_mpb_same = V_mpb[i, j]
        v_mpb_yflip = V_mpb[i, 63-j]
        print(f"[{i:2d},{j:2d}]: BLAZE={v_blaze:+.4f}, MPB_same={v_mpb_same:+.4f} (diff={abs(v_blaze-v_mpb_same):.4f}), MPB_yflip={v_mpb_yflip:+.4f} (diff={abs(v_blaze-v_mpb_yflip):.4f})")
