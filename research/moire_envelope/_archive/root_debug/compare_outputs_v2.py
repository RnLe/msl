import numpy as np
import matplotlib.pyplot as plt
import h5py

# Load all three outputs
blaze_path = 'runs/phase0_blaze_20251201_140949/candidate_0000/phase1_band_data.h5'
mpb_fixed_path = 'runs/phase0_blaze_20251201_140949_mpb_fixed/candidate_0000/phase1_band_data.h5'

with h5py.File(blaze_path, 'r') as hf:
    V_blaze = hf['V'][:]
    R_blaze = hf['R_grid'][:]
    print("BLAZE R_grid shape:", R_blaze.shape)
    print("BLAZE x range:", R_blaze[:,:,0].min(), R_blaze[:,:,0].max())
    print("BLAZE y range:", R_blaze[:,:,1].min(), R_blaze[:,:,1].max())

with h5py.File(mpb_fixed_path, 'r') as hf:
    V_mpb_fixed = hf['V'][:]
    R_mpb_fixed = hf['R_grid'][:]
    print("\nMPB FIXED R_grid shape:", R_mpb_fixed.shape)
    print("MPB FIXED x range:", R_mpb_fixed[:,:,0].min(), R_mpb_fixed[:,:,0].max())
    print("MPB FIXED y range:", R_mpb_fixed[:,:,1].min(), R_mpb_fixed[:,:,1].max())

# Check the actual grid structure
print("\nBLAZE R_grid[0,0,:]:", R_blaze[0,0,:])
print("BLAZE R_grid[-1,-1,:]:", R_blaze[-1,-1,:])
print("BLAZE R_grid[0,:,1] (first few y values along first x):", R_blaze[0,:5,1])

print("\nMPB FIXED R_grid[0,0,:]:", R_mpb_fixed[0,0,:])
print("MPB FIXED R_grid[-1,-1,:]:", R_mpb_fixed[-1,-1,:])
print("MPB FIXED R_grid[0,:,1] (first few y values along first x):", R_mpb_fixed[0,:5,1])

# Create comparison figure - no interpolation, just raw contours
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

vmin = min(V_blaze.min(), V_mpb_fixed.min())
vmax = max(V_blaze.max(), V_mpb_fixed.max())
contour_levels = np.linspace(vmin, vmax, 15)

# Row 1: Original data
ax1, ax2, ax3 = axes[0]

cs1 = ax1.contourf(R_blaze[:,:,0], R_blaze[:,:,1], V_blaze, levels=contour_levels, cmap='RdBu_r')
ax1.set_title('BLAZE (original)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
plt.colorbar(cs1, ax=ax1)

cs2 = ax2.contourf(R_mpb_fixed[:,:,0], R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, cmap='RdBu_r')
ax2.set_title('MPB FIXED (original)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')
plt.colorbar(cs2, ax=ax2)

# Try flipping y: use -y for MPB
cs3 = ax3.contourf(R_mpb_fixed[:,:,0], -R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, cmap='RdBu_r')
ax3.set_title('MPB FIXED (y flipped)')
ax3.set_xlabel('x')
ax3.set_ylabel('-y')
ax3.set_aspect('equal')
plt.colorbar(cs3, ax=ax3)

# Row 2: Try other transformations
ax4, ax5, ax6 = axes[1]

# Try flipping x
cs4 = ax4.contourf(-R_mpb_fixed[:,:,0], R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, cmap='RdBu_r')
ax4.set_title('MPB FIXED (x flipped)')
ax4.set_xlabel('-x')
ax4.set_ylabel('y')
ax4.set_aspect('equal')
plt.colorbar(cs4, ax=ax4)

# Try flipping both
cs5 = ax5.contourf(-R_mpb_fixed[:,:,0], -R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, cmap='RdBu_r')
ax5.set_title('MPB FIXED (x,y flipped)')
ax5.set_xlabel('-x')
ax5.set_ylabel('-y')
ax5.set_aspect('equal')
plt.colorbar(cs5, ax=ax5)

# Overlay comparison: BLAZE contours vs MPB y-flipped contours
ax6.contour(R_blaze[:,:,0], R_blaze[:,:,1], V_blaze, levels=contour_levels, colors='blue', linewidths=1, linestyles='solid')
ax6.contour(R_mpb_fixed[:,:,0], -R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, colors='red', linewidths=1, linestyles='dashed')
ax6.set_title('Overlay: BLAZE (blue) vs MPB y-flipped (red dashed)')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('runs/phase0_blaze_20251201_140949/comparison_mirror_test.png', dpi=150)
print('\nSaved to runs/phase0_blaze_20251201_140949/comparison_mirror_test.png')

# Quantitative test with y-flip
from scipy.interpolate import RegularGridInterpolator

x_mpb = R_mpb_fixed[:, 0, 0]
y_mpb = R_mpb_fixed[0, :, 1]

# Create interpolator with flipped y (need to flip y array and V accordingly)
y_mpb_flipped = -y_mpb[::-1]  # Reverse and negate
V_mpb_flipped = V_mpb_fixed[:, ::-1]  # Reverse along y axis

interp_mpb_yflip = RegularGridInterpolator((x_mpb, y_mpb_flipped), V_mpb_flipped, bounds_error=False, fill_value=np.nan)

query = np.stack([R_blaze[:,:,0].ravel(), R_blaze[:,:,1].ravel()], axis=-1)
V_mpb_yflip_interp = interp_mpb_yflip(query).reshape(V_blaze.shape)

valid = ~np.isnan(V_mpb_yflip_interp)
rmse = np.sqrt(np.mean((V_blaze[valid] - V_mpb_yflip_interp[valid])**2))
corr = np.corrcoef(V_blaze[valid].ravel(), V_mpb_yflip_interp[valid].ravel())[0,1]

print(f'\nWith y-flip: RMSE = {rmse:.5f}, Correlation = {corr:.4f}')
