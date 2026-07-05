import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import RegularGridInterpolator

# Load all three outputs
blaze_path = 'runs/phase0_blaze_20251201_140949/candidate_0000/phase1_band_data.h5'
mpb_fixed_path = 'runs/phase0_blaze_20251201_140949_mpb_fixed/candidate_0000/phase1_band_data.h5'
mpb_old_path = 'runs/phase0_real_run_20251120_151835/candidate_1737/phase1_band_data.h5'

with h5py.File(blaze_path, 'r') as hf:
    V_blaze = hf['V'][:]
    R_blaze = hf['R_grid'][:]

with h5py.File(mpb_fixed_path, 'r') as hf:
    V_mpb_fixed = hf['V'][:]
    R_mpb_fixed = hf['R_grid'][:]

with h5py.File(mpb_old_path, 'r') as hf:
    V_mpb_old = hf['V'][:]
    R_mpb_old = hf['R_grid'][:]

print('Comparison of V(R) outputs:')
print()
print('BLAZE (with hex transform):')
print(f'  V range: [{V_blaze.min():.5f}, {V_blaze.max():.5f}]')

print('\nMPB FIXED (lattice coords):')  
print(f'  V range: [{V_mpb_fixed.min():.5f}, {V_mpb_fixed.max():.5f}]')

print('\nMPB OLD (buggy - Cartesian as center):')  
print(f'  V range: [{V_mpb_old.min():.5f}, {V_mpb_old.max():.5f}]')

# Interpolate all to same grid for comparison
x_mpb_fixed = R_mpb_fixed[:, 0, 0]
y_mpb_fixed = R_mpb_fixed[0, :, 1]
x_mpb_old = R_mpb_old[:, 0, 0]
y_mpb_old = R_mpb_old[0, :, 1]

interp_mpb_fixed = RegularGridInterpolator((x_mpb_fixed, y_mpb_fixed), V_mpb_fixed, bounds_error=False, fill_value=np.nan)
interp_mpb_old = RegularGridInterpolator((x_mpb_old, y_mpb_old), V_mpb_old, bounds_error=False, fill_value=np.nan)

query = np.stack([R_blaze[:,:,0].ravel(), R_blaze[:,:,1].ravel()], axis=-1)
V_mpb_fixed_interp = interp_mpb_fixed(query).reshape(V_blaze.shape)
V_mpb_old_interp = interp_mpb_old(query).reshape(V_blaze.shape)

# Compute statistics
valid = ~np.isnan(V_mpb_fixed_interp) & ~np.isnan(V_mpb_old_interp)

rmse_blaze_mpb_fixed = np.sqrt(np.mean((V_blaze[valid] - V_mpb_fixed_interp[valid])**2))
rmse_blaze_mpb_old = np.sqrt(np.mean((V_blaze[valid] - V_mpb_old_interp[valid])**2))
corr_blaze_mpb_fixed = np.corrcoef(V_blaze[valid].ravel(), V_mpb_fixed_interp[valid].ravel())[0,1]
corr_blaze_mpb_old = np.corrcoef(V_blaze[valid].ravel(), V_mpb_old_interp[valid].ravel())[0,1]

print()
print('=== Quantitative comparison ===')
print(f'BLAZE vs MPB_FIXED:  RMSE = {rmse_blaze_mpb_fixed:.5f}, Correlation = {corr_blaze_mpb_fixed:.4f}')
print(f'BLAZE vs MPB_OLD:    RMSE = {rmse_blaze_mpb_old:.5f}, Correlation = {corr_blaze_mpb_old:.4f}')
print()
if rmse_blaze_mpb_fixed > 0:
    print(f'The FIXED MPB matches BLAZE {rmse_blaze_mpb_old/rmse_blaze_mpb_fixed:.1f}x better!')

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

vmin = min(V_blaze.min(), V_mpb_fixed.min(), V_mpb_old.min())
vmax = max(V_blaze.max(), V_mpb_fixed.max(), V_mpb_old.max())

# Row 1: Heatmaps
ax1, ax2, ax3 = axes[0]

im1 = ax1.imshow(V_blaze.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                 extent=[R_blaze[:,:,0].min(), R_blaze[:,:,0].max(),
                        R_blaze[:,:,1].min(), R_blaze[:,:,1].max()])
ax1.set_title('V(R) - BLAZE (CORRECT)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(V_mpb_fixed.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                 extent=[R_mpb_fixed[:,:,0].min(), R_mpb_fixed[:,:,0].max(),
                        R_mpb_fixed[:,:,1].min(), R_mpb_fixed[:,:,1].max()])
ax2.set_title('V(R) - MPB FIXED (CORRECT)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2)

im3 = ax3.imshow(V_mpb_old.T, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                 extent=[R_mpb_old[:,:,0].min(), R_mpb_old[:,:,0].max(),
                        R_mpb_old[:,:,1].min(), R_mpb_old[:,:,1].max()])
ax3.set_title('V(R) - MPB OLD (BUGGY - sheared)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_aspect('equal')
plt.colorbar(im3, ax=ax3)

# Row 2: Contours
ax4, ax5, ax6 = axes[1]

contour_levels = np.linspace(vmin, vmax, 12)
cs1 = ax4.contour(R_blaze[:,:,0], R_blaze[:,:,1], V_blaze, levels=contour_levels, cmap='RdBu_r')
ax4.set_title('Contours - BLAZE')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_aspect('equal')

cs2 = ax5.contour(R_mpb_fixed[:,:,0], R_mpb_fixed[:,:,1], V_mpb_fixed, levels=contour_levels, cmap='RdBu_r')
ax5.set_title('Contours - MPB FIXED')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_aspect('equal')

cs3 = ax6.contour(R_mpb_old[:,:,0], R_mpb_old[:,:,1], V_mpb_old, levels=contour_levels, cmap='RdBu_r')
ax6.set_title('Contours - MPB OLD (see the shear)')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_aspect('equal')

plt.tight_layout()
plt.savefig('runs/phase0_blaze_20251201_140949/final_comparison_all_three.png', dpi=150)
print('\nSaved comparison to runs/phase0_blaze_20251201_140949/final_comparison_all_three.png')
