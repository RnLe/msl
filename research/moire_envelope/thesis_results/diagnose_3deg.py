import numpy as np
d = np.load('supercell_3deg_100modes_comparison.npz')
mpb = np.sort(d['freqs_mpb'])
fdfd = np.sort(d['freqs_fdfd'])

print("First 15 modes:")
print("  i       MPB       FDFD      err%")
for i in range(15):
    err = abs(mpb[i] - fdfd[i]) / max(mpb[i], 1e-10) * 100
    print(f"  {i:3d}  {mpb[i]:10.6f}  {fdfd[i]:10.6f}  {err:7.2f}%")

print()
print(f"MPB omega_max = {mpb[-1]:.6f}")
print(f"FDFD omega_max = {fdfd[-1]:.6f}")
print(f"FDFD covers {fdfd[-1]/mpb[-1]*100:.0f}% of MPB range")
print()

# resolution analysis
print("Resolution analysis:")
print(f"  rod diameter in pixels: {2*0.2*32:.1f} pts (at res=32/cell)")
print(f"  MPB is spectral (plane-wave); FDFD is 2nd-order FD")
print(f"  At (11,1) res=32, we had <0.5% error — same rod resolution")
print(f"  Issue might be systematic offset, not random scatter")
print()

# Check if it's a systematic shift
ratios = fdfd[1:] / mpb[1:]
print(f"  FDFD/MPB ratio: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
print(f"  Consistent scale factor? {np.std(ratios)/np.mean(ratios)*100:.2f}% variation")
