#!/usr/bin/env python
"""Analyze Phase A convergence data with corrected MPB frequency scaling."""
import numpy as np

d = np.load('phase_a_convergence_data.npz')
C1_len = 6.082763
mpb = d['mpb_freqs'] / C1_len  # Rescale from c/|C1| to c/a
resolutions = list(d['fdfd_resolutions'])

print(f"MPB freqs (rescaled, c/a): {mpb[:5]}...{mpb[-1]:.6f}")
print(f"|C1| = {C1_len}")
print(f"Resolutions: {resolutions}")
print()

for label in ['binary', 'smoothed']:
    print(f"=== {label.upper()} ===")
    errs = []
    for res in resolutions:
        fdfd = d[f'fdfd_{label}_res{res}']
        e = np.abs(fdfd - mpb).mean()
        emax = np.abs(fdfd - mpb).max()
        errs.append(e)
        print(f"  res={res:4d}  mean|dw|={e:.6e}  max|dw|={emax:.6e}")
    # pairwise rates
    for i in range(1, len(resolutions)):
        if errs[i] > 0 and errs[i-1] > 0:
            rate = np.log(errs[i-1]/errs[i]) / np.log(resolutions[i]/resolutions[i-1])
            print(f"  rate {resolutions[i-1]}->{resolutions[i]}: {rate:.2f}")
    # overall fit
    res_arr = np.array(resolutions, dtype=float)
    errs_arr = np.array(errs)
    valid = errs_arr > 0
    coeffs = np.polyfit(np.log(res_arr[valid]), np.log(errs_arr[valid]), 1)
    print(f"  overall: p = {-coeffs[0]:.2f}")
    print()

# Per-band at highest resolution
best = max(resolutions)
print(f"=== PER-BAND ERROR at res={best} ===")
print(f"{'band':>5s}  {'mpb':>10s}  {'binary':>10s}  {'smoothed':>10s}  {'err_bin':>10s}  {'err_smo':>10s}")
fdfd_b = d[f'fdfd_binary_res{best}']
fdfd_s = d[f'fdfd_smoothed_res{best}']
for i in range(len(mpb)):
    print(f"{i+1:5d}  {mpb[i]:10.6f}  {fdfd_b[i]:10.6f}  {fdfd_s[i]:10.6f}  {fdfd_b[i]-mpb[i]:+10.6f}  {fdfd_s[i]-mpb[i]:+10.6f}")
