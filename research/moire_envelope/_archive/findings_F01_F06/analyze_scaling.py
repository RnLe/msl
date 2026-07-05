#!/usr/bin/env python3
"""Analyze per-band miniband scaling from corrected sweep data."""
import json
import numpy as np

SWEEP = '../runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
with open(f'{SWEEP}/sweep_results_corrected.json') as f:
    data = json.load(f)

data_sorted = sorted(data, key=lambda x: x['theta_deg'])

# --- Power-law fits ---
etas = np.array([d['eta'] for d in data_sorted])
log_eta = np.log(etas)

print("=" * 80)
print("POWER-LAW FITS: BW20 ~ eta^alpha")
print("=" * 80)

for band in range(3):
    bw20 = []
    delta_shallow_vals = []
    for d in data_sorted:
        pb = d['per_band'][band]
        evals = np.sort(pb['eigenvalues'])
        bw20.append(abs(evals[-1] - evals[0]))
        V_max, V_min = pb['V_max'], pb['V_min']
        if pb['type'] == 'hole':
            delta_shallow_vals.append(abs(V_max - evals[-1]))
        else:
            delta_shallow_vals.append(abs(evals[0] - V_min))
    
    bw20 = np.array(bw20)
    delta_shallow_vals = np.array(delta_shallow_vals)
    
    log_bw = np.log(bw20)
    coeffs = np.polyfit(log_eta, log_bw, 1)
    coeffs_small = np.polyfit(log_eta[:5], log_bw[:5], 1)
    
    log_delta = np.log(delta_shallow_vals)
    coeffs_d = np.polyfit(log_eta, log_delta, 1)
    
    btype = data_sorted[0]['per_band'][band]['type']
    M = data_sorted[0]['per_band'][band]['mean_mass_trace']
    print(f"\nBand {band} ({btype}, M={M:.2f}):")
    print(f"  BW20 full-range exponent:  alpha = {coeffs[0]:.3f}")
    print(f"  BW20 small-eta exponent:   alpha = {coeffs_small[0]:.3f}")
    print(f"  delta_shallow exponent:    alpha = {coeffs_d[0]:.3f}")
    print(f"  BW20 range: {bw20[0]:.2e} to {bw20[-1]:.2e}")

# --- Kinetic energy scale analysis ---
print("\n" + "=" * 80)
print("KINETIC vs POTENTIAL ENERGY SCALES")
print("=" * 80)
print(f"{'theta':>6} {'eta':>8} {'T_q1_b0':>10} {'T/V_b0':>8} {'T_q1_b1':>10} {'T/V_b1':>8} {'T_q1_b2':>10} {'T/V_b2':>8}")

for d in data_sorted:
    eta = d['eta']
    vals = []
    for band in range(3):
        pb = d['per_band'][band]
        M_abs = abs(pb['mean_mass_trace'])
        V_range = pb['V_max'] - pb['V_min']
        T_q1 = eta**2 / (2 * M_abs)
        vals.extend([T_q1, T_q1 / V_range])
    print(f"{d['theta_deg']:6.1f} {eta:8.5f} {vals[0]:10.2e} {vals[1]:8.3f} {vals[2]:10.2e} {vals[3]:8.3f} {vals[4]:10.2e} {vals[5]:8.3f}")

# --- Eigenvalue spacings near band edge ---
print("\n" + "=" * 80)
print("EIGENVALUE SPACINGS NEAR BAND EDGE (sorted, 5 closest to extremum)")
print("=" * 80)

for band in range(3):
    btype = data_sorted[0]['per_band'][band]['type']
    M = data_sorted[0]['per_band'][band]['mean_mass_trace']
    print(f"\n--- Band {band} ({btype}, M={M:.2f}) ---")
    print(f"{'theta':>6} {'eta':>8} {'near-edge spacings (4 gaps among 5 states)':>60}")
    
    for d in data_sorted:
        pb = d['per_band'][band]
        eta = d['eta']
        evals = np.sort(pb['eigenvalues'])
        
        if btype == 'hole':
            near_edge = evals[-5:]  # 5 closest to V_max
        else:
            near_edge = evals[:5]   # 5 closest to V_min
        
        spacings = np.diff(near_edge)
        sp_str = '  '.join([f'{s:+.4e}' for s in spacings])
        print(f"{d['theta_deg']:6.1f} {eta:8.5f}   {sp_str}")

# --- Full eigenvalue dump for smallest angle ---
print("\n" + "=" * 80)
print("FULL SORTED EIGENVALUES AT theta=0.5 deg (all 20)")
print("=" * 80)

d0 = data_sorted[0]
for band in range(3):
    pb = d0['per_band'][band]
    evals = np.sort(pb['eigenvalues'])
    btype = pb['type']
    V_max, V_min = pb['V_max'], pb['V_min']
    print(f"\nBand {band} ({btype}), V=[{V_min:.6f}, {V_max:.6f}]")
    for i, e in enumerate(evals):
        marker = ""
        if e > V_max:
            marker = " ** ABOVE V_max **"
        elif e < V_min:
            marker = " ** BELOW V_min **"
        print(f"  [{i:2d}] {e:.10f}{marker}")
