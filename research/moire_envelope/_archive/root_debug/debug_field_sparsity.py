
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add phasesV3 to path
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase4_field_reconstruction as p4

def print_ascii_histogram(data, bins=20, title="Histogram"):
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = np.max(hist)
    print(f"\n{title}:")
    print(f"Range: [{np.min(data):.5f}, {np.max(data):.5f}]")
    for i in range(bins):
        bar_len = int(50 * hist[i] / max_count)
        bar = "#" * bar_len
        bin_range = f"{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}"
        print(f"{bin_range:>20} | {hist[i]:>6} {bar}")

def main():
    try:
        run_dir = p4.find_latest_run_dir()
        candidate_id = 0 # Assume candidate 0
        cdir = p4.candidate_dir(run_dir, candidate_id)
        
        print(f"Loading data from {cdir}...")
        
        bloch_fields, subspace_bands, all_bands = p4.load_phase1_bloch_fields(cdir)
        F_spinor, eigenvalues, mode_stats = p4.load_phase3_envelopes(cdir)
        band_indices = p4.get_subspace_band_indices(subspace_bands, all_bands)
        
        mode_idx = 10
        print(f"\nAnalyzing Mode {mode_idx}...")
        
        # Determine component (copied logic)
        comp_z_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 2]))
        component = 2 if comp_z_max > 1e-10 else 0 # Default to Ex if TE, but logic in p4 handles this.
        # Actually p4.plot_reconstructed_modes handles the logic of reconstructing full vector field or specific component.
        # Let's just reconstruct the component used in the Real(R) plot.
        # In p4 line 535: it reconstructs 'E_field' which is single component or Ex.
        
        use_component = -1
        if comp_z_max > 1e-10:
             use_component = 2
             print("Mode is TM (Ez dominant)")
        else:
             print("Mode is TE (Ex/Ey in-plane)")
             # For TE, the Real(E) plot usually shows Ex (component 0)
             use_component = 0

        bloch_cache = {}
        E_field = p4.reconstruct_full_field_single_mode(
            mode_idx=mode_idx,
            F_spinor=F_spinor,
            bloch_fields=bloch_fields,
            band_indices=band_indices,
            component=use_component if use_component >= 0 else 0,
            include_bloch_phase=False,
            bloch_interp_cache=bloch_cache,
        )
        
        e_real = E_field.real
        vmax = np.max(np.abs(e_real))
        e_real_norm = e_real / vmax if vmax > 1e-15 else e_real
        
        print_ascii_histogram(np.abs(e_real_norm).flatten(), bins=20, title="|Real(E)| Normalized Histogram")
        
        # Check sparsity specifically near 1.0
        threshold = 0.5
        count_high = np.sum(np.abs(e_real_norm) > threshold)
        total_pixels = e_real_norm.size
        print(f"\nSparsity Check:")
        print(f"Total pixels: {total_pixels}")
        print(f"Pixels with |val| > {threshold}: {count_high} ({count_high/total_pixels*100:.2f}%)")
        
        threshold = 0.1
        count_med = np.sum(np.abs(e_real_norm) > threshold)
        print(f"Pixels with |val| > {threshold}: {count_med} ({count_med/total_pixels*100:.2f}%)")
        
        # Check kurtosis
        mean = np.mean(e_real_norm)
        std = np.std(e_real_norm)
        kurtosis = np.mean(((e_real_norm - mean)/std)**4) - 3
        print(f"\nKurtosis: {kurtosis:.2f} (Normal dist = 0, Higher = heavier tails/spikier)")

        # === Investigate Envelope F ===
        F_mode = F_spinor[mode_idx] # (Ns1, Ns2, 3)
        F_amp = np.sqrt(np.sum(np.abs(F_mode)**2, axis=2))
        
        print_ascii_histogram(F_amp.flatten(), bins=20, title="|F| Amplitude Histogram")
        
        f_kurtosis = np.mean(((F_amp.flatten() - np.mean(F_amp))/np.std(F_amp))**4) - 3
        print(f"Envelope Kurtosis: {f_kurtosis:.2f}")

        # === Investigate Bloch Function u ===
        # Just check one band, one component
        u_slice = bloch_fields[:, :, 0, :, :, 0] # (Ns1, Ns2, Nx, Ny)
        u_slice_amp = np.abs(u_slice).flatten()
        print_ascii_histogram(u_slice_amp, bins=20, title="|u| Amplitude Histogram (Band 0, Ex)")
        
        u_kurtosis = np.mean(((u_slice_amp - np.mean(u_slice_amp))/np.std(u_slice_amp))**4) - 3
        print(f"Bloch u Kurtosis: {u_kurtosis:.2f}")

        # === Percentiles u ===
        u_p99 = np.percentile(u_slice_amp, 99)
        u_max = np.max(u_slice_amp)
        print(f"Bloch u max/99%: {u_max/u_p99:.2f}x")

        # === Percentiles E ===
        p90 = np.percentile(np.abs(e_real_norm), 90)
        p99 = np.percentile(np.abs(e_real_norm), 99)
        p999 = np.percentile(np.abs(e_real_norm), 99.9)
        p_max = np.max(np.abs(e_real_norm))
        
        print("\nPercentiles of |Real(E)| normalized:")
        print(f"90.0%: {p90:.5f}")
        print(f"99.0%: {p99:.5f}")
        print(f"99.9%: {p999:.5f}")
        print(f"Max  : {p_max:.5f}")
        print(f"Max/99.9%: {p_max/p999:.2f}x")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
