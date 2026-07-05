"""
Thesis Results — Shared utilities for T01-T10 compute/plot scripts.

Provides:
  - Candidate definitions loading
  - Run directory discovery
  - Phase 1/2/3 data loading
  - Common plotting configuration
  - η-sweep data loading
"""

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ===========================================================================
# Paths
# ===========================================================================

THESIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THESIS_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runsV3"
FIGURES_DIR = THESIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Plotting config — thesis-quality defaults
# ===========================================================================

THESIS_RC = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6.5, 4.5),
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Candidate colors (colorblind-friendly)
CANDIDATE_COLORS = {
    'hex_M_b1': '#0072B2',    # Blue
    'hex_M_b3': '#D55E00',    # Orange
    'square_M_b3': '#009E73',  # Green
    'honeycomb_K_b1': '#CC79A7',  # Pink/Magenta
}

CANDIDATE_MARKERS = {
    'hex_M_b1': 'o',
    'hex_M_b3': 's',
    'square_M_b3': '^',
    'honeycomb_K_b1': 'D',
}

CANDIDATE_LABELS = {
    'hex_M_b1': 'C1: hex M b1',
    'hex_M_b3': 'C2: hex M b3',
    'square_M_b3': 'C3: sq M b3',
    'honeycomb_K_b1': 'C_hc: hc K Dirac',
}


def apply_thesis_style():
    """Apply thesis-quality matplotlib settings."""
    plt.rcParams.update(THESIS_RC)


# ===========================================================================
# Candidate management
# ===========================================================================


def load_candidates_yaml() -> dict:
    """Load candidates.yaml and return the full dict."""
    yaml_path = THESIS_DIR / "candidates.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_candidate_names() -> list:
    """Return ordered list of candidate names."""
    data = load_candidates_yaml()
    return list(data['candidates'].keys())


def get_candidate(name: str) -> dict:
    """Return candidate definition dict."""
    data = load_candidates_yaml()
    return data['candidates'][name]


def get_pipeline_settings() -> dict:
    """Return shared pipeline settings."""
    data = load_candidates_yaml()
    return data['pipeline']


# ===========================================================================
# Run directory discovery
# ===========================================================================


def find_thesis_run_dir(candidate_name: str) -> Path:
    """Find the latest thesis run directory for a candidate."""
    pattern = f"thesis_{candidate_name}_*"
    matches = sorted(p for p in RUNS_DIR.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"No thesis run directory found for {candidate_name}. "
            f"Run setup_thesis_candidates.py first."
        )
    return matches[-1]


def find_candidate_dir(candidate_name: str) -> Path:
    """Find the candidate_XXXX subdirectory for a thesis candidate."""
    run_dir = find_thesis_run_dir(candidate_name)
    # Thesis candidates use id=0,1,2 in order
    cand_dirs = sorted(run_dir.glob("candidate_*"))
    if not cand_dirs:
        raise FileNotFoundError(f"No candidate directory found in {run_dir}")
    return cand_dirs[0]


def find_all_candidate_dirs() -> dict:
    """Return {candidate_name: candidate_dir} for all thesis candidates."""
    result = {}
    for name in get_candidate_names():
        try:
            result[name] = find_candidate_dir(name)
        except FileNotFoundError:
            result[name] = None
    return result


# ===========================================================================
# Data loading
# ===========================================================================


def load_phase1_data(cand_dir: Path) -> dict:
    """Load Phase 1 multiband data from HDF5."""
    h5_path = cand_dir / "phase1_multiband_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 1 data not found: {h5_path}")

    data = {}
    with h5py.File(h5_path, 'r') as hf:
        for key in hf.keys():
            obj = hf[key]
            if isinstance(obj, h5py.Dataset):
                # Skip very large datasets (bloch_fields ~19 GB, epsilon ~0.5 GB)
                if obj.nbytes > 1e9:
                    continue
                data[key] = obj[:]
            # Skip groups (like 'stencil') — load specific keys if needed
        for key, val in hf.attrs.items():
            data[f'attr_{key}'] = val
    return data


def load_phase2_data(cand_dir: Path) -> dict:
    """Load Phase 2 multiband data from HDF5."""
    h5_path = cand_dir / "phase2_multiband_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {h5_path}")

    data = {}
    with h5py.File(h5_path, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
        for key, val in hf.attrs.items():
            data[f'attr_{key}'] = val
    return data


def load_phase3_data(cand_dir: Path) -> dict:
    """Load Phase 3 envelope modes from HDF5."""
    h5_path = cand_dir / "phase3_multiband_modes.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 3 data not found: {h5_path}")

    data = {}
    with h5py.File(h5_path, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
        for key, val in hf.attrs.items():
            data[f'attr_{key}'] = val
    return data


def load_phase0_meta(cand_dir: Path) -> dict:
    """Load phase0_meta.json for a candidate."""
    meta_path = cand_dir / "phase0_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Phase 0 meta not found: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def load_eta_sweep_data(cand_dir: Path) -> list:
    """Load η-sweep results from all theta subdirs."""
    sweep_dirs = sorted(cand_dir.parent.glob("eta_sweep_*"))
    if not sweep_dirs:
        return []

    sweep_dir = sweep_dirs[-1]  # Latest sweep
    results = []
    for theta_dir in sorted(sweep_dir.glob("theta_*")):
        # Phase 3 HDF5 is in candidate_0000/ subdirectory
        h5_path = theta_dir / "candidate_0000" / "phase3_multiband_modes.h5"
        if not h5_path.exists():
            # Fallback: check directly in theta_dir
            h5_path = theta_dir / "phase3_multiband_modes.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, 'r') as hf:
            result = {
                'theta_deg': float(hf.attrs.get('theta_deg', 0)),
                'eta': float(hf.attrs.get('eta', 0)),
                'eigenvalues': hf['eigenvalues'][:] if 'eigenvalues' in hf else None,
            }
            if 'envelope_modes' in hf:
                result['envelope_modes'] = hf['envelope_modes'][:]
        results.append(result)
    return results


# ===========================================================================
# Helpers
# ===========================================================================


def ensure_output_dir(task_name: str) -> Path:
    """Create and return output directory for a task (e.g., 'T01_candidate_selection')."""
    out_dir = THESIS_DIR / task_name
    out_dir.mkdir(exist_ok=True)
    return out_dir


def save_figure(fig, task_name: str, fig_name: str, also_pdf=True):
    """Save figure to task dir and figures/ dir."""
    task_dir = ensure_output_dir(task_name)

    # Save PNG
    png_path = task_dir / f"{fig_name}.png"
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"  Saved: {png_path}")

    # Save PDF for LaTeX
    if also_pdf:
        pdf_path = task_dir / f"{fig_name}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved: {pdf_path}")

    # Copy to figures/ for easy access
    fig_copy = FIGURES_DIR / f"{fig_name}.png"
    fig.savefig(fig_copy, bbox_inches='tight', dpi=300)

    plt.close(fig)
