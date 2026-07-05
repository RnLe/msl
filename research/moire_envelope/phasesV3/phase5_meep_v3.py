"""
Phase 5 (Meep): FDTD Validation & Q-Factor Analysis — V3 Multi-Band Pipeline

Validates envelope approximation predictions using full Meep FDTD simulations.

FEATURES:
1. Continuous-wave source at cavity frequency (smallest spread mode)
2. Multiple Q-value estimation methods:
   - Harminv ringdown analysis
   - Energy decay fitting (U(t) ∝ exp(-2γt))
   - Power loss method (Q = ωU/P)
3. Streaming data to metrics and MP4 video
4. Large-scale simulation support (64×64 per unit cell, 2×2 moiré supercell)

VALIDATION METRICS:
- Q-factor from multiple methods
- Energy density maps (comparable to |F|²)
- Mode localization (IPR, spread)
- Frequency comparison (EA vs Meep)

THEORY REFERENCE: docs/envelopeApproximationDerivation/6_ValidationStrategiesAndPitfalls.md
"""

# ==============================================================================
# Threading Configuration for Meep FDTD
# ==============================================================================
# Meep can use OpenMP for multi-threaded FDTD. Unlike MPB (where we use Python
# multiprocessing), Meep benefits from internal OpenMP parallelism.
#
# The number of threads can be configured via:
# 1. MEEP_NUM_THREADS environment variable (set before import)
# 2. Meep's `num_chunks` parameter in Simulation (for MPI decomposition)
#
# For pure OpenMP (no MPI), MEEP_NUM_THREADS is the key variable.
# ==============================================================================
import os

# Get number of threads from environment or use all available cores
_n_threads = os.environ.get('MEEP_NUM_THREADS', None)
if _n_threads is None:
    import multiprocessing
    _n_threads = str(multiprocessing.cpu_count())
    
# Set threading environment for OpenMP-based parallelism
os.environ['OMP_NUM_THREADS'] = _n_threads
os.environ['OPENBLAS_NUM_THREADS'] = _n_threads
os.environ['MKL_NUM_THREADS'] = _n_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = _n_threads
os.environ['NUMEXPR_NUM_THREADS'] = _n_threads

import argparse
import math
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    import meep as mp
except ImportError:
    print("ERROR: meep package not installed. Install with: pip install meep")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, save_json, load_json


def log(message: str):
    """Print message with flush, only on rank 0."""
    if mp.am_master():
        print(message, flush=True)


# ==============================================================================
# Data Classes for Simulation Configuration
# ==============================================================================

@dataclass
class SimulationConfig:
    """Configuration for Meep simulation."""
    # Resolution: pixels per monolayer lattice constant
    resolution_per_a: int = 64
    
    # Moiré supercell tiling (2×2 recommended for validation)
    supercell_tiles: Tuple[int, int] = (2, 2)
    
    # PML thickness in lattice constants
    pml_thickness: float = 2.0
    
    # Source settings
    source_type: str = 'continuous'  # 'continuous' or 'gaussian'
    source_ramp_time: float = 20.0  # Time to ramp up CW source
    
    # Simulation timing (in units of 1/frequency)
    steady_state_time: float = 100.0  # Time to reach steady state
    measurement_delay: float = 50.0   # Delay after source off before measurement
    ringdown_time: float = 200.0      # Time to measure ringdown
    
    # Data streaming
    video_fps: int = 30
    video_capture_interval: float = 1.0  # Simulation time between frames
    
    # Measurement regions
    roi_fraction: float = 0.5  # Fraction of moiré cell for ROI
    
    # Memory optimization
    max_video_frames: int = 600
    field_snapshot_stride: int = 4  # Downsample field snapshots
    
    @property
    def total_time(self) -> float:
        """Total simulation time."""
        return (self.source_ramp_time + self.steady_state_time + 
                self.measurement_delay + self.ringdown_time)


def build_simulation_config(config: Dict) -> SimulationConfig:
    """
    Build SimulationConfig from YAML config dictionary.
    
    Handles:
    - Threading configuration (sets environment variables)
    - Computing video_capture_interval from target_video_frames
    - All standard config parameters
    """
    import multiprocessing
    
    # --- Threading Setup ---
    n_threads = config.get('meep_num_threads', 'auto')
    if n_threads in ('auto', 0, '0', None):
        n_threads = multiprocessing.cpu_count()
    n_threads = int(n_threads)
    
    # Update environment (must be done before Meep uses OpenMP)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    
    log(f"  Threading: Using {n_threads} OpenMP threads for Meep FDTD")
    
    # --- Timing Parameters ---
    source_ramp = config.get('phase5_source_ramp_time', 50.0)
    steady_state = config.get('phase5_steady_state_time', 200.0)
    delay = config.get('phase5_measurement_delay', 50.0)
    ringdown = config.get('phase5_ringdown_time', 100.0)
    
    total_sim_time = source_ramp + steady_state + delay + ringdown
    
    # --- Video Parameters ---
    target_frames = config.get('phase5_target_video_frames', 200)
    video_fps = config.get('phase5_video_fps', 30)
    max_frames = config.get('phase5_max_video_frames', 500)
    
    # Compute video interval to achieve target frame count
    # interval = total_time / target_frames
    video_interval = total_sim_time / max(target_frames, 1)
    
    # Ensure we don't exceed max_frames
    actual_frames = min(int(total_sim_time / video_interval), max_frames)
    video_duration = actual_frames / video_fps
    
    log(f"  Timing: total={total_sim_time:.1f} (ramp={source_ramp}, steady={steady_state}, "
        f"delay={delay}, ringdown={ringdown})")
    log(f"  Video: {actual_frames} frames at {video_fps} fps = {video_duration:.1f}s playback")
    log(f"         capture interval = {video_interval:.2f} sim time units")
    
    return SimulationConfig(
        resolution_per_a=config.get('phase5_resolution_per_a', 64),
        supercell_tiles=tuple(config.get('phase5_supercell_tiles', [2, 2])),
        pml_thickness=config.get('phase5_pml_thickness', 2.0),
        source_ramp_time=source_ramp,
        steady_state_time=steady_state,
        measurement_delay=delay,
        ringdown_time=ringdown,
        video_fps=video_fps,
        video_capture_interval=video_interval,
        roi_fraction=config.get('phase5_roi_fraction', 0.5),
        max_video_frames=max_frames,
        field_snapshot_stride=config.get('phase5_field_stride', 8),
    )

@dataclass
class SimulationMetrics:
    """Accumulated metrics during simulation."""
    # Time series
    time_points: List[float] = field(default_factory=list)
    field_energy: List[float] = field(default_factory=list)
    roi_energy: List[float] = field(default_factory=list)
    flux_out: List[float] = field(default_factory=list)
    
    # Harminv results
    harminv_modes: List[Dict] = field(default_factory=list)
    
    # Computed Q values
    Q_harminv: float = float('nan')
    Q_energy_decay: float = float('nan')
    Q_power_loss: float = float('nan')
    
    # Frequency measurements
    omega_target: float = float('nan')
    omega_measured: float = float('nan')


# ==============================================================================
# Geometry Construction
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build monolayer lattice basis B = (a1 | a2)."""
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3)/2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def rotation_matrix(theta_rad: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """Compute moiré lattice basis: B_moire = (R(θ) - I)^{-1} @ B_mono."""
    R = rotation_matrix(theta_rad)
    Delta_R = R - np.eye(2)
    return np.linalg.inv(Delta_R) @ B_mono


def lattice_points_in_region(
    a1: np.ndarray, a2: np.ndarray,
    bounds: Tuple[float, float, float, float],
    padding: float = 0.0
) -> np.ndarray:
    """Generate lattice points within rectangular bounds."""
    xmin, xmax, ymin, ymax = bounds
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding
    
    # Convert bounds to lattice coordinates
    basis = np.column_stack([a1[:2], a2[:2]])
    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        return np.zeros((0, 2))
    
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]])
    frac = (basis_inv @ corners.T).T
    
    i_min = int(np.floor(frac[:, 0].min())) - 2
    i_max = int(np.ceil(frac[:, 0].max())) + 2
    j_min = int(np.floor(frac[:, 1].min())) - 2
    j_max = int(np.ceil(frac[:, 1].max())) + 2
    
    points = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            pt = i * a1[:2] + j * a2[:2]
            if xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax:
                points.append(pt)
    
    return np.array(points) if points else np.zeros((0, 2))


def build_bilayer_geometry(
    lattice_type: str,
    theta_deg: float,
    a: float,
    r_over_a: float,
    eps_bg: float,
    supercell_tiles: Tuple[int, int],
    moire_length: float,
    center_offset: Optional[np.ndarray] = None,
) -> Dict:
    """
    Build Meep geometry for twisted bilayer photonic crystal.
    
    Args:
        lattice_type: 'triangular' or 'square'
        theta_deg: Twist angle in degrees
        a: Monolayer lattice constant
        r_over_a: Hole radius as fraction of a
        eps_bg: Background dielectric constant
        supercell_tiles: (Tx, Ty) number of moiré cells in each direction
        moire_length: Length of one moiré unit cell
        center_offset: Optional [x, y] offset to shift geometry. 
                       If provided, ALL holes are shifted by -center_offset,
                       effectively centering the simulation around this point.
                       Use this to center the cavity mode at (0,0).
    
    Returns dict with geometry, bounds, and metadata.
    """
    theta_rad = math.radians(theta_deg)
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    
    a1_mono = B_mono[:, 0]
    a2_mono = B_mono[:, 1]
    
    # Rotated top layer basis
    R = rotation_matrix(theta_rad)
    a1_top = R @ a1_mono
    a2_top = R @ a2_mono
    
    # Simulation window: supercell_tiles × moire_length
    window_x = supercell_tiles[0] * moire_length
    window_y = supercell_tiles[1] * moire_length
    bounds = (-window_x/2, window_x/2, -window_y/2, window_y/2)
    
    # If center_offset provided, we need to generate lattice points 
    # around that center, then shift everything to origin
    if center_offset is not None:
        # Expand bounds around the offset center
        shifted_bounds = (
            bounds[0] + center_offset[0],
            bounds[1] + center_offset[0],
            bounds[2] + center_offset[1],
            bounds[3] + center_offset[1],
        )
        log(f"  Centering geometry: shifting by ({-center_offset[0]:.3f}, {-center_offset[1]:.3f})")
    else:
        shifted_bounds = bounds
    
    # Generate lattice points for both layers (in original coordinates)
    padding = r_over_a * a
    bottom_points = lattice_points_in_region(a1_mono, a2_mono, shifted_bounds, padding)
    top_points = lattice_points_in_region(a1_top, a2_top, shifted_bounds, padding)
    
    # Apply the shift to center geometry at origin
    if center_offset is not None:
        bottom_points = bottom_points - center_offset
        top_points = top_points - center_offset
    
    # Create Meep geometry: air holes in dielectric
    radius = r_over_a * a
    air = mp.Medium(epsilon=1.0)
    geometry = []
    
    for pt in bottom_points:
        geometry.append(mp.Cylinder(
            radius=radius,
            height=mp.inf,
            center=mp.Vector3(pt[0], pt[1], 0),
            material=air
        ))
    
    for pt in top_points:
        geometry.append(mp.Cylinder(
            radius=radius,
            height=mp.inf,
            center=mp.Vector3(pt[0], pt[1], 0),
            material=air
        ))
    
    return {
        'geometry': geometry,
        'bounds': bounds,  # Original bounds (simulation window stays centered at origin)
        'window_size': (window_x, window_y),
        'bottom_points': bottom_points,
        'top_points': top_points,
        'radius': radius,
        'eps_bg': eps_bg,
        'B_mono': B_mono,
        'B_moire': B_moire,
        'a1_mono': a1_mono,
        'a2_mono': a2_mono,
        'a1_top': a1_top,
        'a2_top': a2_top,
        'n_bottom': len(bottom_points),
        'n_top': len(top_points),
        'center_offset': center_offset,  # Store for reference
    }


# ==============================================================================
# Mode Selection
# ==============================================================================

def load_phase3_modes(cdir: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load Phase 3 eigenvalues and find mode with smallest spread."""
    # Try multiple possible file formats
    json_path = cdir / "phase3_mode_stats.json"
    csv_path = cdir / "phase3_eigenvalues.csv"
    h5_path = cdir / "phase3_multiband_modes.h5"
    
    df = None
    F_best = None
    
    # Try JSON format first (V3 format)
    if json_path.exists():
        try:
            mode_stats = load_json(json_path)
            df = pd.DataFrame(mode_stats)
            log(f"  Loaded {len(df)} modes from phase3_mode_stats.json")
        except Exception as e:
            log(f"    Warning: Could not load JSON mode stats: {e}")
    
    # Try CSV format
    if df is None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            log(f"  Loaded {len(df)} modes from phase3_eigenvalues.csv")
        except Exception as e:
            log(f"    Warning: Could not load CSV: {e}")
    
    if df is None:
        raise FileNotFoundError(
            f"Phase 3 mode data not found in {cdir}. "
            "Expected phase3_mode_stats.json or phase3_eigenvalues.csv"
        )
    
    # Try to load envelope data for cavity position
    if h5_path.exists():
        try:
            with h5py.File(h5_path, 'r') as hf:
                # Check for different dataset names
                for key in ['F_spinor', 'F_envelope', 'eigenvectors']:
                    if key in hf:
                        F_all = hf[key][:]
                        # Find mode with smallest spread
                        if 'spread' in df.columns:
                            best_idx = int(df['spread'].idxmin())
                            if best_idx < len(F_all):
                                F_best = F_all[best_idx]
                                log(f"  Loaded envelope for mode {best_idx} (smallest spread)")
                        break
        except Exception as e:
            log(f"    Warning: Could not load envelope data: {e}")
    
    return df, F_best


def has_phase3_results(cdir: Path) -> bool:
    """Check if candidate directory has Phase 3 results."""
    return (
        (cdir / "phase3_mode_stats.json").exists() or
        (cdir / "phase3_eigenvalues.csv").exists() or
        (cdir / "phase3_multiband_modes.h5").exists()
    )
    
    return df, F_best


def select_target_mode(df: pd.DataFrame, config: Dict) -> pd.Series:
    """Select target mode for validation (smallest spread by default)."""
    mode_selection = config.get('phase5_mode_selection', 'min_spread')
    
    if mode_selection == 'min_spread' and 'spread' in df.columns:
        idx = df['spread'].idxmin()
        mode = df.loc[idx].copy()
    elif mode_selection == 'min_delta' and 'delta_omega' in df.columns:
        idx = df['delta_omega'].abs().idxmin()
        mode = df.loc[idx].copy()
    else:
        # Default: first mode
        mode = df.iloc[0].copy()
    
    # Normalize field names for compatibility
    # The JSON format uses 'omega', CSV might use 'omega_cavity'
    if 'omega' in mode and 'omega_cavity' not in mode:
        mode['omega_cavity'] = mode['omega']
    
    return mode


def compute_cavity_position(
    F: Optional[np.ndarray],
    s_grid_shape: Tuple[int, int],
    B_moire: np.ndarray
) -> np.ndarray:
    """Compute weighted average position of cavity mode."""
    if F is None:
        # Default to origin
        return np.array([0.0, 0.0])
    
    # Compute |F|² = Σ_n |F_n|²
    prob = np.sum(np.abs(F)**2, axis=-1)
    prob /= prob.sum()
    
    # Grid in fractional coordinates
    Ns1, Ns2 = s_grid_shape
    s1 = np.arange(Ns1) / Ns1
    s2 = np.arange(Ns2) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    
    # Weighted average in fractional coords
    s1_avg = np.sum(prob * S1)
    s2_avg = np.sum(prob * S2)
    
    # Convert to Cartesian
    s_avg = np.array([s1_avg, s2_avg])
    r_avg = B_moire @ s_avg
    
    return r_avg


# ==============================================================================
# Pre-Simulation Plots
# ==============================================================================

def plot_mpb_geometry(sim: mp.Simulation, out_path: Path, dpi: int = 150):
    """Generate MPB-style geometry plot using Meep's plot2D."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    sim.plot2D(ax=ax)
    ax.set_title("Meep Simulation Geometry (plot2D)")
    ax.set_xlabel("x (a)")
    ax.set_ylabel("y (a)")
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log(f"  Saved geometry plot: {out_path}")


def plot_simulation_setup(
    geo_ctx: Dict,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    roi_bounds: Tuple[float, float, float, float],
    flux_positions: List[np.ndarray],
    out_path: Path,
    dpi: int = 150
):
    """
    Custom plot showing simulation setup:
    - Both monolayers (bottom blue, top pink)
    - Theoretical cavity location
    - Source position
    - ROI and flux monitor regions
    """
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    
    # Plot bottom layer holes
    bottom_pts = geo_ctx['bottom_points']
    if len(bottom_pts) > 0:
        ax.scatter(
            bottom_pts[:, 0], bottom_pts[:, 1],
            s=8, c='#3b82f6', alpha=0.6, edgecolors='none',
            label=f'Bottom layer ({len(bottom_pts)} holes)'
        )
    
    # Plot top layer holes
    top_pts = geo_ctx['top_points']
    if len(top_pts) > 0:
        ax.scatter(
            top_pts[:, 0], top_pts[:, 1],
            s=8, c='#ec4899', alpha=0.6, edgecolors='none',
            label=f'Top layer ({len(top_pts)} holes)'
        )
    
    # Plot theoretical cavity position
    ax.scatter(
        [cavity_pos[0]], [cavity_pos[1]],
        s=200, c='#f59e0b', marker='*', edgecolors='black', linewidths=1,
        label='Cavity (EA prediction)', zorder=10
    )
    
    # Plot source position
    ax.scatter(
        [source_pos[0]], [source_pos[1]],
        s=150, c='#22c55e', marker='o', edgecolors='black', linewidths=1,
        label='CW Source', zorder=10
    )
    
    # Plot ROI bounds
    roi_xmin, roi_xmax, roi_ymin, roi_ymax = roi_bounds
    roi_rect = plt.Rectangle(
        (roi_xmin, roi_ymin), roi_xmax - roi_xmin, roi_ymax - roi_ymin,
        fill=False, edgecolor='#8b5cf6', linewidth=2, linestyle='--',
        label='Energy ROI'
    )
    ax.add_patch(roi_rect)
    
    # Plot flux monitor positions
    for i, pos in enumerate(flux_positions):
        marker = 's' if i < 2 else '^'
        label = 'Flux monitors' if i == 0 else None
        ax.scatter(
            [pos[0]], [pos[1]],
            s=80, c='#ef4444', marker=marker, edgecolors='white', linewidths=0.5,
            label=label, zorder=9
        )
    
    # Window bounds
    xmin, xmax, ymin, ymax = geo_ctx['bounds']
    ax.axhline(ymin, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(ymax, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(xmin, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(xmax, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlim(xmin * 1.1, xmax * 1.1)
    ax.set_ylim(ymin * 1.1, ymax * 1.1)
    ax.set_xlabel('x (a)', fontsize=12)
    ax.set_ylabel('y (a)', fontsize=12)
    ax.set_title('Simulation Setup: Twisted Bilayer PhC', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log(f"  Saved setup plot: {out_path}")


# ==============================================================================
# Streaming Video Generation
# ==============================================================================

class VideoStreamer:
    """Stream simulation frames to MP4 video without storing all in memory."""
    
    def __init__(
        self,
        output_path: Path,
        frame_shape: Tuple[int, int],
        fps: int = 30,
        cmap: str = 'RdBu_r',
        timing_config: Optional[Dict] = None,
        omega: float = 1.0
    ):
        self.output_path = output_path
        self.frame_shape = frame_shape
        self.fps = fps
        self.cmap = plt.get_cmap(cmap)
        self.frame_count = 0
        self.temp_dir = None
        # Fixed colorbar scale from -1 to 1
        self.vmin = -1.0
        self.vmax = 1.0
        # Store timing configuration for phase labels
        self.timing_config = timing_config or {}
        self.omega = omega  # For computing oscillation count
        # Cumulative energy density
        self.cumulative_energy = None
        
    def start(self):
        """Initialize temporary directory for frames."""
        self.temp_dir = tempfile.mkdtemp(prefix='meep_video_')
        self.frame_count = 0
        self.cumulative_energy = None
    
    def _get_phase_label(self, sim_time: float) -> str:
        """Determine the current simulation phase based on time."""
        ramp = self.timing_config.get('ramp', 0)
        steady = self.timing_config.get('steady', 0)
        delay = self.timing_config.get('delay', 0)
        
        t_end_ramp = ramp
        t_end_steady = ramp + steady
        t_end_delay = ramp + steady + delay
        
        if sim_time < t_end_ramp:
            return "Ramping Up"
        elif sim_time < t_end_steady:
            return "Steady State"
        elif sim_time < t_end_delay:
            return "Source Off"
        else:
            return "Ringdown"
        
    def add_frame(self, field_data: np.ndarray, sim_time: float):
        """Add a frame to the video stream."""
        if self.temp_dir is None:
            self.start()
        
        # Accumulate energy density (|E|^2)
        frame_energy = field_data ** 2
        if self.cumulative_energy is None:
            self.cumulative_energy = frame_energy.copy()
        else:
            self.cumulative_energy += frame_energy
        
        # Compute oscillation count: N = omega * t / (2*pi)
        oscillations = self.omega * sim_time / (2 * np.pi)
        
        # Get phase label
        phase_label = self._get_phase_label(sim_time)
        
        # Create frame image (use fixed size to ensure even dimensions for H.264)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        im = ax.imshow(
            field_data.T, origin='lower', cmap=self.cmap,
            norm=norm, aspect='equal'
        )
        # Title with phase and oscillation count
        ax.set_title(f't = {sim_time:.1f} | {phase_label} | {oscillations:.0f} oscillations', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        frame_path = Path(self.temp_dir) / f'frame_{self.frame_count:06d}.png'
        # Don't use bbox_inches='tight' to keep fixed size (800x800)
        fig.savefig(frame_path, dpi=100)
        plt.close(fig)
        
        self.frame_count += 1
        
    def finalize(self) -> bool:
        """Encode frames to MP4 using ffmpeg."""
        if self.temp_dir is None or self.frame_count == 0:
            log(f"  Warning: No frames to encode (temp_dir={self.temp_dir}, count={self.frame_count})")
            return False
        
        frame_pattern = str(Path(self.temp_dir) / 'frame_%06d.png')
        
        # Debug: check if frames exist
        import glob
        existing_frames = glob.glob(str(Path(self.temp_dir) / 'frame_*.png'))
        log(f"  Found {len(existing_frames)} frame files in {self.temp_dir}")
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.fps),
                '-i', frame_pattern,
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Ensure even dimensions for H.264
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                str(self.output_path)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                # Extract actual error (skip version info)
                stderr_lines = result.stderr.strip().split('\n')
                error_lines = [l for l in stderr_lines if 'error' in l.lower() or 'invalid' in l.lower() or 'no such' in l.lower()]
                if error_lines:
                    log(f"  Warning: ffmpeg error: {error_lines[0][:200]}")
                else:
                    log(f"  Warning: ffmpeg failed with code {result.returncode}")
                    log(f"  Frames captured: {self.frame_count}")
                    log(f"  Temp dir: {self.temp_dir}")
                return False
            
            log(f"  Video saved: {self.output_path} ({self.frame_count} frames)")
            return True
            
        except FileNotFoundError:
            log("  Warning: ffmpeg not found, video not created")
            return False
        except Exception as e:
            log(f"  Warning: Video encoding failed: {e}")
            return False
        finally:
            # Cleanup temp files
            import shutil
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def save_cumulative_energy_plot(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Save accumulated |E|^2 as a plot (Phase 3-style energy density).
        
        This gives a time-integrated view of where field energy resided,
        similar to the envelope approximation mode plots from Phase 3.
        """
        if self.cumulative_energy is None:
            log("  Warning: No cumulative energy data to plot")
            return None
        
        if output_path is None:
            output_path = self.output_path.parent / 'phase5_cumulative_energy.png'
        
        # Normalize for plotting
        energy = self.cumulative_energy / self.cumulative_energy.max()
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        im = ax.imshow(
            energy.T, origin='lower', cmap='hot',
            norm=Normalize(vmin=0, vmax=1), aspect='equal'
        )
        ax.set_title('Cumulative Field Energy Density (Time-Integrated |E|^2)', fontsize=12)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Energy Density', fontsize=10)
        plt.tight_layout()
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        log(f"  Cumulative energy plot saved: {output_path}")
        return output_path
    
    def get_cumulative_energy(self) -> Optional[np.ndarray]:
        """Return the cumulative energy array for external use."""
        return self.cumulative_energy


# ==============================================================================
# Meep Simulation Runner
# ==============================================================================

def estimate_simulation_resources(
    geo_ctx: Dict,
    config: SimulationConfig,
    phase1_meta: Dict
) -> Dict:
    """Estimate simulation time and memory requirements."""
    a = phase1_meta.get('a', 1.0)
    window_x, window_y = geo_ctx['window_size']
    
    # Grid size
    resolution = config.resolution_per_a
    nx = int(resolution * window_x / a)
    ny = int(resolution * window_y / a)
    
    # Add PML
    pml_pixels = int(config.pml_thickness * resolution)
    nx_total = nx + 2 * pml_pixels
    ny_total = ny + 2 * pml_pixels
    
    total_pixels = nx_total * ny_total
    
    # Estimate time steps
    # Courant factor: dt ≈ 0.5 * dx / c ≈ 0.5 / resolution
    dt = 0.5 / resolution
    total_time = (
        config.source_ramp_time +
        config.steady_state_time +
        config.measurement_delay +
        config.ringdown_time
    )
    n_timesteps = int(total_time / dt)
    
    # Memory estimate (rough)
    # Meep stores multiple field components, each complex
    bytes_per_pixel = 8 * 6  # ~6 field components, 8 bytes each
    base_memory_mb = (total_pixels * bytes_per_pixel) / (1024**2)
    
    # Video frames memory
    video_frames = min(
        int(total_time / config.video_capture_interval),
        config.max_video_frames
    )
    stride = config.field_snapshot_stride
    frame_pixels = (nx // stride) * (ny // stride)
    video_memory_mb = (video_frames * frame_pixels * 4) / (1024**2)  # float32
    
    # Time estimate (very rough: ~1000 timesteps/sec on good hardware)
    estimated_time_sec = n_timesteps / 500  # Conservative estimate
    
    return {
        'grid_size': (nx_total, ny_total),
        'total_pixels': total_pixels,
        'n_timesteps': n_timesteps,
        'dt': dt,
        'total_sim_time': total_time,
        'base_memory_mb': base_memory_mb,
        'video_memory_mb': video_memory_mb,
        'total_memory_mb': base_memory_mb + video_memory_mb,
        'estimated_time_sec': estimated_time_sec,
        'video_frames': video_frames,
    }


def run_meep_simulation(
    geo_ctx: Dict,
    phase1_meta: Dict,
    target_mode: pd.Series,
    cavity_pos: np.ndarray,
    config: SimulationConfig,
    cdir: Path,
    test_mode: bool = False,
    frequency_override: float = None
) -> SimulationMetrics:
    """
    Run the full Meep FDTD simulation with streaming.
    
    Phases:
    1. Ramp up continuous-wave source
    2. Steady state (collect energy data)
    3. Turn off source
    4. Measurement delay
    5. Ringdown (Harminv + energy decay)
    """
    metrics = SimulationMetrics()
    
    # Extract parameters
    a = phase1_meta.get('a', 1.0)
    eps_bg = geo_ctx['eps_bg']
    window_x, window_y = geo_ctx['window_size']
    
    # Target frequency
    if frequency_override is not None and frequency_override > 0:
        omega = frequency_override
        log(f"  Using frequency override: ω = {omega:.6f}")
    else:
        # Use Phase 3 eigenvalue (may have unit issues)
        omega_raw = float(target_mode.get('omega_cavity', target_mode.get('omega', 0.5)))
        omega = abs(omega_raw)
        if omega_raw < 0:
            log(f"  Warning: Negative frequency {omega_raw:.6f} from Phase 3, using |ω| = {omega:.6f}")
    metrics.omega_target = omega
    
    # Resolution and cell
    resolution = config.resolution_per_a
    pml_thickness = config.pml_thickness * a
    cell_x = window_x + 2 * pml_thickness
    cell_y = window_y + 2 * pml_thickness
    
    # Compute source timing
    source_end_time = config.source_ramp_time + config.steady_state_time
    total_time = config.total_time
    
    log(f"  Target frequency: ω = {omega:.6f}")
    log(f"  Resolution: {resolution} pixels/a")
    log(f"  Cell size: {cell_x:.2f} × {cell_y:.2f} a")
    log(f"  Total holes: {geo_ctx['n_bottom'] + geo_ctx['n_top']}")
    log(f"  Source ON: t=0 to t={source_end_time:.1f} (ramp={config.source_ramp_time}, steady={config.steady_state_time})")
    log(f"  Source OFF + ringdown: t={source_end_time:.1f} to t={total_time:.1f}")
    
    # Find safe source position (not inside a hole)
    source_pos = find_safe_source_position(cavity_pos, geo_ctx)
    log(f"  Cavity position: ({cavity_pos[0]:.3f}, {cavity_pos[1]:.3f})")
    log(f"  Source position: ({source_pos[0]:.3f}, {source_pos[1]:.3f})")
    
    # Create ROI for energy measurement
    roi_half = config.roi_fraction * min(window_x, window_y) / 2
    roi_bounds = (
        cavity_pos[0] - roi_half,
        cavity_pos[0] + roi_half,
        cavity_pos[1] - roi_half,
        cavity_pos[1] + roi_half
    )
    
    # Flux monitor positions (on ROI boundary)
    flux_positions = [
        np.array([roi_bounds[0], cavity_pos[1]]),  # left
        np.array([roi_bounds[1], cavity_pos[1]]),  # right
        np.array([cavity_pos[0], roi_bounds[2]]),  # bottom
        np.array([cavity_pos[0], roi_bounds[3]]),  # top
    ]
    
    # Generate pre-simulation plots
    log("  Generating pre-simulation plots...")
    
    # Compute when source should turn off: after ramp + steady_state
    source_end_time = config.source_ramp_time + config.steady_state_time
    
    # Build simulation object for geometry plot
    # IMPORTANT: ContinuousSource needs end_time to turn off!
    # Without end_time, it pumps energy forever → energy explodes
    sources = [mp.Source(
        mp.ContinuousSource(
            frequency=omega, 
            width=config.source_ramp_time,
            end_time=source_end_time  # Source turns OFF after ramp + steady state
        ),
        component=mp.Ez,
        center=mp.Vector3(source_pos[0], source_pos[1], 0),
    )]
    
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, 0),
        geometry=geo_ctx['geometry'],
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(thickness=pml_thickness)],
        default_material=mp.Medium(epsilon=eps_bg),
    )
    
    # Plot 1: MPB-style geometry (Meep's plot2D)
    plot_mpb_geometry(sim, cdir / 'phase5_geometry_meep.png')
    
    # Plot 2: Custom setup plot
    plot_simulation_setup(
        geo_ctx, cavity_pos, source_pos, roi_bounds,
        flux_positions, cdir / 'phase5_simulation_setup.png'
    )
    
    if test_mode:
        log("  TEST MODE: Skipping actual simulation")
        sim.reset_meep()
        return metrics
    
    # Initialize video streamer with timing info for phase labels
    stride = config.field_snapshot_stride
    frame_nx = int(resolution * window_x / a) // stride
    frame_ny = int(resolution * window_y / a) // stride
    timing_config = {
        'ramp': config.source_ramp_time,
        'steady': config.steady_state_time,
        'delay': config.measurement_delay,
        'ringdown': config.ringdown_time
    }
    video = VideoStreamer(
        cdir / 'phase5_simulation.mp4',
        (frame_nx, frame_ny),
        fps=config.video_fps,
        timing_config=timing_config,
        omega=omega
    )
    video.start()
    
    # Add flux monitors
    flux_size = roi_half * 0.1  # Small flux monitor
    flux_monitors = []
    for i, pos in enumerate(flux_positions):
        direction = mp.X if i < 2 else mp.Y
        fr = sim.add_flux(
            omega, 0, 1,
            mp.FluxRegion(
                center=mp.Vector3(pos[0], pos[1], 0),
                size=mp.Vector3(0 if direction == mp.X else flux_size,
                               flux_size if direction == mp.X else 0, 0),
                direction=direction
            )
        )
        flux_monitors.append(fr)
    
    # Harminv monitor at source position
    harminv = mp.Harminv(
        mp.Ez,
        mp.Vector3(source_pos[0], source_pos[1], 0),
        omega, 0.1  # frequency and bandwidth
    )
    
    # Streaming callbacks
    last_capture_time = [-1.0]  # Mutable for closure
    
    def capture_frame(sim_obj):
        t = sim_obj.meep_time()
        if t - last_capture_time[0] >= config.video_capture_interval:
            if len(metrics.time_points) < config.max_video_frames:
                # Get field in window region
                arr = sim_obj.get_array(
                    center=mp.Vector3(0, 0, 0),
                    size=mp.Vector3(window_x, window_y, 0),
                    component=mp.Ez
                )
                if arr is not None:
                    # Downsample
                    arr_down = arr[::stride, ::stride]
                    video.add_frame(arr_down, t)
                    
                    # Also compute energy
                    energy = np.sum(arr**2)
                    metrics.time_points.append(t)
                    metrics.field_energy.append(energy)
            
            last_capture_time[0] = t
    
    # Run simulation phases
    total_time = (
        config.source_ramp_time +
        config.steady_state_time +
        config.measurement_delay +
        config.ringdown_time
    )
    
    log(f"  Starting simulation (total time: {total_time:.1f})...")
    t_start = time.time()
    
    try:
        sim.run(
            mp.at_every(config.video_capture_interval, capture_frame),
            mp.after_sources(harminv),
            until=total_time
        )
    except KeyboardInterrupt:
        log("  Simulation interrupted!")
    
    t_elapsed = time.time() - t_start
    log(f"  Simulation completed in {t_elapsed:.1f} seconds")
    
    # Extract Harminv results
    for mode in harminv.modes:
        if mode.Q > 10:  # Filter noise
            metrics.harminv_modes.append({
                'freq': float(mode.freq),
                'Q': float(mode.Q),
                'decay': float(mode.decay),
            })
    
    if metrics.harminv_modes:
        # Find mode closest to target frequency
        freq_diffs = [abs(m['freq'] - omega) for m in metrics.harminv_modes]
        best_idx = np.argmin(freq_diffs)
        best_mode = metrics.harminv_modes[best_idx]
        metrics.Q_harminv = best_mode['Q']
        metrics.omega_measured = best_mode['freq']
        log(f"  Harminv Q = {metrics.Q_harminv:.1f} at ω = {metrics.omega_measured:.6f}")
    
    # Compute Q from energy decay
    if len(metrics.field_energy) > 10:
        metrics.Q_energy_decay = compute_Q_from_energy_decay(
            np.array(metrics.time_points),
            np.array(metrics.field_energy),
            omega
        )
        log(f"  Energy decay Q = {metrics.Q_energy_decay:.1f}")
    
    # Finalize video
    log(f"  Video frames captured: {video.frame_count}")
    if video.frame_count > 0:
        video.finalize()
        # Save cumulative energy plot (Phase 3-style)
        video.save_cumulative_energy_plot()
    else:
        log("  Warning: No video frames were captured!")
    
    # Cleanup
    sim.reset_meep()
    
    return metrics


def run_preliminary_simulation(
    cdir: Path,
    geo_ctx: Dict,
    config: SimulationConfig,
    phase1_meta: Dict,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    prelim_frequency: float,
    prelim_source_time: float,
    prelim_total_time: float,
    yaml_config: Dict
) -> bool:
    """
    Run a preliminary "counterpoint" simulation at a non-resonant frequency.
    
    This creates a comparison video showing what field propagation looks like
    when NOT at the cavity resonance. Useful for validating that localization
    in the main simulation is genuinely due to the cavity mode.
    
    Key differences from main simulation:
    - Different frequency (typically off-resonance)
    - No source ramp (immediate full-amplitude emission)
    - Shorter total duration
    - No Q-factor measurement
    
    Args:
        cdir: Candidate directory for output
        geo_ctx: Geometry context from build_bilayer_geometry
        config: SimulationConfig for resolution/stride/fps settings
        phase1_meta: Phase 1 metadata
        cavity_pos: Cavity center position
        source_pos: Source position
        prelim_frequency: CW source frequency for preliminary run
        prelim_source_time: How long the source emits
        prelim_total_time: Total simulation time
        yaml_config: Full config dict for threading setup
        
    Returns:
        True if successful, False otherwise
    """
    log("\n" + "-"*60)
    log("PRELIMINARY RUN (Counterpoint)")
    log("-"*60)
    log(f"  Frequency: {prelim_frequency:.6f} (c/a)")
    log(f"  Source time: {prelim_source_time:.1f}")
    log(f"  Total time: {prelim_total_time:.1f}")
    
    a = phase1_meta.get('a', 1.0)
    eps_bg = phase1_meta.get('eps_bg', 1.0)
    window_x, window_y = geo_ctx['window_size']
    
    # Resolution and cell
    resolution = config.resolution_per_a
    pml_thickness = config.pml_thickness * a
    cell_x = window_x + 2 * pml_thickness
    cell_y = window_y + 2 * pml_thickness
    
    # Source: NO RAMP - immediate full amplitude CW
    sources = [mp.Source(
        mp.ContinuousSource(
            frequency=prelim_frequency, 
            width=0,  # width=0 = no ramp
            end_time=prelim_source_time  # Source turns off after this time
        ),
        component=mp.Ez,
        center=mp.Vector3(source_pos[0], source_pos[1], 0),
    )]
    
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, 0),
        geometry=geo_ctx['geometry'],
        sources=sources,
        resolution=resolution,
        boundary_layers=[mp.PML(thickness=pml_thickness)],
        default_material=mp.Medium(epsilon=eps_bg),
    )
    
    # Initialize video streamer (shares FPS and stride from main config)
    stride = config.field_snapshot_stride
    frame_nx = int(resolution * window_x / a) // stride
    frame_ny = int(resolution * window_y / a) // stride
    
    # Timing config for phase labels
    timing_config = {
        'ramp': 0,  # No ramp
        'steady': prelim_source_time,  # Source on
        'delay': 0,
        'ringdown': prelim_total_time - prelim_source_time  # After source off
    }
    
    # Compute target frame count from total time
    target_frames = yaml_config.get('phase5_target_video_frames', 200)
    video_interval = prelim_total_time / max(target_frames, 1)
    
    video = VideoStreamer(
        cdir / 'phase5_prelim_simulation.mp4',
        (frame_nx, frame_ny),
        fps=config.video_fps,
        timing_config=timing_config,
        omega=prelim_frequency
    )
    video.start()
    
    # Frame capture callback
    last_capture_time = [-1.0]
    max_frames = config.max_video_frames
    
    def capture_frame(sim_obj):
        t = sim_obj.meep_time()
        if t - last_capture_time[0] >= video_interval:
            if video.frame_count < max_frames:
                arr = sim_obj.get_array(
                    center=mp.Vector3(0, 0, 0),
                    size=mp.Vector3(window_x, window_y, 0),
                    component=mp.Ez
                )
                if arr is not None:
                    arr_down = arr[::stride, ::stride]
                    video.add_frame(arr_down, t)
            last_capture_time[0] = t
    
    # Run simulation
    log(f"  Starting preliminary simulation...")
    t_start = time.time()
    
    try:
        sim.run(
            mp.at_every(video_interval, capture_frame),
            until=prelim_total_time
        )
    except KeyboardInterrupt:
        log("  Preliminary simulation interrupted!")
        sim.reset_meep()
        return False
    
    t_elapsed = time.time() - t_start
    log(f"  Preliminary simulation completed in {t_elapsed:.1f} seconds")
    
    # Finalize video
    log(f"  Preliminary video frames captured: {video.frame_count}")
    success = False
    if video.frame_count > 0:
        success = video.finalize()
        video.save_cumulative_energy_plot(cdir / 'phase5_prelim_cumulative_energy.png')
    
    sim.reset_meep()
    
    log("-"*60)
    log("PRELIMINARY RUN COMPLETE")
    log("-"*60)
    
    return success


def find_safe_source_position(cavity_pos: np.ndarray, geo_ctx: Dict) -> np.ndarray:
    """Find a position for the source that's not inside a hole."""
    radius = geo_ctx['radius']
    all_holes = np.vstack([geo_ctx['bottom_points'], geo_ctx['top_points']])
    
    def in_hole(pos):
        if len(all_holes) == 0:
            return False
        dists = np.linalg.norm(all_holes - pos, axis=1)
        return np.any(dists < radius)
    
    if not in_hole(cavity_pos):
        return cavity_pos.copy()
    
    # Search in expanding circles
    for r in np.linspace(radius, radius * 5, 20):
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            test_pos = cavity_pos + r * np.array([np.cos(angle), np.sin(angle)])
            if not in_hole(test_pos):
                return test_pos
    
    # Fallback: just offset from cavity
    return cavity_pos + np.array([radius * 2, 0])


def compute_Q_from_energy_decay(
    times: np.ndarray,
    energies: np.ndarray,
    omega: float
) -> float:
    """
    Compute Q from energy decay: U(t) ∝ exp(-2γt), Q = ω/(2γ).
    
    Uses linear fit to log(U) to find decay rate.
    """
    # Find peak and fit decay after it
    peak_idx = np.argmax(energies)
    if peak_idx >= len(energies) - 5:
        return float('nan')
    
    # Use data after peak
    t_decay = times[peak_idx:]
    e_decay = energies[peak_idx:]
    
    # Filter positive energies
    mask = e_decay > 0
    if mask.sum() < 5:
        return float('nan')
    
    t_fit = t_decay[mask]
    log_e = np.log(e_decay[mask])
    
    # Linear fit: log(E) = -2γt + const
    try:
        coeffs = np.polyfit(t_fit, log_e, 1)
        gamma_2 = -coeffs[0]  # -2γ
        if gamma_2 <= 0:
            return float('nan')
        gamma = gamma_2 / 2
        Q = omega / (2 * gamma)
        return Q
    except:
        return float('nan')


# ==============================================================================
# Results Saving and Summary
# ==============================================================================

def save_results(metrics: SimulationMetrics, cdir: Path, target_mode: pd.Series):
    """Save simulation results to files."""
    # Save metrics summary
    summary = {
        'omega_target': metrics.omega_target,
        'omega_measured': metrics.omega_measured,
        'Q_harminv': metrics.Q_harminv,
        'Q_energy_decay': metrics.Q_energy_decay,
        'Q_power_loss': metrics.Q_power_loss,
        'n_harminv_modes': len(metrics.harminv_modes),
        'harminv_modes': metrics.harminv_modes,
        'target_mode_index': int(target_mode.get('mode_index', 0)),
        'target_mode_spread': float(target_mode.get('spread', float('nan'))),
    }
    
    save_json(summary, cdir / 'phase5_results.json')
    log(f"  Saved results to phase5_results.json")
    
    # Save time series
    if len(metrics.time_points) > 0:
        df_ts = pd.DataFrame({
            'time': metrics.time_points,
            'field_energy': metrics.field_energy,
        })
        df_ts.to_csv(cdir / 'phase5_time_series.csv', index=False)
    
    # Save Harminv modes
    if metrics.harminv_modes:
        df_harm = pd.DataFrame(metrics.harminv_modes)
        df_harm.to_csv(cdir / 'phase5_harminv_modes.csv', index=False)


def print_summary(metrics: SimulationMetrics, resources: Dict):
    """Print summary of results."""
    log("\n" + "="*60)
    log("PHASE 5 RESULTS SUMMARY")
    log("="*60)
    
    log(f"  Target frequency: ω = {metrics.omega_target:.6f}")
    log(f"  Measured frequency: ω = {metrics.omega_measured:.6f}")
    
    freq_error = abs(metrics.omega_measured - metrics.omega_target) / metrics.omega_target * 100
    log(f"  Frequency error: {freq_error:.2f}%")
    
    log("\n  Q-Factor Estimates:")
    log(f"    Harminv:      Q = {metrics.Q_harminv:.1f}")
    log(f"    Energy decay: Q = {metrics.Q_energy_decay:.1f}")
    
    if len(metrics.harminv_modes) > 1:
        log(f"\n  All Harminv modes ({len(metrics.harminv_modes)} found):")
        for m in sorted(metrics.harminv_modes, key=lambda x: x['Q'], reverse=True)[:5]:
            log(f"    ω = {m['freq']:.6f}, Q = {m['Q']:.1f}")


# ==============================================================================
# Main Entry Points
# ==============================================================================

def run_test_mode(run_dir: Path, config_path: Path, config: Dict):
    """
    Run in test mode: generate plots, estimate resources, verify video.
    """
    log("\n" + "="*70)
    log("PHASE 5 V3 (Meep) — TEST MODE")
    log("="*70)
    
    # Find latest Phase 3 candidate
    candidates = list(run_dir.glob("candidate_*"))
    if not candidates:
        raise FileNotFoundError(f"No candidate directories in {run_dir}")
    
    cdir = sorted(candidates)[0]
    cid = int(cdir.name.split('_')[1])
    log(f"Using candidate {cid} from: {cdir}")
    
    # Load metadata
    phase0_meta = load_json(cdir / 'phase0_meta.json')
    
    # Check for Phase 3 data (may not exist yet)
    phase3_exists = has_phase3_results(cdir)
    
    # Build simulation config from YAML (handles threading and video timing)
    sim_config = build_simulation_config(config)
    
    # Load Phase 1 data for geometry
    phase1_meta = {}
    h5_path = cdir / 'phase1_band_data.h5'
    if h5_path.exists():
        with h5py.File(h5_path, 'r') as hf:
            for key in ['a', 'lattice_type', 'r_over_a', 'eps_bg', 'theta_deg', 'moire_length']:
                if key in hf.attrs:
                    val = hf.attrs[key]
                    phase1_meta[key] = val.decode() if isinstance(val, bytes) else val
    else:
        # Use phase0 meta
        phase1_meta = {
            'a': phase0_meta.get('a', 1.0),
            'lattice_type': phase0_meta.get('lattice_type', 'square'),
            'r_over_a': phase0_meta.get('r_over_a', 0.2),
            'eps_bg': phase0_meta.get('eps_bg', 6.0),
            'theta_deg': phase0_meta.get('theta_deg', 1.1),
            'moire_length': phase0_meta.get('moire_length', 50.0),
        }
    
    log(f"\nGeometry parameters:")
    log(f"  Lattice: {phase1_meta['lattice_type']}")
    log(f"  r/a = {phase1_meta['r_over_a']}, ε_bg = {phase1_meta['eps_bg']}")
    log(f"  θ = {phase1_meta['theta_deg']}°, L_m = {phase1_meta['moire_length']:.2f}a")
    
    # Build geometry
    geo_ctx = build_bilayer_geometry(
        lattice_type=phase1_meta['lattice_type'],
        theta_deg=phase1_meta['theta_deg'],
        a=phase1_meta['a'],
        r_over_a=phase1_meta['r_over_a'],
        eps_bg=phase1_meta['eps_bg'],
        supercell_tiles=sim_config.supercell_tiles,
        moire_length=phase1_meta['moire_length'],
    )
    
    log(f"\nGeometry built:")
    log(f"  Window: {geo_ctx['window_size'][0]:.2f} × {geo_ctx['window_size'][1]:.2f} a")
    log(f"  Holes: {geo_ctx['n_bottom']} (bottom) + {geo_ctx['n_top']} (top)")
    
    # Estimate resources
    resources = estimate_simulation_resources(geo_ctx, sim_config, phase1_meta)
    
    log(f"\n{'='*60}")
    log("RESOURCE ESTIMATES")
    log(f"{'='*60}")
    log(f"  Grid size: {resources['grid_size'][0]} × {resources['grid_size'][1]} pixels")
    log(f"  Total pixels: {resources['total_pixels']:,}")
    log(f"  Time steps: {resources['n_timesteps']:,}")
    log(f"  dt = {resources['dt']:.6f}")
    log(f"  Total simulation time: {resources['total_sim_time']:.1f}")
    log(f"\n  Memory estimates:")
    log(f"    Base (Meep): ~{resources['base_memory_mb']:.1f} MB")
    log(f"    Video frames: ~{resources['video_memory_mb']:.1f} MB")
    log(f"    Total: ~{resources['total_memory_mb']:.1f} MB")
    log(f"\n  Time estimate: ~{resources['estimated_time_sec']/60:.1f} minutes")
    log(f"  Video frames: {resources['video_frames']}")
    
    # Generate test plots
    log(f"\n{'='*60}")
    log("GENERATING TEST PLOTS")
    log(f"{'='*60}")
    
    # Mock target mode if Phase 3 doesn't exist
    if phase3_exists:
        df_modes, F_best = load_phase3_modes(cdir)
        target_mode = select_target_mode(df_modes, config)
    else:
        log("  Phase 3 data not found, using mock target mode")
        target_mode = pd.Series({
            'mode_index': 0,
            'omega_cavity': 0.5,
            'spread': 0.1,
        })
        F_best = None
    
    # Compute cavity position
    cavity_pos = compute_cavity_position(
        F_best,
        (128, 128),  # Assume standard grid
        geo_ctx['B_moire']
    )
    source_pos = find_safe_source_position(cavity_pos, geo_ctx)
    
    # ROI for test plot
    roi_half = sim_config.roi_fraction * min(*geo_ctx['window_size']) / 2
    roi_bounds = (
        cavity_pos[0] - roi_half,
        cavity_pos[0] + roi_half,
        cavity_pos[1] - roi_half,
        cavity_pos[1] + roi_half
    )
    flux_positions = [
        np.array([roi_bounds[0], cavity_pos[1]]),
        np.array([roi_bounds[1], cavity_pos[1]]),
        np.array([cavity_pos[0], roi_bounds[2]]),
        np.array([cavity_pos[0], roi_bounds[3]]),
    ]
    
    # Create minimal simulation for geometry plot
    omega_test = float(target_mode.get('omega_cavity', 0.5))
    sources = [mp.Source(
        mp.ContinuousSource(frequency=omega_test),
        component=mp.Ez,
        center=mp.Vector3(source_pos[0], source_pos[1], 0)
    )]
    
    pml_thickness = sim_config.pml_thickness * phase1_meta['a']
    cell_x = geo_ctx['window_size'][0] + 2 * pml_thickness
    cell_y = geo_ctx['window_size'][1] + 2 * pml_thickness
    
    # Use lower resolution for test
    test_resolution = 8
    
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, 0),
        geometry=geo_ctx['geometry'],
        sources=sources,
        resolution=test_resolution,
        boundary_layers=[mp.PML(thickness=pml_thickness)],
        default_material=mp.Medium(epsilon=geo_ctx['eps_bg']),
    )
    
    # Plot 1: Meep geometry
    plot_mpb_geometry(sim, cdir / 'phase5_test_geometry.png')
    
    # Plot 2: Custom setup
    plot_simulation_setup(
        geo_ctx, cavity_pos, source_pos, roi_bounds,
        flux_positions, cdir / 'phase5_test_setup.png'
    )
    
    # Test video generation
    log("\n  Testing video generation...")
    test_timing = {'ramp': 5, 'steady': 10, 'delay': 2, 'ringdown': 3}
    video = VideoStreamer(
        cdir / 'phase5_test_video.mp4',
        (50, 50),
        fps=10,
        timing_config=test_timing,
        omega=0.7
    )
    video.start()
    
    # Generate test frames
    for i in range(30):
        # Create test pattern
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        frame = np.sin(2*np.pi*(X + 0.1*i)) * np.exp(-(X**2 + Y**2))
        video.add_frame(frame, i * 0.5)
    
    success = video.finalize()
    if success:
        log("  Video generation test PASSED")
        video.save_cumulative_energy_plot()
    else:
        log("  Video generation test FAILED (ffmpeg may not be installed)")
    
    sim.reset_meep()
    
    log(f"\n{'='*60}")
    log("TEST MODE COMPLETE")
    log(f"{'='*60}")
    log(f"\nGenerated files in {cdir}:")
    log(f"  - phase5_test_geometry.png")
    log(f"  - phase5_test_setup.png")
    if success:
        log(f"  - phase5_test_video.mp4")


def run_phase5_v3(run_dir: Path, config_path: Path):
    """Main Phase 5 driver."""
    log("\n" + "="*70)
    log("PHASE 5 V3 (Meep): FDTD Validation & Q-Factor Analysis")
    log("="*70)
    
    config = load_yaml(config_path)
    log(f"Loaded config from: {config_path}")
    
    # Find candidates with Phase 3 results
    candidates = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        if has_phase3_results(cdir):
            cid = int(cdir.name.split('_')[1])
            candidates.append((cid, cdir))
    
    if not candidates:
        raise FileNotFoundError(
            f"No candidates with Phase 3 results found in {run_dir}. "
            "Run Phase 3 first."
        )
    
    log(f"Found {len(candidates)} candidates with Phase 3 results")
    
    # Process first candidate (or specified one)
    K_candidates = config.get('K_candidates', 1)
    for cid, cdir in candidates[:K_candidates]:
        log(f"\n{'='*60}")
        log(f"Processing Candidate {cid}")
        log(f"{'='*60}")
        
        try:
            process_candidate_phase5(cdir, config)
        except Exception as e:
            log(f"ERROR processing candidate {cid}: {e}")
            import traceback
            traceback.print_exc()


def process_candidate_phase5(cdir: Path, config: Dict):
    """Process a single candidate for Phase 5 validation."""
    # Load metadata from phase0
    phase0_meta = load_json(cdir / 'phase0_meta.json')
    
    # Start with phase0 meta as base
    phase1_meta = {
        'a': phase0_meta.get('a', 1.0),
        'lattice_type': phase0_meta.get('lattice_type', 'square'),
        'r_over_a': phase0_meta.get('r_over_a', 0.2),
        'eps_bg': phase0_meta.get('eps_bg', 6.0),
        'theta_deg': phase0_meta.get('theta_deg', 1.1),
        'moire_length': phase0_meta.get('moire_length', 50.0),
    }
    
    # Try to load from Phase 1 HDF5 (may have updated values)
    for h5_name in ['phase1_multiband_data.h5', 'phase1_band_data.h5']:
        h5_path = cdir / h5_name
        if h5_path.exists():
            try:
                with h5py.File(h5_path, 'r') as hf:
                    for key in ['a', 'lattice_type', 'r_over_a', 'eps_bg', 'theta_deg', 'moire_length']:
                        if key in hf.attrs:
                            val = hf.attrs[key]
                            phase1_meta[key] = val.decode() if isinstance(val, bytes) else val
                log(f"  Loaded geometry from {h5_name}")
                break
            except Exception as e:
                log(f"    Warning: Could not load {h5_name}: {e}")
    
    # Load Phase 3 modes
    df_modes, F_best = load_phase3_modes(cdir)
    target_mode = select_target_mode(df_modes, config)
    
    log(f"  Selected mode {int(target_mode.get('mode_index', 0))} with spread = {target_mode.get('spread', 'N/A')}")
    
    # Build simulation config from YAML (uses helper for threading + video timing)
    sim_config = build_simulation_config(config)
    
    # Compute cavity position FIRST (needed for centering)
    # We need the moiré basis, so compute it here
    theta_rad = math.radians(phase1_meta['theta_deg'])
    B_mono = build_monolayer_basis(phase1_meta['lattice_type'], phase1_meta['a'])
    B_moire = compute_moire_basis(B_mono, theta_rad)
    
    Ns1 = int(config.get('phase1_Ns1', 128))
    Ns2 = int(config.get('phase1_Ns2', 128))
    cavity_pos_raw = compute_cavity_position(F_best, (Ns1, Ns2), B_moire)
    
    log(f"  Raw cavity position: ({cavity_pos_raw[0]:.3f}, {cavity_pos_raw[1]:.3f})")
    
    # Build geometry centered around cavity position
    # This shifts ALL holes so the cavity ends up at (0,0)
    geo_ctx = build_bilayer_geometry(
        lattice_type=phase1_meta['lattice_type'],
        theta_deg=phase1_meta['theta_deg'],
        a=phase1_meta['a'],
        r_over_a=phase1_meta['r_over_a'],
        eps_bg=phase1_meta['eps_bg'],
        supercell_tiles=sim_config.supercell_tiles,
        moire_length=phase1_meta['moire_length'],
        center_offset=cavity_pos_raw,  # Center around cavity
    )
    
    # After centering, cavity is at origin
    cavity_pos = np.array([0.0, 0.0])
    log(f"  Cavity centered at origin (0, 0)")
    
    # Resource estimate
    resources = estimate_simulation_resources(geo_ctx, sim_config, phase1_meta)
    log(f"\n  Grid: {resources['grid_size'][0]}×{resources['grid_size'][1]}, "
        f"{resources['n_timesteps']:,} timesteps")
    log(f"  Estimated memory: {resources['total_memory_mb']:.1f} MB")
    log(f"  Estimated time: {resources['estimated_time_sec']/60:.1f} min")
    
    # Get frequency override if specified
    frequency_override = config.get('phase5_frequency_override', None)
    
    # Check for preliminary run configuration
    prelim_frequency = config.get('phase5_prelim_frequency', None)
    if prelim_frequency is not None and prelim_frequency > 0:
        prelim_source_time = config.get('phase5_prelim_source_time', 50.0)
        prelim_total_time = config.get('phase5_prelim_total_time', 100.0)
        
        # Compute source position for preliminary run
        source_pos = find_safe_source_position(cavity_pos, geo_ctx)
        
        run_preliminary_simulation(
            cdir=cdir,
            geo_ctx=geo_ctx,
            config=sim_config,
            phase1_meta=phase1_meta,
            cavity_pos=cavity_pos,
            source_pos=source_pos,
            prelim_frequency=prelim_frequency,
            prelim_source_time=prelim_source_time,
            prelim_total_time=prelim_total_time,
            yaml_config=config
        )
    
    # Run main simulation
    metrics = run_meep_simulation(
        geo_ctx, phase1_meta, target_mode, cavity_pos,
        sim_config, cdir, test_mode=False,
        frequency_override=frequency_override
    )
    
    # Save results
    save_results(metrics, cdir, target_mode)
    print_summary(metrics, resources)


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase5_mpb.yaml"


def resolve_run_dir(run_dir_arg: str, config: Dict) -> Path:
    """Resolve run directory from argument or find latest."""
    if run_dir_arg in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runsV3'))
        # Look for Phase 0 runs
        phase0_runs = sorted(runs_base.glob('phase0_mpb_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No MPB phase0 runs found in {runs_base}")
        return phase0_runs[-1]
    return Path(run_dir_arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 (Meep): FDTD Validation & Q-Factor Analysis"
    )
    parser.add_argument(
        'run_dir', nargs='?', default='auto',
        help="Run directory (or 'auto' for latest)"
    )
    parser.add_argument(
        'config', nargs='?', default=None,
        help="Config file path"
    )
    parser.add_argument(
        '--test', action='store_true',
        help="Test mode: generate plots, estimate resources, verify video"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config) if args.config else get_default_config_path()
    if not config_path.exists():
        # Create default config
        log(f"Config not found at {config_path}, using defaults")
        config = {}
    else:
        config = load_yaml(config_path)
    
    # Resolve run directory
    run_dir = resolve_run_dir(args.run_dir, config)
    
    if args.test:
        run_test_mode(run_dir, config_path, config)
    else:
        run_phase5_v3(run_dir, config_path)
