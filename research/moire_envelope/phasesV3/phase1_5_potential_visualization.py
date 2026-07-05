import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import h5py
import json
from pathlib import Path
import argparse
import sys
import math
from scipy.ndimage import map_coordinates

# Add project root to path for common imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "research/moire_envelope"))

try:
    from common.plotting import MONO_BLUE, LAYER_TWO_ORANGE, MOIRE_PURPLE
except ImportError:
    MONO_BLUE = 'blue'
    LAYER_TWO_ORANGE = 'orange'
    MOIRE_PURPLE = 'purple'

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3)/2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # The Moiré lattice vectors (AA centers) are given by r such that (1 - R^-1)r = R_lattice
    # Formula: A_moire = (I - R(-theta))^-1 @ B_mono
    R_inv = np.array([[c, s], [-s, c]]) # Rotation by -theta
    return np.linalg.inv(np.eye(2) - R_inv) @ B_mono

def load_data(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    
    data = {}
    with h5py.File(file_path, "r") as f:
        data["s_grid"] = f["s_grid"][:]
        # Transpose V to (Nbands, Ns1, Ns2) for faster access
        V_in = f["V"][:] 
        data["V"] = np.moveaxis(V_in, -1, 0)
        
    meta_path = file_path.parent / "phase0_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data["meta"] = json.load(f)
        except:
             data["meta"] = {}
    else:
        data["meta"] = {"lattice_type": "hex", "a": 1.0}
    
    if "lattice_type" not in data["meta"]: data["meta"]["lattice_type"] = "hex"
    if "a" not in data["meta"]: data["meta"]["a"] = 1.0
        
    return data

class InteractiveMoirePlot:
    def __init__(self, data):
        self.data = data
        self.meta = data["meta"]
        self.V_grid = data["V"] # (Nbands, Ns1, Ns2)
        self.n_bands, self.Ns1, self.Ns2 = self.V_grid.shape
        
        self.lattice_type = self.meta.get("lattice_type", "hex")
        self.a = float(self.meta.get("a", 1.0))
        self.B_mono = build_monolayer_basis(self.lattice_type, self.a)
        
        # Determine which band to feature in the big plot (center of subspace)
        self.feature_band_idx = self.n_bands // 2
        
        # Pre-compute grid for V(s) plots
        self.extent_s = [0, 1, 0, 1]
        
        # Setup plot
        self.theta_deg = 1.1
        self.setup_plot()
        self.update_plots(self.theta_deg)

    def setup_plot(self):
        # Layout: 
        # Left side: n_bands columns for V(s) and V(R) small plots
        # Right side: Large plot spanning both rows
        # We'll assign relative width ratios: 1 unit per small band col, 
        # and say 3 units for the big plot.
        
        total_cols = self.n_bands + 1
        width_ratios = [1] * self.n_bands + [3]
        
        self.fig = plt.figure(figsize=(3 * self.n_bands + 6, 8))
        gs = self.fig.add_gridspec(2, total_cols, width_ratios=width_ratios,
                                  wspace=0.4, hspace=0.3, 
                                  bottom=0.15, top=0.9, left=0.05, right=0.95)
        
        self.axes_Vs = []
        self.imgs_Vs = []
        self.imgs_VR = []
        self.axes_VR = []
        
        # Row 1: V(s) - Small Plots
        for b in range(self.n_bands):
            ax = self.fig.add_subplot(gs[0, b])
            im = ax.imshow(
                self.V_grid[b].T, 
                origin='lower', cmap='viridis', 
                extent=self.extent_s, aspect='equal'
            )
            ax.set_title(f"Band {b+1} V(s)")
            if b == 0: ax.set_ylabel("s2")
            ax.set_xlabel("s1")
            self.axes_Vs.append(ax)
            self.imgs_Vs.append(im)
            
        # Row 2: V(R) - Small Plots
        self.N_pts_small = 100
        for b in range(self.n_bands):
            ax = self.fig.add_subplot(gs[1, b])
            im = ax.imshow(
                np.zeros((self.N_pts_small, self.N_pts_small)), 
                origin='lower', cmap='viridis',
                aspect='equal'
            )
            ax.set_title(f"Band {b+1} V(R)")
            if b == 0: ax.set_ylabel("y (a)")
            ax.set_xlabel("x (a)")
            self.axes_VR.append(ax)
            self.imgs_VR.append(im)

        # Big Plot: Moiré Crystal Geometry
        self.ax_big = self.fig.add_subplot(gs[:, -1])
        self.ax_big.set_aspect('equal')
        self.ax_big.set_title("Moiré Crystal Geometry")
        self.ax_big.set_xlabel("x (a)")
        self.ax_big.set_ylabel("y (a)")
        
        # Geometry: Scatter plots for Layer 1 and Layer 2 holes (zorder=2 to sit on top of potential if desired, 
        # or below? User wants "overlay" potential on geometry. Usually geometry is the reference.
        # Let's put Holes (z=0) and Potential (z=1, alpha < 1).
        
        # Fixed Layer 1 
        self.scat_L1 = self.ax_big.scatter([], [], s=1, c='#444444', alpha=0.6, label='Layer 1', zorder=1)
        # Twisted Layer 2
        self.scat_L2 = self.ax_big.scatter([], [], s=1, c='#444444', alpha=0.6, label='Layer 2', zorder=1)
        # Moiré Points (Orange dots)
        self.scat_moire = self.ax_big.scatter([], [], s=1, c='orange', alpha=1.0, label='Moiré', zorder=10)
        
        # Potential Overlay (zorder=2)
        # Initialize empty image.
        self.N_pts_big = 300
        self.img_overlay = self.ax_big.imshow(
            np.zeros((self.N_pts_big, self.N_pts_big)),
            origin='lower', cmap='viridis', alpha=0.0, zorder=2,
            aspect='equal'
        )
        self.overlay_band_idx = None # None means no overlay
        
        # Quiver for lattice vectors
        self.quiver = None

        # Slider for Twist Angle
        ax_slider = self.fig.add_axes([0.25, 0.08, 0.45, 0.03])
        self.slider = Slider(
            ax=ax_slider, label='Twist Angle (°)',
            valmin=0.1, valmax=5.0, valinit=self.theta_deg, valstep=0.1
        )
        self.slider.on_changed(self.update_plots)

        # Slider for Overlay Transparency
        ax_slider_alpha = self.fig.add_axes([0.25, 0.03, 0.45, 0.03])
        self.slider_alpha = Slider(
            ax=ax_slider_alpha, label='Overlay Alpha',
            valmin=0.0, valmax=1.0, valinit=0.4, valstep=0.05
        )
        self.slider_alpha.on_changed(self.update_alpha)
        
        # Radio Buttons for Band Overlay
        ax_radio = self.fig.add_axes([0.80, 0.05, 0.15, 0.10])
        labels = [f"Band {b+1}" for b in range(self.n_bands)] + ["None"]
        self.radio = RadioButtons(ax_radio, labels, active=self.n_bands) # Default to None
        self.radio.on_clicked(self.set_overlay_band)
        
    def set_overlay_band(self, label):
        if label == "None":
            self.overlay_band_idx = None
        else:
            # "Band X" -> index X-1
            self.overlay_band_idx = int(label.split()[1]) - 1
        self.update_plots(None)
        
    def update_alpha(self, val):
        if self.overlay_band_idx is not None:
             self.img_overlay.set_alpha(val)
             self.fig.canvas.draw_idle()

    def _generate_VR_image(self, b_idx, n_pts, R_span, B_moire_inv):
        """Helper to generate V(R) image data efficiently."""
        # 1. Coordinate Grid
        r_vec = np.linspace(-R_span/2, R_span/2, n_pts)
        
        # 2. Vectorized transform s = B_inv @ x
        X, Y = np.meshgrid(r_vec, r_vec) 
        
        # s1 = M00 X + M01 Y
        s1_map = B_moire_inv[0,0] * X + B_moire_inv[0,1] * Y
        s2_map = B_moire_inv[1,0] * X + B_moire_inv[1,1] * Y
        
        s1_map = np.mod(s1_map, 1.0)
        s2_map = np.mod(s2_map, 1.0)
        
        # 3. Map to pixel coords for V_grid (Ns1, Ns2)
        coords = np.stack([
            s1_map * (self.Ns1 - 1), 
            s2_map * (self.Ns2 - 1)
        ])
        
        # 4. Interpolate
        return map_coordinates(
            self.V_grid[b_idx], coords, 
            order=1, mode='wrap', prefilter=False
        )
        
    def _get_lattice_points_in_view(self, B, R_span):
        """Generate lattice points n1*b1 + n2*b2 within [-R_span/2, R_span/2]."""
        # Estimate index range
        # Bounds: max projection of B on axes
        # Safe over-estimation: R_span / min_norm
        # Lattice constant a=1 approx.
        N_max = int(np.ceil(R_span * 0.8)) # Heuristic factor
        
        n_vec = np.arange(-N_max, N_max + 1)
        N1, N2 = np.meshgrid(n_vec, n_vec)
        N1_flat = N1.flatten()
        N2_flat = N2.flatten()
        
        # Points P = N1*b1 + N2*b2
        # P = B @ [N1, N2]
        # B is (2,2) with columns b1, b2
        
        # P_x = B00 N1 + B01 N2
        # P_y = B10 N1 + B11 N2
        Px = B[0,0]*N1_flat + B[0,1]*N2_flat
        Py = B[1,0]*N1_flat + B[1,1]*N2_flat
        
        # Filter to viewport square
        mask = (np.abs(Px) <= R_span/2) & (np.abs(Py) <= R_span/2)
        return Px[mask], Py[mask]

    def update_plots(self, val):
        theta_deg = self.slider.val
        theta_rad = np.radians(theta_deg)
        
        B_moire = compute_moire_basis(self.B_mono, theta_rad)
        B_moire_inv = np.linalg.inv(B_moire)
        
        # --- Update Small Plots (View: 1.2 Moiré Periods) ---
        Lm = self.a / (2 * np.sin(theta_rad/2))
        R_span_small = 1.2 * Lm
        
        for b in range(self.n_bands):
            V_img = self._generate_VR_image(b, self.N_pts_small, R_span_small, B_moire_inv)
            
            self.imgs_VR[b].set_data(V_img)
            self.imgs_VR[b].set_extent([-R_span_small/2, R_span_small/2, 
                                      -R_span_small/2, R_span_small/2])
            
            vmin, vmax = np.min(self.V_grid[b]), np.max(self.V_grid[b])
            self.imgs_VR[b].set_clim(vmin, vmax)
            self.imgs_Vs[b].set_clim(vmin, vmax)
            
        # --- Update Big Plot: Moiré Geometry ---
        # Viewport: 3.0 Moiré periods (Zoomed out as requested)
        R_span_big = 3.0 * Lm
        self.ax_big.set_xlim(-R_span_big/2, R_span_big/2)
        self.ax_big.set_ylim(-R_span_big/2, R_span_big/2)
        
        # Layer 1 Basis (Fixed)
        B1 = self.B_mono
        # Layer 2 Basis (Twisted)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        rot = np.array([[c, -s], [s, c]])
        B2 = rot @ self.B_mono
        
        # Generate Points
        px1, py1 = self._get_lattice_points_in_view(B1, R_span_big)
        px2, py2 = self._get_lattice_points_in_view(B2, R_span_big)
        px_moire, py_moire = self._get_lattice_points_in_view(B_moire, R_span_big)
        
        # Update Scatter (Holes)
        # Scale hole size with zoom level. 
        hole_size = 20000.0 / (R_span_big**2) # Heuristic
        self.scat_L1.set_offsets(np.column_stack([px1, py1]))
        self.scat_L1.set_sizes([hole_size]*len(px1))
        
        self.scat_L2.set_offsets(np.column_stack([px2, py2]))
        self.scat_L2.set_sizes([hole_size]*len(px2))
        
        self.scat_moire.set_offsets(np.column_stack([px_moire, py_moire]))
        self.scat_moire.set_sizes([100]*len(px_moire)) # Fixed large size (no scaling)

        # Update Overlay Image (if selected)
        if self.overlay_band_idx is not None:
            V_overlay = self._generate_VR_image(self.overlay_band_idx, self.N_pts_big, R_span_big, B_moire_inv)
            self.img_overlay.set_data(V_overlay)
            self.img_overlay.set_extent([-R_span_big/2, R_span_big/2, 
                                       -R_span_big/2, R_span_big/2])
            self.img_overlay.set_alpha(self.slider_alpha.val) 
            # Normalize to band distinct range
            self.img_overlay.set_clim(
                np.min(self.V_grid[self.overlay_band_idx]), 
                np.max(self.V_grid[self.overlay_band_idx])
            )
        else:
            # Hide if None
            self.img_overlay.set_data(np.zeros((2,2))) # Dummy
            self.img_overlay.set_alpha(0.0)

        # Draw Moiré Lattice Vectors
        if self.quiver is not None:
             self.quiver.remove()
        
        vectors = B_moire 
        self.quiver = self.ax_big.quiver(
            [0, 0], [0, 0], 
            vectors[0, :], vectors[1, :],
            color='red', angles='xy', scale_units='xy', scale=1,
            width=0.015, headwidth=4, zorder=10
        )
        
        self.fig.suptitle(
            f"Moiré Potentials @ θ = {theta_deg:.1f}° ($L_m$ ≈ {Lm:.1f} a)\n"
            f"Left: Potential V(R) | Right: Twisted Crystal Geometry"
        )
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5: V3 Potential Visualization")
    parser.add_argument("args", nargs="*", help="[file_path] OR [run_dir] [candidate_id]")
    args = parser.parse_args()
    
    file_path = None
    
    # Logic to resolve file path from arguments
    if len(args.args) == 0:
        # No args: auto-select latest
        base_dir = Path(__file__).resolve().parents[1] / "runsV3"
        candidates = sorted(base_dir.glob("**/phase1_multiband_data.h5"))
        if candidates:
            file_path = candidates[-1]
            print(f"Auto-selected file: {file_path}")
    elif len(args.args) == 1:
        # One arg: direct file path, or run_dir (auto-select candidate 0?)
        arg = args.args[0]
        if arg.endswith('.h5'):
            file_path = Path(arg)
        else:
             # Assume it's a run dir
            run_dir = arg
            runs_base = Path(__file__).resolve().parents[1] / "runsV3"
            if not Path(run_dir).exists() and (runs_base / run_dir).exists():
                run_dir = runs_base / run_dir
            
            # Find first candidate in this run
            candidates = sorted(Path(run_dir).glob("candidate_*/phase1_multiband_data.h5"))
            if candidates:
                file_path = candidates[0]
                print(f"Auto-selected first candidate in run: {file_path}")
            else:
                 print(f"No phase1_multiband_data.h5 found in {run_dir}")
                 sys.exit(1)
                 
    elif len(args.args) >= 2:
        # Two args: [run_dir] [candidate_id] (any order)
        arg1, arg2 = args.args[0], args.args[1]
        
        run_dir = None
        candidate_id = None
        
        # Try to parse candidate ID from arg2
        try:
            candidate_id = int(arg2)
            run_dir = arg1
        except ValueError:
            # Maybe arg1 is the candidate ID?
            try:
                candidate_id = int(arg1)
                run_dir = arg2
            except ValueError:
                # Neither is an int, maybe a path?
                pass
        
        if run_dir and candidate_id is not None:
             runs_base = Path(__file__).resolve().parents[1] / "runsV3"
             if not Path(run_dir).exists() and (runs_base / run_dir).exists():
                run_dir = runs_base / run_dir
             
             file_path = Path(run_dir) / f"candidate_{candidate_id}" / "phase1_multiband_data.h5"
        else:
            # Fallback: maybe given as separate paths
            print("Could not parse arguments. Usage: python phase1_5.py [run_dir] [candidate_id]")
            sys.exit(1)

    if file_path is None or not file_path.exists():
        if file_path:
            print(f"Error: File {file_path} not found.")
        else:
             base_dir = Path(__file__).resolve().parents[1] / "runsV3"
             print(f"No data file found in {base_dir}")
        sys.exit(1)
            
    data = load_data(file_path)
    app = InteractiveMoirePlot(data)
    plt.show()
