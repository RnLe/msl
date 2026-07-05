import math
import os
import sys

# Backend fallback (avoid Qt error if no Wayland/X11 Qt plugin)
import matplotlib
try:
    # If a Qt backend was auto-selected but plugins missing, force Agg fallback early
    if matplotlib.get_backend().lower().startswith("qt"):
        matplotlib.use("Agg")  # non-interactive fallback
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import moire_lattice_py as ml
import numpy as np

# Optional rich logging -----------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
except ImportError:  # Fallback: simple print
    class _Fallback:
        def log(self, *a, **k):
            print(*a)
        def rule(self, *a, **k):
            print("-" * 60, *a)
        def print(self, *a, **k):
            print(*a)
    console = _Fallback()

# --- Parameters ---
# Twist angle (degrees)
angle_deg = 5.0  # updated from 2.0 to 5.0 per latest request
angle = math.radians(angle_deg)

a = 1.0  # monolayer lattice constant

#############################
# Multi-lattice generation  #
#############################

# Lattice specifications: (display_name, slug, constructor callable returning PyLattice2D)
def _make_specs(a):
    return [
        ("Triangular (Hexagonal)", "triangular", lambda: ml.create_hexagonal_lattice(a)),
        ("Square", "square", lambda: ml.create_square_lattice(a)),
        ("Rectangular 1:2", "rectangular", lambda: ml.create_rectangular_lattice(a, 2.0 * a)),
        ("Oblique", "oblique", lambda: ml.oblique_lattice_create(a, 1.3 * a, 75.0)),  # angle = 75°
    ]

specs = _make_specs(a)

# Defensive: ensure required binding functions exist
missing = []
required_funcs = [
    'py_registry_centers',
    'py_local_cells_preliminary',
    'py_local_cell_at_point_preliminary',
]
for fn in required_funcs:
    if not hasattr(ml, fn):
        missing.append(fn)
if missing:
    console.print(f"[bold red]ERROR[/] Missing exported functions: {missing}")
    sys.exit(1)

# Global shift d0 = (0,0)
d0x = 0.0
d0y = 0.0

def process_lattice(name: str, slug: str, lat):
    console.rule(f"Lattice: {name}")
    moire = ml.py_twisted_bilayer(lat, angle)
    console.log(f"Twist angle (deg): {moire.twist_angle_degrees():.4f}")

    centers = ml.py_registry_centers(moire, d0x, d0y)
    local_cells = ml.py_local_cells_preliminary(moire, d0x, d0y)
    console.log(f"Centers: {len(centers)} | Local cells: {len(local_cells)}")

    base_v1, base_v2 = lat.primitive_vectors()
    center_points = []
    labels = []
    for c in centers:
        labels.append(c['label'])
        px, py, _ = c['position']
        center_points.append((px, py))

    # (Rich table display omitted in multi-lattice batch for brevity)

    fig, (ax_moire, ax_cells) = plt.subplots(2, 1, figsize=(13, 9.5))

    # --- Plot 1: Underlying layer lattices + registry centers ---
    ax_moire.set_title(f"{name} layer lattices & registries (angle={angle_deg}°)")
    layer1 = moire.lattice_1()
    layer2 = moire.lattice_2()
    plot_radius = 25.0
    try:
        pts1 = layer1.generate_points(plot_radius)
        pts2 = layer2.generate_points(plot_radius)
    except Exception as e:
        console.log(f"[red]Failed to generate lattice points:[/] {e}")
        pts1, pts2 = [], []
    if pts1:
        ax_moire.scatter([p[0] for p in pts1], [p[1] for p in pts1], c='#1f77b4', s=8, zorder=1)
    if pts2:
        ax_moire.scatter([p[0] for p in pts2], [p[1] for p in pts2], c='#ff7f0e', s=8, zorder=1)
    xs = [p[0] for p in center_points]
    ys = [p[1] for p in center_points]
    ax_moire.scatter(xs, ys, c='tab:red', edgecolor='k', linewidths=0.3, s=65, zorder=4)
    for (x, y), lbl in zip(center_points, labels):
        ax_moire.text(x, y, lbl, fontsize=8, ha='center', va='bottom')
    ax_moire.set_aspect('equal')
    ax_moire.set_xlim(min(xs) - 5, max(xs) + 5)
    ax_moire.set_ylim(min(ys) - 5, max(ys) + 5)
    ax_moire.set_xlabel('x')
    ax_moire.set_ylabel('y')

    # --- Plot 2: Local preliminary unit cells (zoomed) ---
    ax_cells.set_title(f"{name} local preliminary cells (zoomed)")
    if pts1:
        ax_cells.scatter([p[0] for p in pts1], [p[1] for p in pts1], c='lightgray', s=6, zorder=0)
    if pts2:
        ax_cells.scatter([p[0] for p in pts2], [p[1] for p in pts2], c='gainsboro', s=6, zorder=0)
    for lbl, cell in local_cells.items():
        basis = cell['basis']
        if len(basis) < 2:
            continue
        tau = basis[1]
        c_obj = next(c for c in centers if c['label'] == lbl)
        cx, cy, _ = c_obj['position']
        v1 = base_v1
        v2 = base_v2
        p0 = (cx, cy)
        p1 = (cx + v1[0], cy + v1[1])
        p2 = (cx + v1[0] + v2[0], cy + v1[1] + v2[1])
        p3 = (cx + v2[0], cy + v2[1])
        poly_x = [p0[0], p1[0], p2[0], p3[0], p0[0]]
        poly_y = [p0[1], p1[1], p2[1], p3[1], p0[1]]
        ax_cells.plot(poly_x, poly_y, '-', lw=1.0, color='dimgray', zorder=2)
        ax_cells.scatter([cx], [cy], c='black', s=28, zorder=3)
        ax_cells.scatter([cx + tau[0]], [cy + tau[1]], c='orange', s=28, zorder=3)
        ax_cells.text(cx, cy, lbl, fontsize=7, ha='center', va='bottom', color='blue', zorder=4)

    console.rule("Rendering stacked plots")
    ax_cells.set_aspect('equal')
    ax_cells.set_xlabel('x')
    ax_cells.set_ylabel('y')
    xs_cells = [c['position'][0] for c in centers]
    ys_cells = [c['position'][1] for c in centers]
    pad = 2.5 * max(
        math.sqrt(base_v1[0]**2 + base_v1[1]**2),
        math.sqrt(base_v2[0]**2 + base_v2[1]**2)
    )
    ax_cells.set_xlim(min(xs_cells) - pad, max(xs_cells) + pad)
    ax_cells.set_ylim(min(ys_cells) - pad, max(ys_cells) + pad)
    plt.tight_layout()
    out_path_png = f"moire_local_cells_demo_{slug}.png"
    try:
        fig.savefig(out_path_png, dpi=400)
        console.log(f"Saved stacked figure to {out_path_png}")
    except Exception as e:
        console.log(f"[red]Failed to save stacked figure for {slug}:[/] {e}")
    plt.close(fig)

    # ---- Per-local-cell basis lattice figure (restored) ----
    n_cells = len(local_cells)
    if n_cells > 0:
        cols = min(3, n_cells)
        rows = (n_cells + cols - 1) // cols
        fig_cells, axs_cells = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows))
        # Normalize axs list
        if isinstance(axs_cells, np.ndarray):
            axs_iter = axs_cells.flatten()
        else:
            axs_iter = [axs_cells]
        base_v1_local = base_v1
        base_v2_local = base_v2
        for ax_i, (lbl, cell) in zip(axs_iter, local_cells.items()):
            basis = cell['basis']
            if len(basis) < 2:
                ax_i.set_axis_off()
                continue
            tau = basis[1]
            patch_r = 2
            R_x, R_y, T_x, T_y = [], [], [], []
            for i in range(-patch_r, patch_r + 1):
                for j in range(-patch_r, patch_r + 1):
                    xR = i * base_v1_local[0] + j * base_v2_local[0]
                    yR = i * base_v1_local[1] + j * base_v2_local[1]
                    R_x.append(xR)
                    R_y.append(yR)
                    T_x.append(xR + tau[0])
                    T_y.append(yR + tau[1])
            ax_i.scatter(R_x, R_y, c='black', s=10, zorder=1)
            ax_i.scatter(T_x, T_y, c='orange', s=10, zorder=1)
            # Primitive cell at origin
            p0 = (0.0, 0.0)
            p1 = (base_v1_local[0], base_v1_local[1])
            p2 = (base_v1_local[0] + base_v2_local[0], base_v1_local[1] + base_v2_local[1])
            p3 = (base_v2_local[0], base_v2_local[1])
            poly_x = [p0[0], p1[0], p2[0], p3[0], p0[0]]
            poly_y = [p0[1], p1[1], p2[1], p3[1], p0[1]]
            ax_i.plot(poly_x, poly_y, '-', lw=1.0, color='dimgray')
            ax_i.scatter([0.0], [0.0], c='black', s=34, zorder=3)
            ax_i.scatter([tau[0]], [tau[1]], c='orange', s=34, zorder=3)
            ax_i.set_title(lbl, fontsize=9)
            ax_i.set_aspect('equal')
            ax_i.set_xticks([])
            ax_i.set_yticks([])
        # Turn off any leftover axes
        for ax_extra in axs_iter[n_cells:]:
            ax_extra.set_axis_off()
        fig_cells.tight_layout()
        out_cells_png = f"moire_local_cells_local_bases_{slug}.png"
        try:
            fig_cells.savefig(out_cells_png, dpi=320)
            console.log(f"Saved per-local-cell basis figure to {out_cells_png}")
        except Exception as e:
            console.log(f"[red]Failed to save per-cell figure for {slug}:[/] {e}")
        plt.close(fig_cells)

# Execute processing for each lattice spec
for name, slug, ctor in specs:
    try:
        process_lattice(name, slug, ctor())
    except Exception as e:
        console.log(f"[red]Failed processing lattice {slug}:[/] {e}")

 # (Interactive display disabled for batch multi-lattice run unless forced)
if os.environ.get("ML_SHOW", "0") == "1":
    try:
        plt.show()
    except Exception as e:
        console.log(f"[yellow]Interactive show failed:[/] {e}")
