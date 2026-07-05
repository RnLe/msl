"""Animate a twisted bilayer triangular (hexagonal) lattice using moire_lattice_py.

This script renders a matplotlib animation showing:
- Left: the moiré superlattice points (zoomed out) as the twist angle increases.
- Right: the base triangular lattice with its primitive vectors and unit cell.
- Bottom right: live annotations for twist angle and moiré lattice constant relative to the base lattice constant.

The resulting animation is saved as a GIF next to this script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.colors import ListedColormap
import numpy as np

import moire_lattice_py as ml


@dataclass
class AnimationConfig:
    """Configuration parameters for the animation."""

    lattice_constant: float = 1.0
    max_angle_deg: float = 10.0
    min_angle_deg: float = 1e-3
    duration_seconds: float = 45.0
    interval_ms: int = 33
    growth_rate: float = 5.0
    moire_radius: float = 30.0
    base_axis_limit: float = 1.2
    max_moire_points: int = 2500
    max_layer_points: int = 2000
    output_filename: str = "twisted_triangular_moire.gif"
    debug: bool = False


def _array_from_points(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2))
    return np.asarray(points)


def _build_base_lattice(
    ax,
    lattice: ml.PyLattice2D,
    lattice_constant: float,
    axis_limit: float,
    *,
    arrow_lw: float = 5.0,
    tick_labelsize: int = 20,
    label_fontsize: int = 30,
    center_size: float = 60.0,
    neighbor_size: float = 45.0,
    show_unit_cell_outline: bool = True,
) -> None:
    """Plot the monolayer lattice highlighting the central site and first coordination shell.

    Parameters customize styling for export without affecting animation defaults.
    """
    raw_points = lattice.generate_points(axis_limit + lattice_constant)
    if raw_points:
        points = np.asarray(raw_points, dtype=float)
        radii = np.linalg.norm(points, axis=1)
    else:
        points = np.empty((0, 2))
        radii = np.empty((0,))

    center_mask = radii < 1e-8
    neighbor_mask = (radii > 0.85 * lattice_constant) & (radii < 1.15 * lattice_constant)

    center = points[center_mask]
    neighbors = points[neighbor_mask]
    if neighbors.size:
        order = np.argsort(np.arctan2(neighbors[:, 1], neighbors[:, 0]))
        neighbors = neighbors[order]

    a1, a2 = lattice.lattice_vectors()
    unit_cell = np.array(
        [
            [0.0, 0.0],
            [a1[0], a1[1]],
            [a1[0] + a2[0], a1[1] + a2[1]],
            [a2[0], a2[1]],
            [0.0, 0.0],
        ]
    )

    if center.size:
        ax.scatter(center[:, 0], center[:, 1], c="black", s=center_size, zorder=3)
    if neighbors.size:
        ax.scatter(neighbors[:, 0], neighbors[:, 1], c="#1f77b4", s=neighbor_size, zorder=2)
        ax.plot(
            np.append(neighbors[:, 0], neighbors[0, 0]),
            np.append(neighbors[:, 1], neighbors[0, 1]),
            color="#1f77b4",
            linewidth=2,
            alpha=0.9,
        )

    if show_unit_cell_outline:
        ax.plot(unit_cell[:, 0], unit_cell[:, 1], "k-", lw=2)

    ax.annotate(
        "",
        xy=(a1[0], a1[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="darkred", linewidth=arrow_lw),
    )
    ax.annotate(
        "",
        xy=(a2[0], a2[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="darkgreen", linewidth=arrow_lw),
    )
    ax.text(a1[0] * 0.55, a1[1] * 0.55, r"$\mathbf{a}_1$", color="darkred", fontsize=label_fontsize)
    ax.text(a2[0] * 0.55, a2[1] * 0.55, r"$\mathbf{a}_2$", color="darkgreen", fontsize=label_fontsize)

    ax.set_title("", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="both", labelsize=tick_labelsize)


def _points_from_vectors(a1: Tuple[float, float], a2: Tuple[float, float], extent: float, max_points: int) -> np.ndarray:
    vec1 = np.array(a1, dtype=float)
    vec2 = np.array(a2, dtype=float)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return np.empty((0, 2))

    step = min(norm1, norm2)
    max_index = int(np.ceil((extent + step) / step)) + 2

    limit = float(extent)
    points: list[tuple[float, float]] = []
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            position = i * vec1 + j * vec2
            if abs(position[0]) <= limit and abs(position[1]) <= limit:
                points.append((position[0], position[1]))

    if not points:
        return np.empty((0, 2))

    points.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    points_array = np.asarray(points, dtype=float)
    if max_points and len(points_array) > max_points:
        indices = np.linspace(0, len(points_array) - 1, max_points, dtype=int)
        points_array = points_array[indices]
    return points_array


def _layer_points(lattice_1: ml.PyLattice2D, lattice_2: ml.PyLattice2D, extent: float, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    a1_l1, a2_l1 = lattice_1.lattice_vectors()
    a1_l2, a2_l2 = lattice_2.lattice_vectors()

    points_l1 = _points_from_vectors(a1_l1, a2_l1, extent, max_points)
    points_l2 = _points_from_vectors(a1_l2, a2_l2, extent, max_points)

    return points_l1, points_l2


def _rectangle_points(lattice: ml.PyLattice2D, width: float, height: float) -> Sequence[Tuple[float, float]]:
    raw = lattice.get_direct_lattice_points_in_rectangle(width, height)
    if not raw:
        return []
    return [(x, y) for (x, y, _z) in raw]


def _precompute_lattice_indices(a1: Tuple[float, float], a2: Tuple[float, float], extent: float) -> list[tuple[int, int]]:
    vec1 = np.array(a1, dtype=float)
    vec2 = np.array(a2, dtype=float)
    step = min(np.linalg.norm(vec1), np.linalg.norm(vec2))
    # cover the square [-extent, extent]^2 with a circular margin
    R = float(extent) * np.sqrt(2.0) + 2.0 * step
    max_index = int(np.ceil((R + step) / max(step, 1e-9)))
    indices: list[tuple[int, int]] = []
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            pos = i * vec1 + j * vec2
            if np.linalg.norm(pos) <= R:
                indices.append((i, j))
    return indices


def _points_from_indices(indices: list[tuple[int, int]], a1: Tuple[float, float], a2: Tuple[float, float], extent: float, stride: int = 1) -> np.ndarray:
    vec1 = np.array(a1, dtype=float)
    vec2 = np.array(a2, dtype=float)
    if not indices:
        return np.empty((0, 2))
    # Stable subsampling by index grid stride
    is_sorted = indices
    i0 = is_sorted[0][0]
    j0 = is_sorted[0][1]
    lim = float(extent)
    out: list[tuple[float, float]] = []
    for (i, j) in is_sorted:
        if stride > 1 and (((i - i0) % stride) != 0 or ((j - j0) % stride) != 0):
            continue
        p = i * vec1 + j * vec2
        if -lim <= p[0] <= lim and -lim <= p[1] <= lim:
            out.append((p[0], p[1]))
    if not out:
        return np.empty((0, 2))
    return np.asarray(out, dtype=float)


def _sample_points(points: Sequence[Tuple[float, float]], max_points: int) -> Sequence[Tuple[float, float]]:
    if len(points) <= max_points:
        return points
    points_sorted = sorted(points, key=lambda p: p[0] ** 2 + p[1] ** 2)
    step = int(np.ceil(len(points_sorted) / max_points))
    return points_sorted[::step][:max_points]


def _exponential_angle_schedule(num_frames: int, max_angle_deg: float, growth_rate: float) -> np.ndarray:
    if num_frames <= 1:
        return np.array([max_angle_deg])

    t = np.linspace(0.0, 1.0, num_frames)
    if growth_rate <= 0:
        scaled = t
    else:
        # ease-out exponential: fast start, slower end
        e = np.expm1(growth_rate * t) / np.expm1(growth_rate)
        scaled = 1.0 - (1.0 - e) ** 1.5
    return max_angle_deg * scaled


def create_animation(config: AnimationConfig) -> Path:
    base_lattice = ml.create_hexagonal_lattice(config.lattice_constant)

    num_frames = max(2, int(round((config.duration_seconds * 1000) / config.interval_ms)))
    angle_schedule_deg = _exponential_angle_schedule(num_frames, config.max_angle_deg, config.growth_rate)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 1)
    ax_moire = fig.add_subplot(gs[0, 0])

    # No top title per request
    ax_moire.set_title("")
    ax_moire.set_aspect("equal")
    ax_moire.set_xlim(-config.moire_radius, config.moire_radius)
    ax_moire.set_ylim(-config.moire_radius, config.moire_radius)
    ax_moire.grid(False)
    tick_positions = np.linspace(-config.moire_radius, config.moire_radius, 7)
    ax_moire.set_xticks(tick_positions)
    ax_moire.set_yticks(tick_positions)
    ax_moire.tick_params(axis="both", which="both", direction="out", labelsize=22)

    # Use mid-dark grey for both layers
    lattice1_scatter = ax_moire.scatter([], [], c="#555555", s=3.2, alpha=0.55)
    lattice2_scatter = ax_moire.scatter([], [], c="#555555", s=3.2, alpha=0.55)
    moire_scatter = ax_moire.scatter([], [], c="#9467bd", s=20, alpha=0.9, label="Moiré")
    unit_cell_patch = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fill=False, ec="black", lw=2)
    ax_moire.add_patch(unit_cell_patch)

    # Placeholder; we'll set a figure-level label each frame to reduce unused whitespace
    info_text = None

    def init():
        lattice1_scatter.set_offsets(np.empty((0, 2)))
        lattice2_scatter.set_offsets(np.empty((0, 2)))
        moire_scatter.set_offsets(np.empty((0, 2)))
        unit_cell_patch.set_xy(np.zeros((4, 2)))
        # Remove any prior figure-level label (handled in update)
        return [lattice1_scatter, lattice2_scatter, moire_scatter, unit_cell_patch]

    # Precompute stable index grids for layers to avoid flicker from dynamic sampling
    a1_base, a2_base = base_lattice.lattice_vectors()
    l1_indices = _precompute_lattice_indices(a1_base, a2_base, config.moire_radius)

    def update(frame: int):
        display_angle_deg = angle_schedule_deg[frame]
        actual_angle_deg = max(display_angle_deg, config.min_angle_deg)
        angle_rad = np.deg2rad(actual_angle_deg)

        moire = ml.py_twisted_bilayer(base_lattice, angle_rad)
        moire_lattice = moire.as_lattice2d()
        lattice_1 = moire.lattice_1()
        lattice_2 = moire.lattice_2()

        layer_extent = config.moire_radius
        # Build layer points via stable indices and current layer vectors
        a1_l1, a2_l1 = lattice_1.lattice_vectors()
        a1_l2, a2_l2 = lattice_2.lattice_vectors()
        # Use full index set to maintain nearest-neighbor spacing; rely on small marker size
        points_l1 = _points_from_indices(l1_indices, a1_l1, a2_l1, layer_extent, stride=1)
        points_l2 = _points_from_indices(l1_indices, a1_l2, a2_l2, layer_extent, stride=1)
        rect_extent = config.moire_radius * 2.0
        moire_points = _rectangle_points(moire_lattice, rect_extent, rect_extent)

        if config.debug and frame in (0, int(len(angle_schedule_deg)/2), len(angle_schedule_deg)-1):
            a1_l1, a2_l1 = lattice_1.lattice_vectors()
            a1_l2, a2_l2 = lattice_2.lattice_vectors()
            a1_m, a2_m = moire_lattice.lattice_vectors()
            print("--- Debug frame:", frame)
            print("angle_deg(display/actual):", float(display_angle_deg), float(actual_angle_deg))
            print("|a1_l1|, |a2_l1|:", np.linalg.norm(a1_l1), np.linalg.norm(a2_l1))
            print("|a1_l2|, |a2_l2|:", np.linalg.norm(a1_l2), np.linalg.norm(a2_l2))
            print("|a1_m|, |a2_m|:", np.linalg.norm(a1_m), np.linalg.norm(a2_m))
            # Compare custom vs framework for a small window
            cmp_extent = min(15.0, layer_extent)
            l1_rect = np.array(_rectangle_points(lattice_1, 2*cmp_extent, 2*cmp_extent))
            l2_rect = np.array(_rectangle_points(lattice_2, 2*cmp_extent, 2*cmp_extent))
            print("L1 custom count:", len(points_l1), "rect count:", (0 if l1_rect.size==0 else len(l1_rect)))
            print("L2 custom count:", len(points_l2), "rect count:", (0 if l2_rect.size==0 else len(l2_rect)))
            if l1_rect.size:
                # check a few nearest distances
                for label, arr in (("L1 custom", points_l1), ("L1 rect", l1_rect[:, :2])):
                    if len(arr) >= 2:
                        dists = np.linalg.norm(arr - arr[0], axis=1)
                        print(label, "first 5 dists:", np.round(np.sort(dists)[1:6], 4))
            if l2_rect.size:
                for label, arr in (("L2 custom", points_l2), ("L2 rect", l2_rect[:, :2])):
                    if len(arr) >= 2:
                        dists = np.linalg.norm(arr - arr[0], axis=1)
                        print(label, "first 5 dists:", np.round(np.sort(dists)[1:6], 4))

        lattice1_scatter.set_offsets(points_l1)
        lattice2_scatter.set_offsets(points_l2)
        moire_scatter.set_offsets(
            _array_from_points(_sample_points(moire_points, config.max_moire_points))
        )

        a1_moire, a2_moire = moire_lattice.lattice_vectors()
        cell_vertices = np.array(
            [
                [0.0, 0.0],
                [a1_moire[0], a1_moire[1]],
                [a1_moire[0] + a2_moire[0], a1_moire[1] + a2_moire[1]],
                [a2_moire[0], a2_moire[1]],
            ]
        )
        unit_cell_patch.set_xy(cell_vertices)

        moire_vec_norm = np.linalg.norm(a1_moire)
        ratio = moire_vec_norm / config.lattice_constant
        # Centered figure-level label below the plot to use space efficiently
        fig.subplots_adjust(bottom=0.14, left=0.08, right=0.98, top=0.96)
        # Remove any existing figure-level texts (we only add the footer here)
        for artist in list(fig.texts):
            artist.remove()
        fig.text(
            0.5,
            0.05,
            f"Twist angle: {display_angle_deg:6.3f}°    |    L / a: {ratio:8.2f}",
            ha="center",
            va="center",
            fontsize=24,
        )
        return [lattice1_scatter, lattice2_scatter, moire_scatter, unit_cell_patch]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=config.interval_ms,
        blit=False,
    )

    output_path = Path(__file__).resolve().parent / config.output_filename
    fps = max(1, int(round(1000 / config.interval_ms)))
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=120)

    plt.close(fig)
    return output_path


def export_base_lattice_image(path: Path | None = None, lattice_constant: float = 1.0, axis_limit: float = 1.2) -> Path:
    """Export a standalone image of the base triangular lattice with labels under the plot.

    Uses a larger canvas with tight layout to make best use of space.
    """
    base = ml.create_hexagonal_lattice(lattice_constant)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    _build_base_lattice(ax, base, lattice_constant, axis_limit)
    # Title below the axes
    fig.subplots_adjust(bottom=0.15, left=0.08, right=0.98, top=0.98)
    fig.text(0.5, 0.03, "Base triangular lattice", ha="center", va="center", fontsize=14)
    if path is None:
        path = Path(__file__).resolve().parent / "base_triangular_lattice.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def export_unitcell_grid_image(
    path: Path | None = None,
    *,
    lattice_constant: float = 1.0,
    circle_radius: float = 0.3,
    grid_res: int = 32,
    unit_multiplier: int = 2,
) -> Path:
    """Create an illustration of a triangular lattice with r=0.3 circles and a 32x32 grid in the unit cell.

    - Paint a 2x2 unit-cell region using nearest-neighbor classification: inside any circle -> circle color; else white.
    - Draw explicit 32x32 grid lines inside the single unit cell for demonstration.
    - Reuse the base lattice styling (axes, vectors) and overlay circles at lattice points.
    """
    base = ml.create_hexagonal_lattice(lattice_constant)
    a1, a2 = base.lattice_vectors()
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)

    # Figure and base lattice
    fig = plt.figure(figsize=(7.5, 7.0))
    ax = fig.add_subplot(111)

    # Centered region: u,v in [-uv_extent, +uv_extent] (slightly larger than 2x2 to show more circles)
    uv_extent = 1.2
    corners_2x2 = np.array(
        [
            -uv_extent * a1 - uv_extent * a2,
            +uv_extent * a1 - uv_extent * a2,
            +uv_extent * a1 + uv_extent * a2,
            -uv_extent * a1 + uv_extent * a2,
        ],
        dtype=float,
    )
    min_xy = corners_2x2.min(axis=0) - (circle_radius + 0.2)
    max_xy = corners_2x2.max(axis=0) + (circle_radius + 0.2)
    axis_limit = float(max(np.abs(min_xy).max(), np.abs(max_xy).max()))

    # Base lattice styling (arrows, unit cell outline, ticks)
    _build_base_lattice(
        ax,
        base,
        lattice_constant,
        axis_limit,
        arrow_lw=5.0,
        tick_labelsize=24,
        label_fontsize=26,
        center_size=70.0,
        neighbor_size=52.0,
        show_unit_cell_outline=False,
    )

    # Paint region via nearest-neighbor classification on a square (Cartesian) grid
    # Grid spacing based on lattice constant 'a': s = a/32
    s = lattice_constant / float(grid_res)
    x0 = s * np.floor(min_xy[0] / s)
    x1 = s * np.ceil(max_xy[0] / s)
    y0 = s * np.floor(min_xy[1] / s)
    y1 = s * np.ceil(max_xy[1] / s)
    x_edges = np.arange(x0, x1 + s * 0.5, s)
    y_edges = np.arange(y0, y1 + s * 0.5, s)
    Xe, Ye = np.meshgrid(x_edges, y_edges, indexing="xy")

    # Cell centers for classification
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    # Candidate lattice points around the 2x2 area for nearest-neighbor search (centered indices)
    cand = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * a1 + j * a2
            cand.append(p)
    cand = np.array(cand)

    # Compute min distance from each center to nearest lattice point
    centers = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
    # Efficient distance computation
    d2 = ((centers[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)
    min_d = np.sqrt(d2.min(axis=1)).reshape(Xc.shape)

    # Determine which square centers fall inside the centered 2x2 region using barycentric coords
    A = np.column_stack([a1, a2])  # 2x2
    invA = np.linalg.inv(A)
    uv = centers @ invA.T
    u = uv[:, 0].reshape(Xc.shape)
    v = uv[:, 1].reshape(Xc.shape)
    inside_2x2 = (u >= -uv_extent) & (u <= uv_extent) & (v >= -uv_extent) & (v <= uv_extent)

    vals = np.full(Xc.shape, np.nan)
    vals[inside_2x2] = (min_d[inside_2x2] <= circle_radius).astype(float)
    cmap = ListedColormap(["#ffffff", "#555555"])  # background white, circle mid-dark grey
    ax.pcolormesh(Xe, Ye, vals, cmap=cmap, shading="auto")

    # Draw the explicit unit cell outline (centered)
    uc_outline = np.array([
        -0.5 * a1 - 0.5 * a2,
        +0.5 * a1 - 0.5 * a2,
        +0.5 * a1 + 0.5 * a2,
        -0.5 * a1 + 0.5 * a2,
        -0.5 * a1 - 0.5 * a2,
    ], dtype=float)
    ax.plot(uc_outline[:, 0], uc_outline[:, 1], color="black", lw=2.2, zorder=4)

    # Draw a faint 32x32 square grid inside the single centered unit cell (clip to parallelogram)
    uc = np.array([
        -0.5 * a1 - 0.5 * a2,
        +0.5 * a1 - 0.5 * a2,
        +0.5 * a1 + 0.5 * a2,
        -0.5 * a1 + 0.5 * a2,
    ])
    # Add an invisible polygon for clipping in data coords
    clip_poly = Polygon(uc, closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip_poly)

    # Vertical lines
    # Build square grid over the full view window then clip to unit cell
    view_x0, view_x1 = -1.5, 1.5
    view_y0, view_y1 = -1.5, 1.5
    xs = np.arange(s * np.floor(view_x0 / s), s * np.ceil(view_x1 / s) + s * 0.5, s)
    ys = np.arange(s * np.floor(view_y0 / s), s * np.ceil(view_y1 / s) + s * 0.5, s)
    for xx in xs:
        ln = ax.plot([xx, xx], [view_y0, view_y1], color="black", lw=0.6, alpha=0.25, zorder=6)[0]
        ln.set_clip_path(clip_poly)
    for yy in ys:
        ln = ax.plot([view_x0, view_x1], [yy, yy], color="black", lw=0.6, alpha=0.25, zorder=6)[0]
        ln.set_clip_path(clip_poly)

    # Draw the 2x2 region outline as a dashed box for clarity
    box2 = np.array([
        -uv_extent * a1 - uv_extent * a2,
        +uv_extent * a1 - uv_extent * a2,
        +uv_extent * a1 + uv_extent * a2,
        -uv_extent * a1 + uv_extent * a2,
        -uv_extent * a1 - uv_extent * a2,
    ], dtype=float)
    ax.plot(box2[:, 0], box2[:, 1], color="#333333", lw=2.0, ls="--", alpha=0.8, zorder=4)
    # No circle border overlays (to avoid obscuring the grid)

    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("")
    # Fix view window as requested
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # Slightly tighten layout for caption space if needed
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.98)

    if path is None:
        path = Path(__file__).resolve().parent / "triangular_unitcell_grid.png"
    # Ensure string path for savefig to appease some linters
    fig.savefig(str(path), dpi=180)
    plt.close(fig)
    assert isinstance(path, Path)
    return path


def main() -> None:
    config = AnimationConfig()
    output_path = create_animation(config)
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
