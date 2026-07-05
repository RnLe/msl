"""Phase 5: Meep-based cavity validation with geometry previews."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, cast

import h5py
import matplotlib.pyplot as plt
try:  # Matplotlib's mathtext parser may be unavailable on minimal builds
    from matplotlib.mathtext import MathTextParser
except Exception:  # pragma: no cover - optional dependency
    MathTextParser = None  # type: ignore
import meep as mp
import numpy as np
import pandas as pd

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None

if MPI is not None:
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
else:
    MPI_COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1

IS_ROOT = MPI_RANK == 0
MPI_ENABLED = MPI_COMM is not None and MPI_SIZE > 1
MATH_TEXT_PARSER: MathTextParser | None = None  # type: ignore[valid-type]


def log(message: str):
    if IS_ROOT:
        print(message)


def log_rank(message: str):
    prefix = f"[Rank {MPI_RANK}] " if MPI_ENABLED else ""
    print(f"{prefix}{message}")


def mpi_barrier():
    if MPI_COMM is not None:
        MPI_COMM.Barrier()


def broadcast_value(value):
    if MPI_COMM is not None:
        return MPI_COMM.bcast(value if IS_ROOT else None, root=0)
    return value


def _mathtext_can_render(text: str) -> bool:
    if not text or MathTextParser is None:
        return False
    global MATH_TEXT_PARSER
    expr = text.strip()
    if expr.startswith("$") and expr.endswith("$") and len(expr) >= 2:
        expr = expr[1:-1].strip()
    if not expr:
        return False
    if MATH_TEXT_PARSER is None:
        try:
            MATH_TEXT_PARSER = MathTextParser("path")
        except Exception:
            MATH_TEXT_PARSER = None
            return False
    try:
        # dpi value is arbitrary; parser only needs to validate the string
        MATH_TEXT_PARSER.parse(expr, dpi=72)
        return True
    except Exception:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json  # noqa: E402
from common.moire_utils import create_twisted_bilayer  # noqa: E402
from phases.phase2_ea_operator import resolve_run_dir  # noqa: E402


def _format_significant(value: float, digits: int = 2) -> str:
    if not math.isfinite(value) or value == 0.0:
        return "0"
    formatted = f"{value:.{digits}g}"
    if "e" in formatted or "E" in formatted:
        numeric = float(formatted)
        formatted = f"{numeric:.{digits}g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted

def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _discover_phase3_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that already contain Phase 3 eigenvalues."""
    results: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase3_eigenvalues.csv").exists():
            results.append((cid, cdir))
    return results


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception as exc:
            print(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return {"candidate_id": candidate_id}


def _load_phase1_metadata(cdir: Path) -> Dict[str, float]:
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        attrs = hf.attrs
        def _attr(name: str, default):
            value = attrs.get(name, default)
            if isinstance(value, bytes):
                return value.decode()
            return value
        meta = {
            "omega_ref": float(attrs.get("omega_ref", math.nan)),
            "eta": float(attrs.get("eta", math.nan)),
            "moire_length": float(attrs.get("moire_length", math.nan)),
            "theta_deg": float(attrs.get("theta_deg", math.nan)),
            "a": float(attrs.get("a", 1.0)),
            "lattice_type": _attr("lattice_type", "hex"),
            "r_over_a": float(attrs.get("r_over_a", 0.2)),
            "eps_bg": float(attrs.get("eps_bg", 12.0)),
        }
    return meta


def _select_mode_row(
    cdir: Path,
    config: Dict,
    return_table: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.DataFrame]:
    csv_path = cdir / "phase3_eigenvalues.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 3 eigenvalues missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Phase 3 eigenvalue table is empty for {cdir}")
    mode_setting = config.get("phase5_target_mode", 0)
    if isinstance(mode_setting, str) and mode_setting.lower() == "min_delta":
        selected = df.nsmallest(1, "delta_omega").iloc[0].copy()
        return (selected, df) if return_table else selected
    try:
        target_idx = int(mode_setting)
        row = df[df["mode_index"] == target_idx]
        if not row.empty:
            selected = row.iloc[0].copy()
            return (selected, df) if return_table else selected
    except Exception:
        pass
    df_sorted = df.sort_values("delta_omega")
    selected = df_sorted.iloc[0].copy()
    return (selected, df) if return_table else selected


def _rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _lattice_points_in_bounds(a1: np.ndarray, a2: np.ndarray, bounds: Tuple[float, float, float, float], pad: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    basis = np.column_stack([a1[:2], a2[:2]])
    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Lattice basis is singular") from exc
    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ])
    frac = (basis_inv @ corners.T).T
    i_min = math.floor(frac[:, 0].min()) - 2
    i_max = math.ceil(frac[:, 0].max()) + 2
    j_min = math.floor(frac[:, 1].min()) - 2
    j_max = math.ceil(frac[:, 1].max()) + 2
    pts: List[np.ndarray] = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            vec = i * a1[:2] + j * a2[:2]
            if xmin <= vec[0] <= xmax and ymin <= vec[1] <= ymax:
                pts.append(vec)
    if not pts:
        return np.zeros((0, 2))
    return np.stack(pts)


def _coerce_pair(value: SupportsFloat | Sequence[SupportsFloat]) -> Tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = [float(item) for item in value]
        if len(seq) != 2:
            raise ValueError(
                "Expected a length-2 sequence for pair-valued configuration, "
                f"received length {len(seq)}"
            )
        return seq[0], seq[1]

    scalar = float(value)
    return scalar, scalar


def _point_inside_bounds(point: np.ndarray, bounds: Tuple[float, float, float, float], margin: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return (xmin + margin) <= point[0] <= (xmax - margin) and (ymin + margin) <= point[1] <= (ymax - margin)


def _aggregate_hole_centers(points: np.ndarray, radius: float) -> List[Tuple[np.ndarray, float]]:
    centers = []
    for pt in points:
        centers.append((np.asarray(pt, dtype=float), radius))
    return centers


def _point_in_any_hole(point: np.ndarray, holes: List[Tuple[np.ndarray, float]], tol: float) -> bool:
    for center, radius in holes:
        if np.linalg.norm(point - center) <= max(radius - tol, 0.0):
            return True
    return False


def _candidate_directions(a1: np.ndarray, a2: np.ndarray) -> List[np.ndarray]:
    base_vectors = [0.5 * (a1 + a2), a1, a2, a1 - a2, np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    directions: List[np.ndarray] = []
    for vec in base_vectors:
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            continue
        unit = vec / norm
        directions.append(unit)
        directions.append(-unit)
    return directions


def _find_source_position(
    cavity_pos: np.ndarray,
    holes: List[Tuple[np.ndarray, float]],
    step: float,
    max_steps: int,
    bounds: Tuple[float, float, float, float],
    directions: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    if not _point_in_any_hole(cavity_pos, holes, 1e-6):
        return cavity_pos, 0.0
    if not directions:
        raise RuntimeError("No candidate directions to place the source")
    for step_idx in range(1, max_steps + 1):
        for direction in directions:
            candidate = cavity_pos + direction * (step * step_idx)
            if not _point_inside_bounds(candidate, bounds, margin=step):
                continue
            if not _point_in_any_hole(candidate, holes, 1e-6):
                shift = candidate - cavity_pos
                return candidate, float(np.linalg.norm(shift))
    raise RuntimeError("Could not place a source outside the etched region")


def _build_geometry(
    phase1_meta: Dict,
    config: Dict,
) -> Dict:
    theta_deg = float(phase1_meta.get("theta_deg", 0.0))
    theta_rad = math.radians(theta_deg)
    lattice_type = str(phase1_meta.get("lattice_type", "hex"))
    a = float(phase1_meta.get("a", 1.0))
    r_over_a = float(phase1_meta.get("r_over_a", 0.2))
    eps_bg = float(phase1_meta.get("eps_bg", 12.0))
    moire_length = float(phase1_meta.get("moire_length", a))

    bilayer = create_twisted_bilayer(lattice_type, theta_deg, a)
    a1 = np.asarray(bilayer["a1"][:2], dtype=float)
    a2 = np.asarray(bilayer["a2"][:2], dtype=float)
    rot = _rotation_matrix(theta_rad)
    top_a1 = rot @ a1
    top_a2 = rot @ a2

    window_cells = cast(SupportsFloat | Sequence[SupportsFloat], config.get("phase5_window_cells", 1.0))
    win_x_cells, win_y_cells = _coerce_pair(window_cells)
    window_x = float(win_x_cells) * moire_length
    window_y = float(win_y_cells) * moire_length
    bounds = (-0.5 * window_x, 0.5 * window_x, -0.5 * window_y, 0.5 * window_y)
    pad = float(config.get("phase5_lattice_padding", r_over_a * a))

    bottom_points = _lattice_points_in_bounds(a1, a2, bounds, pad)
    top_points = _lattice_points_in_bounds(top_a1, top_a2, bounds, pad)

    bottom_shift = np.asarray(config.get("phase5_bottom_shift", [0.0, 0.0]), dtype=float)
    top_shift = np.asarray(config.get("phase5_top_shift", [0.0, 0.0]), dtype=float)
    if bottom_points.size:
        bottom_points = bottom_points + bottom_shift
    if top_points.size:
        top_points = top_points + top_shift

    radius = r_over_a * a
    hole_height = config.get("phase5_cylinder_height", mp.inf)
    air = mp.Medium(epsilon=1.0)
    z_sep = float(config.get("phase5_layer_separation", 0.0)) * 0.5
    geometry: List[mp.Cylinder] = []
    hole_centers: List[Tuple[np.ndarray, float]] = []
    for pt in bottom_points:
        geometry.append(
            mp.Cylinder(
                radius=radius,
                height=hole_height,
                center=mp.Vector3(float(pt[0]), float(pt[1]), -z_sep),
                material=air,
            )
        )
        hole_centers.append((np.asarray(pt, dtype=float), radius))
    for pt in top_points:
        geometry.append(
            mp.Cylinder(
                radius=radius,
                height=hole_height,
                center=mp.Vector3(float(pt[0]), float(pt[1]), z_sep),
                material=air,
            )
        )
        hole_centers.append((np.asarray(pt, dtype=float), radius))

    pml = float(config.get("phase5_pml_thickness", 2.0))
    cell_x = window_x + 2.0 * pml
    cell_y = window_y + 2.0 * pml
    cell_z = float(config.get("phase5_cell_height", 0.0))

    return {
        "geometry": geometry,
        "hole_centers": hole_centers,
        "bounds": bounds,
        "window_span": (window_x, window_y),
        "cell_span": (cell_x, cell_y, cell_z),
        "radius": radius,
        "a1": a1,
        "a2": a2,
        "top_a1": top_a1,
        "top_a2": top_a2,
        "eps_bg": eps_bg,
        "pml": pml,
        "counts": {"bottom": len(bottom_points), "top": len(top_points)},
        "bottom_points": bottom_points.copy() if bottom_points.size else bottom_points,
        "top_points": top_points.copy() if top_points.size else top_points,
    }


def _build_simulation(ctx: Dict, resolution: float, sources=None) -> mp.Simulation:
    span_x, span_y, span_z = ctx["cell_span"]
    cell = mp.Vector3(span_x, span_y, span_z)
    boundary_layers = [mp.PML(thickness=ctx["pml"])] if ctx["pml"] > 0 else []
    sim = mp.Simulation(
        cell_size=cell,
        geometry=ctx["geometry"],
        resolution=resolution,
        boundary_layers=boundary_layers,
        default_material=mp.Medium(epsilon=ctx["eps_bg"]),
        sources=sources or [],
    )
    return sim


def _render_geometry_preview(
    sim: mp.Simulation,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    out_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    sim.plot2D(ax=ax)
    ax.scatter(
        [source_pos[0]],
        [source_pos[1]],
        marker="o",
        s=22,
        c="C0",
        edgecolors="white",
        linewidths=0.4,
        label="Source",
        zorder=5,
    )
    ax.scatter(
        [cavity_pos[0]],
        [cavity_pos[1]],
        marker="o",
        s=18,
        c="C1",
        edgecolors="white",
        linewidths=0.4,
        label="EA peak",
        zorder=6,
    )
    ax.legend(loc="upper right")
    ax.set_title("Meep Geometry Preview")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_gaussian_pulse(
    cdir: Path,
    mode_row: pd.Series,
    config: Dict,
    modes: pd.DataFrame,
):
    fwidth = float(config.get("phase5_source_fwidth", 0.02))
    cutoff = float(config.get("phase5_source_cutoff", 3.0))
    span_multiplier = max(cutoff, float(config.get("phase5_pulse_span_multiplier", 4.0)))
    freq = float(mode_row.get("omega_cavity", 0.0))
    if not math.isfinite(freq):
        freq = 0.0
    if fwidth <= 0:
        baseline = max(abs(freq) * 0.05, 0.01)
        fwidth = baseline
    sigma = fwidth / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sorted_freqs: List[float] = []
    if modes is not None and not modes.empty:
        mode_freqs = modes.get("omega_cavity")
        if mode_freqs is None:
            delta_series = modes.get("delta_omega")
            if delta_series is not None:
                omega_ref = float(mode_row.get("omega_cavity", freq) - mode_row.get("delta_omega", 0.0))
                mode_freqs = omega_ref + delta_series
        if mode_freqs is not None:
            sorted_freqs = [float(value) for value in mode_freqs if math.isfinite(value)]
            sorted_freqs.sort()

    target_idx = -1
    if sorted_freqs:
        diffs = [abs(value - freq) for value in sorted_freqs]
        target_idx = int(np.argmin(diffs))

    axis_half_default = max(span_multiplier * fwidth, fwidth)
    min_display_half = max(0.5 * fwidth, 5e-3)
    axis_half = max(axis_half_default, min_display_half)
    if sorted_freqs:
        span_left = max(freq - sorted_freqs[0], 0.0)
        span_right = max(sorted_freqs[-1] - freq, 0.0)
        mode_half_span = max(span_left, span_right)
        buffer_frac = float(config.get("phase5_mode_axis_buffer", 0.05))
        dynamic_buffer = max(buffer_frac * max(mode_half_span, min_display_half), 1e-4)
        axis_half = max(mode_half_span + dynamic_buffer, min_display_half)

    f_min = freq - axis_half
    f_max = freq + axis_half
    if not math.isfinite(f_min) or not math.isfinite(f_max) or f_max <= f_min:
        f_min = freq - axis_half_default
        f_max = freq + axis_half_default
    x = np.linspace(f_min, f_max, 1024)
    denom = max(sigma, 1e-9)
    gaussian = np.exp(-0.5 * ((x - freq) / denom) ** 2)
    if gaussian.max() > 0:
        gaussian /= gaussian.max()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=int(config.get("phase5_preview_dpi", 400)))
    ax.plot(x, gaussian, color="tab:blue", label="Gaussian pulse")
    ax.set_xlabel("Frequency (ω)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-0.05, 1.05)  # Keep original ylim for compatibility
    ax.set_xlim(f_min, f_max)
    ax.set_title("Phase 5 Source Spectrum")

    half_amp = 0.5
    half_width = 0.5 * (2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
    left = max(freq - half_width, f_min)
    right = min(freq + half_width, f_max)
    fwhm_label = f"FWHM = {_format_significant(2.0 * half_width, digits=2)}"
    ax.hlines(
        half_amp,
        left,
        right,
        colors="tab:purple",
        linestyles="--",
        label=fwhm_label,
    )

    ylim_top = 1.05
    if sorted_freqs:
        target_color = str(config.get("phase5_target_mode_color", "#d97706"))
        other_color = str(config.get("phase5_other_mode_color", "#1f2937"))
        other_labeled = False
        label_height = float(config.get("phase5_target_label_height", 1.08))
        label_text = str(config.get("phase5_target_label_text", r"$\\omega_0$"))
        prefer_math = bool(config.get("phase5_target_label_math", True))
        use_math = prefer_math and _mathtext_can_render(label_text)
        fallback_text = (label_text.replace("$", "") or "omega0") if prefer_math else label_text
        if prefer_math and not use_math:
            print(
                "    WARNING: MathText unavailable for phase5 target label"
                f" '{label_text}'. Rendering '{fallback_text}' instead."
            )
        for idx, value in enumerate(sorted_freqs):
            if idx == target_idx:
                ax.axvline(
                    value,
                    color=target_color,
                    linewidth=1.3,
                    label="Target mode",
                )
                ax.text(
                    value,
                    label_height,
                    label_text if use_math else fallback_text,
                    rotation=0,
                    va="bottom",
                    ha="center",
                    fontsize=9,
                    color=target_color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
                )
                ylim_top = max(ylim_top, label_height + 0.05)
            else:
                ax.axvline(
                    value,
                    color=other_color,
                    alpha=0.85,
                    linewidth=1.0,
                    label="Other modes" if not other_labeled else None,
                )
                other_labeled = True
    ax.set_ylim(-0.05, ylim_top)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    out_path = cdir / "phase5_source_pulse.png"
    fig.savefig(out_path)
    plt.close(fig)


def _build_sources(component, source_pos: np.ndarray, freq: float, config: Dict) -> List[mp.Source]:
    fwidth = float(config.get("phase5_source_fwidth", 0.02))
    cutoff = float(config.get("phase5_source_cutoff", 3.0))
    amplitude = float(config.get("phase5_source_amplitude", 1.0))
    src = mp.GaussianSource(frequency=freq, fwidth=fwidth, cutoff=cutoff)
    return [
        mp.Source(
            src=src,
            component=component,
            center=mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0),
            amplitude=amplitude,
        )
    ]


def _run_meep(
    ctx: Dict,
    component,
    freq: float,
    config: Dict,
    source_pos: np.ndarray,
) -> Tuple[List[Dict[str, float]], List[np.ndarray]]:
    resolution = float(config.get("phase5_resolution", 32))
    sources = _build_sources(component, source_pos, freq, config)
    sim = _build_simulation(ctx, resolution, sources=sources)
    harminv_bw = float(config.get("phase5_harminv_bw", 0.05))
    harminv = mp.Harminv(component, mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0), freq, harminv_bw)
    decay_dt = float(config.get("phase5_decay_dt", 50.0))
    decay_threshold = float(config.get("phase5_decay_threshold", 1e-9))
    run_time = float(config.get("phase5_run_time", 400.0))
    gif_dt = float(config.get("phase5_gif_dt", 6.0))
    capture_dt_default = gif_dt * 0.5 if gif_dt > 0 else 0.0
    gif_capture_dt = float(config.get("phase5_gif_capture_dt", capture_dt_default))
    gif_capture_dt = max(gif_capture_dt, 1e-3) if gif_capture_dt > 0 else 0.0
    gif_max_frames = int(config.get("phase5_gif_max_frames", 240)) * 2
    gif_stride = max(1, int(config.get("phase5_gif_stride", 4)))
    capture_frames = bool(config.get("phase5_capture_frames", True)) and gif_capture_dt > 0
    store_frames = capture_frames and IS_ROOT
    window_x, window_y = ctx["window_span"]
    frames: List[np.ndarray] = []

    def _record_frame(sim_obj: mp.Simulation):
        if not capture_frames:
            return
        if len(frames) >= gif_max_frames:
            return
        arr = sim_obj.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(window_x, window_y, 0.0),
            component=component,
        )
        if arr is None or not store_frames:
            return
        frame = np.array(arr, copy=True, dtype=np.float32)
        if gif_stride > 1:
            frame = frame[::gif_stride, ::gif_stride]
        frames.append(frame)

    decay_stop = mp.stop_when_fields_decayed(
        decay_dt,
        component,
        mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0),
        decay_threshold,
    )
    if gif_capture_dt > 0:
        sim.run(
            mp.after_sources(harminv),
            mp.at_every(gif_capture_dt, _record_frame),
            decay_stop,
            until=run_time,
        )
    else:
        sim.run(
            mp.after_sources(harminv),
            decay_stop,
            until=run_time,
        )
    rows: List[Dict[str, float]] = []
    min_q = float(config.get("phase5_min_Q", 50.0))
    for mode in harminv.modes:
        if mode.Q < min_q:
            continue
        amp_val = getattr(mode, "amp", None)
        if amp_val is None:
            amp_val = getattr(mode, "alpha", None)
        amp_abs = float(abs(amp_val)) if amp_val is not None else float("nan")
        amp_real = float(np.real(amp_val)) if amp_val is not None else float("nan")
        amp_imag = float(np.imag(amp_val)) if amp_val is not None else float("nan")
        rows.append(
            {
                "freq_meep": float(mode.freq),
                "Q": float(mode.Q),
                "decay": float(mode.decay),
                "amplitude_abs": amp_abs,
                "amplitude_real": amp_real,
                "amplitude_imag": amp_imag,
            }
        )
    sim.reset_meep()
    if not store_frames:
        frames = []
    return rows, frames


def _classify_quality(Q_val: float, config: Dict) -> Tuple[str, str]:
    minor = float(config.get("phase5_quality_minor", 250.0))
    good = float(config.get("phase5_quality_good", 1000.0))
    elite = float(config.get("phase5_quality_strong", 2500.0))
    if not math.isfinite(Q_val):
        return "unknown", "No Harminv data captured"
    if Q_val < minor:
        return "diffuse", f"Q < {minor:.0f}: leakage-dominated"
    if Q_val < good:
        return "incipient", f"{minor:.0f} ≤ Q < {good:.0f}: weak confinement"
    if Q_val < elite:
        return "cavity", f"{good:.0f} ≤ Q < {elite:.0f}: good cavity"
    return "elite", f"Q ≥ {elite:.0f}: strong cavity"


def _write_results(
    cdir: Path,
    rows: List[Dict[str, float]],
    cid: int,
    mode_index: int,
    omega_ea: float,
    source_shift: float,
    config: Dict,
) -> List[Dict[str, float]]:
    enriched = []
    for row in rows:
        rel_error = abs(row["freq_meep"] - omega_ea) / max(abs(omega_ea), 1e-12)
        quality_label, quality_note = _classify_quality(row["Q"], config)
        enriched.append(
            {
                "candidate_id": cid,
                "mode_index": mode_index,
                "omega_ea": omega_ea,
                "omega_meep": row["freq_meep"],
                "Q": row["Q"],
                "decay": row["decay"],
                "amplitude_abs": row["amplitude_abs"],
                "amplitude_real": row["amplitude_real"],
                "amplitude_imag": row["amplitude_imag"],
                "relative_error": rel_error,
                "source_shift": source_shift,
                "quality_label": quality_label,
                "quality_note": quality_note,
            }
        )
    results_path = cdir / "phase5_q_factor_results.csv"
    if enriched:
        pd.DataFrame(enriched).to_csv(results_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "candidate_id",
                "mode_index",
                "omega_ea",
                "omega_meep",
                "Q",
                "decay",
                "amplitude_abs",
                "amplitude_real",
                "amplitude_imag",
                "relative_error",
                "source_shift",
                "quality_label",
                "quality_note",
            ]
        ).to_csv(results_path, index=False)
    return enriched


def _render_geometry_preview_static(
    geometry_ctx: Dict,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    out_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    bottom_points = geometry_ctx.get("bottom_points")
    top_points = geometry_ctx.get("top_points")
    if isinstance(bottom_points, np.ndarray) and bottom_points.size:
        ax.scatter(
            bottom_points[:, 0],
            bottom_points[:, 1],
            s=6,
            c="#0ea5e9",
            alpha=0.55,
            edgecolors="none",
            label="Bottom layer",
        )
    if isinstance(top_points, np.ndarray) and top_points.size:
        ax.scatter(
            top_points[:, 0],
            top_points[:, 1],
            s=6,
            c="#ec4899",
            alpha=0.55,
            edgecolors="none",
            label="Top layer",
        )
    ax.scatter(
        [source_pos[0]],
        [source_pos[1]],
        marker="o",
        s=24,
        c="C0",
        edgecolors="white",
        linewidths=0.4,
        label="Source",
        zorder=5,
    )
    ax.scatter(
        [cavity_pos[0]],
        [cavity_pos[1]],
        marker="o",
        s=20,
        c="C1",
        edgecolors="white",
        linewidths=0.4,
        label="EA peak",
        zorder=6,
    )
    ax.legend(loc="upper right")
    ax.set_title("Meep Geometry Preview (static)")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _geometry_mask(shape: Tuple[int, int], ctx: Dict) -> np.ndarray:
    height, width = shape
    xmin, xmax, ymin, ymax = ctx["bounds"]
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymax, ymin, height)
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros((height, width), dtype=bool)
    for center, radius in ctx["hole_centers"]:
        cx, cy = float(center[0]), float(center[1])
        rr = float(radius)
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
    return mask


def _geometry_background(mask: np.ndarray) -> np.ndarray:
    bg = np.zeros(mask.shape + (4,), dtype=np.uint8)
    bg[..., :3] = 196
    bg[..., 3] = 210
    bg[mask, :3] = 255
    bg[mask, 3] = 255
    return bg


def _blend_with_background(field_rgba: np.ndarray, background: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    overlay = background.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay[..., :3] = (1.0 - alpha) * overlay[..., :3] + alpha * field_rgba[..., :3]
    overlay[..., 3] = 255
    return overlay


def _save_field_animation(
    frames: List[np.ndarray],
    out_path: Path,
    frame_duration: float,
    geometry_ctx: Optional[Dict] = None,
):
    if not frames:
        return
    try:
        import imageio.v3 as iio
    except ImportError:  # pragma: no cover - fallback for older versions
        import imageio as iio  # type: ignore

    vmax = max(float(np.max(np.abs(frame))) for frame in frames)
    if vmax <= 0:
        vmax = 1.0
    cmap = plt.colormaps.get_cmap("RdBu_r")
    geometry_images: List[np.ndarray] = []
    background = None
    if geometry_ctx is not None:
        mask = _geometry_mask(frames[0].shape, geometry_ctx)
        background = _geometry_background(mask)
    images = []
    for frame in frames:
        normalized = 0.5 * (frame / vmax) + 0.5
        normalized = np.clip(normalized, 0.0, 1.0)
        rgba = (255 * cmap(normalized)).astype(np.uint8)
        images.append(rgba)
        if background is not None:
            geometry_images.append(_blend_with_background(rgba, background))
    iio.imwrite(out_path, images, duration=max(frame_duration, 0.01), loop=0)
    if geometry_images:
        overlay_path = out_path.with_name(out_path.stem + "_geometry.gif")
        iio.imwrite(overlay_path, geometry_images, duration=max(frame_duration, 0.01), loop=0)


def _quality_legend(config: Dict) -> List[str]:
    minor = float(config.get("phase5_quality_minor", 250.0))
    good = float(config.get("phase5_quality_good", 1000.0))
    elite = float(config.get("phase5_quality_strong", 2500.0))
    return [
        f"diffuse: Q < {minor:.0f}",
        f"incipient: {minor:.0f} ≤ Q < {good:.0f}",
        f"cavity: {good:.0f} ≤ Q < {elite:.0f}",
        f"elite: Q ≥ {elite:.0f}",
    ]


def _write_report(cdir: Path, summary: Dict[str, float], modes: List[Dict[str, float]], config: Dict):
    top_modes = sorted(modes, key=lambda row: row["Q"], reverse=True)
    lines = [
        "# Phase 5 Meep Validation Report",
        "",
        f"**Candidate**: {summary['candidate_id']:04d}",
        "",
        "## Geometry",
        f"- Window span: {summary['window_x']:.3f} × {summary['window_y']:.3f}",
        f"- Cylinders: {summary['n_bottom']} bottom + {summary['n_top']} top",
        f"- Source shift from EA peak: {summary['source_shift']:.4f}",
        "",
        "## Simulation",
        f"- Target mode index: {summary['mode_index']}",
        f"- ω_EA: {summary['omega_ea']:.6f}",
        f"- Ran Meep: {'yes' if summary['ran_meep'] else 'no'}",
    ]
    if summary["ran_meep"]:
        lines.extend(
            [
                f"- Modes recorded: {summary['recorded_modes']}",
                f"- Best Q: {summary['best_q']:.1f}",
                f"- Best ω_meep: {summary['best_freq']:.6f}",
            ]
        )
        if top_modes:
            lines.append("")
            lines.append("### Highlighted modes (top 3 by Q)")
            for row in top_modes[:3]:
                lines.append(
                    f"- {str(row['quality_label']).title()} | Q={row['Q']:.0f} | "
                    f"ω_meep={row['omega_meep']:.6f} | Δω/ω={row['relative_error']:.3e}"
                )
        lines.append("")
        lines.append("### Quality legend")
        for note in _quality_legend(config):
            lines.append(f"- {note}")
    report_path = cdir / "phase5_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_env_overrides(config: Dict):
    if _env_flag("PHASE5_PLOTS_ONLY"):
        config["phase5_run_meep"] = False
        config["phase5_plot_pulse"] = True
        config.setdefault("phase5_preview_dpi", 400)
        log("Environment override: PHASE5_PLOTS_ONLY -> skipping Meep run and rendering plots only.")


def process_candidate(
    row: pd.Series | Dict[str, Any],
    config: Dict,
    run_dir: Path,
    shared_mode: bool = False,
) -> List[Dict[str, float]]:
    if row is None:
        raise ValueError("Candidate row metadata missing on this rank")
    if isinstance(row, dict):
        row = pd.Series(row)
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    if shared_mode and MPI_COMM is not None:
        phase1_meta = broadcast_value(_load_phase1_metadata(cdir) if IS_ROOT else None)
    else:
        phase1_meta = _load_phase1_metadata(cdir)
    if shared_mode and MPI_COMM is not None:
        if IS_ROOT:
            mode_row, mode_table = _select_mode_row(cdir, config, return_table=True)
            mode_payload = mode_row.to_dict()
        else:
            mode_row = None  # type: ignore[assignment]
            mode_table = None
            mode_payload = None
        mode_payload = broadcast_value(mode_payload)
        if mode_payload is None:
            raise RuntimeError(f"Rank {MPI_RANK} failed to receive Phase 3 mode data for candidate {cid}")
        mode_row = pd.Series(mode_payload)
        if mode_table is None:
            mode_table = pd.DataFrame()
    else:
        mode_row, mode_table = _select_mode_row(cdir, config, return_table=True)
    cavity_pos = np.array([
        float(mode_row.get("peak_x", math.nan)),
        float(mode_row.get("peak_y", math.nan)),
    ])
    if not np.all(np.isfinite(cavity_pos)):
        cavity_pos = np.array([
            float(mode_row.get("center_x", 0.0)),
            float(mode_row.get("center_y", 0.0)),
        ])
    geometry_ctx = _build_geometry(phase1_meta, config)
    directions = _candidate_directions(geometry_ctx["a1"], geometry_ctx["a2"])
    default_step = max(geometry_ctx["radius"] * 0.25, 0.05)
    step = float(config.get("phase5_source_step", default_step))
    max_steps = int(config.get("phase5_source_max_steps", 400))
    source_pos, shift_distance = _find_source_position(
        cavity_pos,
        geometry_ctx["hole_centers"],
        step,
        max_steps,
        geometry_ctx["bounds"],
        directions,
    )
    preview_mode = str(config.get("phase5_preview_backend", "meep")).strip().lower()
    render_preview = bool(config.get("phase5_render_preview", True))
    if render_preview and preview_mode != "none":
        preview_path = cdir / "phase5_meep_plot.png"
        preview_dpi = int(config.get("phase5_preview_dpi", 400))
        if preview_mode == "static":
            if IS_ROOT:
                _render_geometry_preview_static(geometry_ctx, cavity_pos, source_pos, preview_path, dpi=preview_dpi)
        else:
            base_resolution = float(config.get("phase5_resolution", 32))
            preview_resolution = float(config.get("phase5_preview_resolution", base_resolution * 1.5))
            preview_sim = _build_simulation(geometry_ctx, preview_resolution)
            preview_sim.init_sim()
            if IS_ROOT:
                _render_geometry_preview(preview_sim, cavity_pos, source_pos, preview_path, dpi=preview_dpi)
            preview_sim.reset_meep()
        mpi_barrier()
    if IS_ROOT and bool(config.get("phase5_plot_pulse", True)):
        _plot_gaussian_pulse(cdir, mode_row, config, mode_table)

    component_name = str(config.get("phase5_component", "Ez"))
    component = getattr(mp, component_name, None)
    if component is None:
        raise ValueError(f"Invalid Meep component '{component_name}'")
    freq = float(mode_row.get("omega_cavity", mode_row.get("delta_omega", 0.0)))

    run_meep = bool(config.get("phase5_run_meep", True))
    rows: List[Dict[str, float]] = []
    frames: List[np.ndarray] = []
    if run_meep:
        rows, frames = _run_meep(geometry_ctx, component, freq, config, source_pos)

    summary = {
        "candidate_id": cid,
        "mode_index": int(mode_row.get("mode_index", 0)),
        "omega_ea": freq,
        "window_x": geometry_ctx["window_span"][0],
        "window_y": geometry_ctx["window_span"][1],
        "n_bottom": geometry_ctx["counts"]["bottom"],
        "n_top": geometry_ctx["counts"]["top"],
        "source_shift": shift_distance,
        "ran_meep": run_meep,
        "recorded_modes": len(rows),
        "best_q": max((row["Q"] for row in rows), default=float("nan")),
        "best_freq": max((row["freq_meep"] for row in rows), default=float("nan")),
    }
    enriched_rows: List[Dict[str, float]] = rows
    if run_meep and (not shared_mode or IS_ROOT):
        enriched_rows = _write_results(cdir, rows, cid, summary["mode_index"], freq, shift_distance, config)
        _write_report(cdir, summary, enriched_rows, config)
    if frames and IS_ROOT:
        gif_path = cdir / "phase5_field_animation.gif"
        base_duration = float(config.get("phase5_gif_frame_duration", 0.08))
        speedup = float(config.get("phase5_gif_speedup", 1.25))
        frame_duration = max(base_duration / max(speedup, 1e-6), 0.01)
        _save_field_animation(frames, gif_path, frame_duration, geometry_ctx)
    rank_logging = bool(config.get("phase5_rank_logging", False))
    if IS_ROOT or rank_logging:
        log_rank(f"    Phase 5 artifacts written for candidate {cid}")
    mpi_barrier()
    return enriched_rows


def run_phase5(run_dir: str | Path, config_path: str | Path):
    if MPI_ENABLED:
        version = MPI.Get_version()
        log(f"\nUsing MPI version {version[0]}.{version[1]}, {MPI_SIZE} processes")
    log("=" * 70)
    log("PHASE 5: Meep Cavity Validation")
    log("=" * 70)

    config = load_yaml(config_path)
    if MPI_COMM is not None:
        config = broadcast_value(config)
    _apply_env_overrides(config)
    run_dir = resolve_run_dir(run_dir, config)
    log(f"Using run directory: {run_dir}")
    parallel_mode = str(config.get("phase5_parallel_mode", "shared")).strip().lower()
    if parallel_mode not in {"shared", "scatter"}:
        log(f"Unknown phase5_parallel_mode='{parallel_mode}', defaulting to 'shared'.")
        parallel_mode = "shared"
    shared_mode = MPI_ENABLED and parallel_mode == "shared"
    if MPI_ENABLED:
        if shared_mode:
            log("MPI parallel mode: shared (all ranks cooperate on each candidate)")
        else:
            log("MPI parallel mode: scatter (candidates divided across ranks)")

    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if IS_ROOT and candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    elif IS_ROOT:
        log(f"WARNING: {candidates_path} not found; relying on per-candidate metadata only.")
    if MPI_COMM is not None:
        candidate_payload = (
            candidate_frame.to_dict(orient="list") if candidate_frame is not None else None
        )
        candidate_payload = broadcast_value(candidate_payload)
        candidate_frame = pd.DataFrame.from_dict(candidate_payload) if candidate_payload is not None else None

    discovered = _discover_phase3_candidates(run_dir) if IS_ROOT else None
    if MPI_COMM is not None:
        discovered = broadcast_value(discovered)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 3 eigenvalues found in {run_dir}. "
            "Run Phase 3 before Phase 5."
        )

    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        discovered = discovered[:K]
        log(f"Processing {len(discovered)} candidate directories (limited to K={K}).")
    else:
        log(f"Processing {len(discovered)} candidate directories (all Phase 3 outputs found).")

    aggregate_rows: List[Dict[str, float]] = []
    if shared_mode:
        for cid, _ in discovered:
            if IS_ROOT:
                row_dict = _load_candidate_metadata(run_dir, cid, candidate_frame)
            else:
                row_dict = None
            if MPI_COMM is not None:
                row_dict = broadcast_value(row_dict if IS_ROOT else None)
            if row_dict is None:
                continue
            try:
                if IS_ROOT:
                    log(f"  Candidate {cid}: running Phase 5 (shared mode)")
                process_rows = process_candidate(row_dict, config, Path(run_dir), shared_mode=True)
                if IS_ROOT and process_rows:
                    aggregate_rows.extend(process_rows)
            except FileNotFoundError as exc:
                log_rank(f"    Skipping candidate {cid}: {exc}")
            except Exception as exc:  # pragma: no cover
                log_rank(f"    Phase 5 failed for candidate {cid}: {exc}")
                if MPI_COMM is not None:
                    MPI_COMM.Abort(1)
                else:
                    raise
    else:
        if MPI_ENABLED:
            assigned_candidates = [entry for idx, entry in enumerate(discovered) if idx % MPI_SIZE == MPI_RANK]
            log_rank(
                f"Assigned {len(assigned_candidates)} of {len(discovered)} candidates to this rank"
            )
        else:
            assigned_candidates = discovered

        aggregate_rows_local: List[Dict[str, float]] = []
        for cid, _ in assigned_candidates:
            row_dict = _load_candidate_metadata(run_dir, cid, candidate_frame)
            try:
                log_rank(f"  Candidate {cid}: running Phase 5")
                candidate_rows = process_candidate(row_dict, config, Path(run_dir))
                aggregate_rows_local.extend(candidate_rows)
            except FileNotFoundError as exc:
                log_rank(f"    Skipping candidate {cid}: {exc}")
            except Exception as exc:  # pragma: no cover
                log_rank(f"    Phase 5 failed for candidate {cid}: {exc}")
                if MPI_COMM is not None:
                    MPI_COMM.Abort(1)
                else:
                    raise

        if MPI_COMM is not None:
            gathered_rows = MPI_COMM.gather(aggregate_rows_local, root=0)
            if IS_ROOT:
                for chunk in gathered_rows:
                    if chunk:
                        aggregate_rows.extend(chunk)
        else:
            aggregate_rows = aggregate_rows_local

    if IS_ROOT and aggregate_rows:
        out_path = Path(run_dir) / "phase5_q_factor_results.csv"
        pd.DataFrame(aggregate_rows).to_csv(out_path, index=False)

    log("Phase 5 completed.\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase5_meep_qfactor.py <run_dir|auto> <phase5_config.yaml>")
    run_phase5(sys.argv[1], sys.argv[2])
