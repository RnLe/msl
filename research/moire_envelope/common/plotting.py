"""Plotting utilities for visualization."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import moire_lattice_py as ml

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None
    ImageFont = None

MONO_BLUE = (37, 99, 235)
LAYER_TWO_ORANGE = (249, 115, 22)
MOIRE_PURPLE = (147, 51, 234)
AXIS_COLOR = (148, 163, 184)
TEXT_COLOR = (15, 23, 42)

from common.moire_utils import create_twisted_bilayer


def _rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _extent_with_padding(extent, pad=0.1):
    xmin, xmax, ymin, ymax = map(float, extent)
    width = xmax - xmin
    height = ymax - ymin
    return (
        xmin - pad * width,
        xmax + pad * width,
        ymin - pad * height,
        ymax + pad * height,
    )


def _lattice_points_from_basis(a1_vec, a2_vec, extent, pad=0.1, max_points=600):
    a1 = np.asarray(a1_vec[:2], dtype=float)
    a2 = np.asarray(a2_vec[:2], dtype=float)
    xmin, xmax, ymin, ymax = _extent_with_padding(extent, pad)
    B = np.column_stack([a1, a2])
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return np.empty((0, 2))

    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ])
    frac = (B_inv @ corners.T).T
    i_min = math.floor(frac[:, 0].min()) - 2
    i_max = math.ceil(frac[:, 0].max()) + 2
    j_min = math.floor(frac[:, 1].min()) - 2
    j_max = math.ceil(frac[:, 1].max()) + 2

    pts = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            vec = i * a1 + j * a2
            if xmin <= vec[0] <= xmax and ymin <= vec[1] <= ymax:
                pts.append(vec)

    pts = np.asarray(pts)
    if pts.size == 0:
        return pts
    if max_points and pts.shape[0] > max_points:
        stride = max(1, pts.shape[0] // max_points)
        pts = pts[::stride]
    return pts


def _sample_lattice_points(lattice_like, extent, oversample=1.1, max_points=600):
    xmin, xmax, ymin, ymax = map(float, extent)
    width = max(1e-9, (xmax - xmin) * oversample)
    height = max(1e-9, (ymax - ymin) * oversample)
    center = np.array([(xmax + xmin) * 0.5, (ymax + ymin) * 0.5])
    origin = center - np.array([width * 0.5, height * 0.5])

    try:
        pts = lattice_like.compute_direct_lattice_points_in_rectangle(width, height)
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            return np.empty((0, 2))
        pts = pts[:, :2] + origin
    except Exception:
        try:
            base_vectors = lattice_like.direct_basis().base_vectors()
            pad = max(0.0, oversample - 1.0)
            pts = _lattice_points_from_basis(base_vectors[0], base_vectors[1], extent, pad=pad, max_points=max_points)
        except Exception:
            return np.empty((0, 2))

    if pts.shape[0] == 0:
        return np.empty((0, 2))
    if max_points and pts.shape[0] > max_points:
        stride = max(1, math.ceil(pts.shape[0] / max_points))
        pts = pts[::stride]
    return np.asarray(pts[:, :2] if pts.shape[1] > 2 else pts)


def _build_rotated_lattice(bilayer, transform):
    if not bilayer:
        raise ValueError("Bilayer metadata required")
    if 'a1' in bilayer and 'a2' in bilayer:
        a1 = np.asarray(bilayer['a1'], dtype=float)
        a2 = np.asarray(bilayer['a2'], dtype=float)
    else:
        base_vectors = bilayer['lattice'].direct_basis().base_vectors()
        a1 = np.asarray(base_vectors[0], dtype=float)
        a2 = np.asarray(base_vectors[1], dtype=float)

    transform_mat = np.asarray(transform.to_matrix(), dtype=float)
    rotated_a1 = transform_mat @ a1[:2]
    rotated_a2 = transform_mat @ a2[:2]

    return ml.Lattice2D.from_basis_vectors(
        [float(rotated_a1[0]), float(rotated_a1[1]), 0.0],
        [float(rotated_a2[0]), float(rotated_a2[1]), 0.0],
    )


def _format_sig(value, digits=2):
    if value is None or not math.isfinite(value) or value == 0:
        return "0"
    return f"{value:.{digits}g}"


def _points_in_supercell(a1_vec, a2_vec, cell_vectors, margin=0.05):
    a1 = np.asarray(a1_vec[:2], dtype=float)
    a2 = np.asarray(a2_vec[:2], dtype=float)
    b1, b2 = [np.asarray(vec[:2], dtype=float) for vec in cell_vectors]
    try:
        A_inv = np.linalg.inv(np.column_stack([a1, a2]))
        B_inv = np.linalg.inv(np.column_stack([b1, b2]))
    except np.linalg.LinAlgError:
        return np.empty((0, 2))

    corners = np.array([
        [0.0, 0.0],
        b1,
        b2,
        b1 + b2,
    ])
    ij = (A_inv @ corners.T).T
    i_min = math.floor(ij[:, 0].min()) - 2
    i_max = math.ceil(ij[:, 0].max()) + 2
    j_min = math.floor(ij[:, 1].min()) - 2
    j_max = math.ceil(ij[:, 1].max()) + 2

    points = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            vec = i * a1 + j * a2
            coeff = B_inv @ vec
            if (-margin <= coeff[0] <= 1.0 + margin and
                    -margin <= coeff[1] <= 1.0 + margin):
                points.append(vec)
    if not points:
        return np.empty((0, 2))
    return np.asarray(points)


def _phase1_candidate_geometry(candidate_params):
    geom = {
        'a_value': None,
        'scale': 1.0,
        'moire_length': None,
        'theta_deg': None,
        'hole_radius': None,
        'hole_radius_display': None,
    }
    if not candidate_params:
        return geom

    geom['moire_length'] = candidate_params.get('moire_length')

    a_value = candidate_params.get('a')
    try:
        a_value = float(a_value)
        if math.isfinite(a_value) and a_value > 0:
            geom['a_value'] = a_value
            geom['scale'] = 1.0 / a_value
    except (TypeError, ValueError):
        pass

    theta = candidate_params.get('theta_deg')
    if theta is not None:
        try:
            geom['theta_deg'] = float(theta)
        except (TypeError, ValueError):
            geom['theta_deg'] = theta

    r_ratio = candidate_params.get('r_over_a')
    if r_ratio is not None and geom['a_value']:
        try:
            hole_radius = float(r_ratio) * float(geom['a_value'])
            geom['hole_radius'] = hole_radius
            geom['hole_radius_display'] = hole_radius * geom['scale']
        except (TypeError, ValueError):
            pass

    return geom


def _load_font(size=20, bold=False):
    if ImageFont is None:
        return None
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size=size)
    except Exception:
        return ImageFont.load_default()


def _measure_text(font, text):
    if font is None or not text:
        return (0.0, 0.0)
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    if hasattr(font, "getsize"):
        width, height = font.getsize(text)
        return float(width), float(height)
    return (float(len(text)) * 6.0, 10.0)


def _draw_centered_text(draw, text, center, font, bold=False, fill=TEXT_COLOR):
    if draw is None or not text:
        return
    width, height = _measure_text(font, text)
    x = center[0] - width / 2.0
    y = center[1] - height / 2.0
    offsets = [(0, 0)] if not bold else [(0, 0), (1, 0), (0, 1), (1, 1)]
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=fill)


def _draw_rotated_text(image, text, center, font, angle=90, fill=TEXT_COLOR):
    if Image is None or ImageDraw is None:
        return
    if image is None or not text or font is None:
        return
    width, height = _measure_text(font, text)
    canvas_width = int(max(1, width) + 6)
    canvas_height = int(max(1, height) + 6)
    text_image = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((3, 3), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    x = int(center[0] - rotated.width / 2)
    y = int(center[1] - rotated.height / 2)
    image.alpha_composite(rotated, dest=(x, y))


def _prepare_monolayer_panel(geom, candidate_params, moire_meta):
    if not moire_meta or 'a1_vec' not in moire_meta or 'a2_vec' not in moire_meta:
        return {'descriptor': 'Monolayer Lattice', 'math_label': r'$r/a$ unavailable', 'error': 'No lattice metadata'}

    a1 = np.array(moire_meta['a1_vec'][:2], dtype=float)
    a2 = np.array(moire_meta['a2_vec'][:2], dtype=float)
    cell_norm = max(np.linalg.norm(a1), np.linalg.norm(a2))
    radius = 2.0 * cell_norm
    min_norm = max(1e-9, min(np.linalg.norm(a1), np.linalg.norm(a2)))
    max_index = int(math.ceil((radius + cell_norm) / min_norm)) + 1
    pts = []
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            vec = i * a1 + j * a2
            if np.linalg.norm(vec) <= radius + cell_norm:
                pts.append(vec)
    if not pts:
        return {'descriptor': 'Monolayer Lattice', 'math_label': r'$r/a$ unavailable', 'error': 'Unable to sample lattice'}

    pts = np.asarray(pts) * geom['scale']
    span_display = (radius + cell_norm * 0.25) * geom['scale']
    descriptor = 'Monolayer Lattice'
    if candidate_params and 'r_over_a' in candidate_params:
        math_line = rf"$r/a = {_format_sig(candidate_params['r_over_a'])}$"
    else:
        math_line = r'$r/a$ unavailable'

    vectors = []
    for vec, color, label in [
        (a1 * geom['scale'], (220, 38, 38), r"$\mathbf{a}_1$"),
        (a2 * geom['scale'], (249, 115, 22), r"$\mathbf{a}_2$"),
    ]:
        vectors.append({'end': vec, 'color': color, 'label': label})

    point_groups = [{
        'points': pts,
        'color': MONO_BLUE,
        'alpha': 0.85,
        'radius_data': geom['hole_radius_display'],
        'radius_px': 6.0 if geom['hole_radius_display'] is None else None,
    }]

    axis_labels = (r'$x/a$', r'$y/a$')
    extent = (-span_display, span_display, -span_display, span_display)
    return {
        'descriptor': descriptor,
        'math_label': math_line,
        'axis_labels': axis_labels,
        'extent': extent,
        'point_groups': point_groups,
        'vectors': vectors,
        'polygons': [],
        'legend': None,
    }


def _prepare_bilayer_panel(geom, candidate_params):
    required = {'lattice_type', 'theta_deg', 'a'}
    if not candidate_params or not required.issubset(candidate_params):
        return {'descriptor': 'Twisted Bilayer Lattice', 'math_label': '', 'error': 'Missing lattice parameters'}
    try:
        bilayer = create_twisted_bilayer(
            candidate_params['lattice_type'],
            float(candidate_params['theta_deg']),
            float(candidate_params['a']),
        )
        base_lattice = bilayer['lattice']
        base_vectors = base_lattice.direct_basis().base_vectors()
        a1 = np.asarray(base_vectors[0][:2], dtype=float)
        a2 = np.asarray(base_vectors[1][:2], dtype=float)
        theta = bilayer['theta_rad']
        rot = _rotation_matrix(theta)
        a1_twisted = rot @ a1
        a2_twisted = rot @ a2
        transform = ml.MoireTransformation.twist(theta)
        moire = ml.Moire2D.from_transformation(base_lattice, transform)
        moire_basis = moire.direct_basis().base_vectors()
        b1 = np.asarray(moire_basis[0][:2], dtype=float)
        b2 = np.asarray(moire_basis[1][:2], dtype=float)
    except Exception as exc:
        return {
            'descriptor': 'Twisted Bilayer Lattice',
            'math_label': '',
            'error': f'Moiré build failed: {exc}',
        }

    shift_vectors = [i * b1 + j * b2 for i in range(2) for j in range(2)]
    center_shift = b1 + b2

    def _tile_points(base_points):
        if base_points.size == 0:
            return base_points
        tiles = [base_points + shift for shift in shift_vectors]
        return np.vstack(tiles) - center_shift

    layer1_pts = _tile_points(_points_in_supercell(a1, a2, (b1, b2)))
    layer2_pts = _tile_points(_points_in_supercell(a1_twisted, a2_twisted, (b1, b2)))
    layer1_pts = layer1_pts * geom['scale'] if layer1_pts.size else layer1_pts
    layer2_pts = layer2_pts * geom['scale'] if layer2_pts.size else layer2_pts

    moire_pts = []
    base_nodes = np.array([[0.0, 0.0], b1, b2, b1 + b2])
    for shift in shift_vectors:
        moire_pts.append(base_nodes + shift)
    moire_pts = (np.vstack(moire_pts) - center_shift) * geom['scale']

    polygons = []
    for shift in shift_vectors:
        corners = np.asarray([[0.0, 0.0], b1, b1 + b2, b2], dtype=float) + shift
        corners = (corners - center_shift) * geom['scale']
        polygons.append(corners)

    all_coords = []
    for arr in (layer1_pts, layer2_pts, moire_pts):
        if arr.size:
            all_coords.append(arr)
    if polygons:
        all_coords.append(np.vstack(polygons))
    if all_coords:
        coords = np.vstack(all_coords)
        extent_max = float(np.abs(coords).max()) * 1.05 + 1e-9
    else:
        extent_max = 1.0
    extent = (-extent_max, extent_max, -extent_max, extent_max)

    point_groups = []
    if layer1_pts.size:
        point_groups.append({'points': layer1_pts, 'color': MONO_BLUE, 'alpha': 0.85, 'radius_px': 5.0, 'label': 'Layer 1'})
    if layer2_pts.size:
        point_groups.append({'points': layer2_pts, 'color': LAYER_TWO_ORANGE, 'alpha': 0.85, 'radius_px': 5.0, 'label': 'Layer 2'})
    if moire_pts.size:
        point_groups.append({'points': moire_pts, 'color': MOIRE_PURPLE, 'alpha': 0.95, 'radius_px': 6.0, 'label': 'Moiré nodes'})

    legend = [(grp.get('label'), grp.get('color')) for grp in point_groups if grp.get('label')]
    axis_labels = (r'$x/a$', r'$y/a$')
    theta_line = rf"$\theta = {_format_sig(geom['theta_deg'])}^\circ$" if isinstance(geom['theta_deg'], (int, float)) else r'$\theta$ undefined'

    return {
        'descriptor': 'Twisted Bilayer Lattice',
        'math_label': theta_line,
        'axis_labels': axis_labels,
        'extent': extent,
        'point_groups': point_groups,
        'vectors': [],
        'polygons': polygons,
        'legend': legend,
    }


def _draw_arrowhead(draw, start, end, color, size=10):
    if draw is None:
        return
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return
    angle = math.atan2(dy, dx)
    left = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.line([end, left], fill=color, width=2)
    draw.line([end, right], fill=color, width=2)


def _draw_lattice_panel(image, draw, panel_data, bounds, title_font, subtitle_font, label_font):
    left, top, right, bottom = bounds
    draw.rectangle(bounds, outline=TEXT_COLOR, width=2)
    descriptor = panel_data.get('descriptor', '')
    math_label = panel_data.get('math_label', '')
    mid_x = (left + right) / 2.0
    _draw_centered_text(draw, descriptor, (mid_x, top + 32), title_font, bold=True)
    if math_label:
        _draw_centered_text(draw, math_label, (mid_x, top + 68), subtitle_font)

    if panel_data.get('error'):
        _draw_centered_text(draw, panel_data['error'], ((left + right) / 2.0, (top + bottom) / 2.0), subtitle_font)
        return

    plot_left = left + 80
    plot_right = right - 60
    plot_top = top + 110
    plot_bottom = bottom - 80
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=AXIS_COLOR, width=1)

    extent = panel_data.get('extent')
    if not extent or len(extent) != 4:
        return
    xmin, xmax, ymin, ymax = extent
    if abs(xmax - xmin) < 1e-9 or abs(ymax - ymin) < 1e-9:
        return

    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    def _to_px(pt):
        x = plot_left + (pt[0] - xmin) / (xmax - xmin) * plot_width
        y = plot_bottom - (pt[1] - ymin) / (ymax - ymin) * plot_height
        return (x, y)

    if xmin < 0 < xmax:
        x0 = _to_px((0.0, ymin))[0]
        draw.line([(x0, plot_top), (x0, plot_bottom)], fill=AXIS_COLOR, width=1)
    if ymin < 0 < ymax:
        y0 = _to_px((xmin, 0.0))[1]
        draw.line([(plot_left, y0), (plot_right, y0)], fill=AXIS_COLOR, width=1)

    for polygon in panel_data.get('polygons', []):
        if polygon.size == 0:
            continue
        pts = [_to_px(pt) for pt in polygon]
        pts.append(pts[0])
        draw.line(pts, fill=TEXT_COLOR, width=2)

    origin = _to_px((0.0, 0.0))
    for vec in panel_data.get('vectors', []):
        end = _to_px(vec.get('end', (0.0, 0.0)))
        color = vec.get('color', TEXT_COLOR)
        draw.line([origin, end], fill=color, width=3)
        _draw_arrowhead(draw, origin, end, color, size=10)
        label = vec.get('label')
        if label and label_font:
            draw.text((end[0] + 6, end[1] + 6), label, font=label_font, fill=color)

    for group in panel_data.get('point_groups', []):
        pts = np.asarray(group.get('points'))
        if pts.size == 0:
            continue
        radius_px = group.get('radius_px')
        if not radius_px:
            radius_px = max(2.0, 0.01 * min(plot_width, plot_height))
        radius_data = group.get('radius_data')
        if radius_data:
            radius_px = max(1.5, radius_data / max(1e-9, (xmax - xmin)) * plot_width)
        color = group.get('color', TEXT_COLOR)
        for pt in pts:
            x, y = _to_px(pt)
            draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=color, outline=None)

    legend_entries = panel_data.get('legend') or []
    if legend_entries:
        legend_x = plot_right - 150
        legend_y = plot_top + 10
        for label, color in legend_entries:
            if not label:
                continue
            draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 18], fill=color, outline=TEXT_COLOR)
            draw.text((legend_x + 26, legend_y - 2), label, font=label_font, fill=TEXT_COLOR)
            legend_y += 26

    axis_labels = panel_data.get('axis_labels', ('', ''))
    if axis_labels[0]:
        _draw_centered_text(draw, axis_labels[0], ((plot_left + plot_right) / 2.0, bottom - 32), label_font)
    if axis_labels[1]:
        _draw_rotated_text(image, axis_labels[1], (left + 34, (plot_top + plot_bottom) / 2.0), label_font, angle=90)


def plot_phase1_lattice_panels(cdir, candidate_params=None, moire_meta=None):
    """Render monolayer and twisted bilayer lattices using Pillow."""
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required for lattice visualization. Install it via 'pip install pillow'.")

    geom = _phase1_candidate_geometry(candidate_params)
    mono_panel = _prepare_monolayer_panel(geom, candidate_params, moire_meta)
    bilayer_panel = _prepare_bilayer_panel(geom, candidate_params)

    width, height = 1400, 720
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(32, bold=True)
    subtitle_font = _load_font(24)
    label_font = _load_font(22)

    panels = [mono_panel, bilayer_panel]
    panel_width = width // 2
    padding = 30
    for idx, panel in enumerate(panels):
        left = idx * panel_width + padding
        right = (idx + 1) * panel_width - padding
        bounds = (left, padding, right, height - padding)
        _draw_lattice_panel(image, draw, panel, bounds, title_font, subtitle_font, label_font)

    out_path = Path(cdir) / 'phase1_lattice_visualization.png'
    image.convert('RGB').save(out_path, dpi=(150, 150))


def plot_phase1_fields(cdir, R_grid, V, vg, M_inv, candidate_params=None, moire_meta=None):
    """
    Create visualization of Phase 1 fields
    
    Args:
        cdir: Candidate directory path
        R_grid: Spatial grid [Nx, Ny, 2]
        V: Potential field [Nx, Ny]
        vg: Group velocity [Nx, Ny, 2]
        M_inv: Inverse mass tensor [Nx, Ny, 2, 2]
        candidate_params: Optional candidate parameters for title
        moire_meta: Optional metadata dict with monolayer basis vectors
    """
    geom = _phase1_candidate_geometry(candidate_params)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    x_coords = R_grid[:, 0, 0]
    y_coords = R_grid[0, :, 1]
    extent = [float(x_coords.min()), float(x_coords.max()), float(y_coords.min()), float(y_coords.max())]
    extent_scaled = [val * geom['scale'] for val in extent]
    x_label = r"$R_x/a$" if geom['a_value'] else r"$R_x$"
    y_label = r"$R_y/a$" if geom['a_value'] else r"$R_y$"
    xticks = np.linspace(extent_scaled[0], extent_scaled[1], num=5)
    yticks = np.linspace(extent_scaled[2], extent_scaled[3], num=5)
    theta_deg = geom['theta_deg']
    formatter = FormatStrFormatter("%.2g")

    def _decorate_field_axis(ax):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(extent_scaled[0], extent_scaled[1])
        ax.set_ylim(extent_scaled[2], extent_scaled[3])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    def _set_title(ax, descriptor, math_label):
        if math_label:
            ax.set_title(math_label)
            ax.text(0.5, 1.08, descriptor, transform=ax.transAxes,
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax.set_title(descriptor, fontweight='bold')

    if candidate_params:
        theta_str = f", theta={_format_sig(theta_deg)} deg" if isinstance(theta_deg, (int, float)) else ""
        title_str = (
            f"Candidate {candidate_params.get('candidate_id', '?')}: "
            f"{candidate_params.get('lattice_type', '?')}, "
            f"r/a={_format_sig(candidate_params.get('r_over_a', 0))}, "
            f"ε={_format_sig(candidate_params.get('eps_bg', 0))}{theta_str}"
        )
        if geom['moire_length']:
            if geom['a_value']:
                lm_units = geom['moire_length'] / geom['a_value']
                lm_line = rf"$L_m = {_format_sig(lm_units)}\,a$"
            else:
                lm_line = rf"$L_m = {_format_sig(geom['moire_length'])}$"
            title_str = f"{title_str}\n{lm_line}"
        fig.suptitle(title_str, fontsize=12, fontweight='bold')

    def _add_colorbar(im, ax, label):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, label=label)

    def _log_field(values):
        safe = np.clip(np.abs(values), 1e-12, None)
        return np.log10(safe)
    
    # Plot V(R)
    ax = axes[0, 0]
    vmax = float(V.max())
    vmin = float(V.min())
    im = ax.imshow(V.T, origin='lower', cmap='RdBu_r', extent=extent_scaled, aspect='equal')
    descriptor = "Potential Landscape"
    math_label = rf"$V(R)$ range: [{_format_sig(vmin)}, {_format_sig(vmax)}]"
    _set_title(ax, descriptor, math_label)
    _decorate_field_axis(ax)
    _add_colorbar(im, ax, label=r'$\Delta \omega$')
    
    # Plot |vg(R)|
    ax = axes[0, 1]
    vg_norm = np.linalg.norm(vg, axis=-1)
    im = ax.imshow(vg_norm.T, origin='lower', cmap='viridis', extent=extent_scaled, aspect='equal')
    descriptor = "Group Velocity Magnitude"
    math_label = rf"$|v_g(R)|$ max: {_format_sig(float(vg_norm.max()))}"
    _set_title(ax, descriptor, math_label)
    _decorate_field_axis(ax)
    _add_colorbar(im, ax, label=r'$|v_g|$')
    
    # Plot eigenvalue 1 of M_inv
    ax = axes[1, 0]
    eigvals = np.linalg.eigvalsh(M_inv)
    log_eig_small = _log_field(eigvals[..., 0])
    im = ax.imshow(log_eig_small.T, origin='lower', cmap='plasma', extent=extent_scaled, aspect='equal')
    descriptor = 'Mass Tensor Eigenvalue'
    _set_title(ax, descriptor, r'$\log_{10}\lambda_{M^{-1}}^{(1)}$ (small)')
    _decorate_field_axis(ax)
    _add_colorbar(im, ax, label=r'$\log_{10}\lambda$')
    
    # Plot eigenvalue 2 of M_inv
    ax = axes[1, 1]
    log_eig_large = _log_field(eigvals[..., 1])
    im = ax.imshow(log_eig_large.T, origin='lower', cmap='plasma', extent=extent_scaled, aspect='equal')
    descriptor = 'Mass Tensor Eigenvalue'
    _set_title(ax, descriptor, r'$\log_{10}\lambda_{M^{-1}}^{(2)}$ (large)')
    _decorate_field_axis(ax)
    _add_colorbar(im, ax, label=r'$\log_{10}\lambda$')

    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase1_fields_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def _render_mode_grid(
    cdir,
    R_grid,
    display_fields,
    eigenvalues,
    n_modes,
    candidate_params,
    figure_name,
    title_suffix,
    *,
    cmap: str = "magma",
    symmetric: bool = False,
):
    total_modes = display_fields.shape[0]
    n_plot = min(n_modes, len(eigenvalues), total_modes)
    if n_plot == 0:
        return

    geom = _phase1_candidate_geometry(candidate_params)
    x_coords = R_grid[:, 0, 0]
    y_coords = R_grid[0, :, 1]
    extent = [float(x_coords.min()), float(x_coords.max()), float(y_coords.min()), float(y_coords.max())]
    extent_scaled = [val * geom['scale'] for val in extent]
    x_label = r"$R_x/a$" if geom['a_value'] else r"$R_x$"
    y_label = r"$R_y/a$" if geom['a_value'] else r"$R_y$"
    xticks = np.linspace(extent_scaled[0], extent_scaled[1], num=5)
    yticks = np.linspace(extent_scaled[2], extent_scaled[3], num=5)
    formatter = FormatStrFormatter("%.2g")

    n_rows = 2 if n_plot > 4 else 1
    n_cols = int(math.ceil(n_plot / n_rows))
    fig_width = 3.5 * n_cols
    fig_height = 3.4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    def _decorate(ax):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(extent_scaled[0], extent_scaled[1])
        ax.set_ylim(extent_scaled[2], extent_scaled[3])
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    for i in range(n_plot):
        ax = axes_flat[i]
        field_vals = np.asarray(display_fields[i])
        im = ax.imshow(
            field_vals.T,
            origin='lower',
            cmap=cmap,
            extent=extent_scaled,
            aspect='equal',
        )
        if symmetric:
            vmax = np.max(np.abs(field_vals))
            im.set_clim(-vmax, vmax)
        delta = eigenvalues[i]
        delta_real = float(delta.real if isinstance(delta, complex) else delta)
        title = rf"$\Delta\omega_{{{i}}} = {_format_sig(delta_real, digits=2)}$"
        ax.set_title(title)
        _decorate(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for i in range(n_plot, len(axes_flat)):
        axes_flat[i].axis('off')

    if candidate_params:
        cid = candidate_params.get('candidate_id', '?')
        lattice = candidate_params.get('lattice_type', '?')
        theta = candidate_params.get('theta_deg')
        theta_str = rf", $\theta={_format_sig(theta, digits=2)}^\circ$" if theta is not None else ""
        suptitle = f"Candidate {cid}: {lattice}{theta_str}\n{title_suffix}"
    else:
        suptitle = title_suffix
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(Path(cdir) / figure_name, dpi=220)
    plt.close()


def plot_envelope_modes(cdir, R_grid, F, eigenvalues, n_modes=8, candidate_params=None):
    """Phase 3 cavity mode visualization."""
    display = []
    for field in F[:n_modes]:
        prob = np.abs(field) ** 2
        max_val = float(prob.max())
        if max_val > 0:
            prob /= max_val
        display.append(prob)
    display_arr = np.asarray(display)
    _render_mode_grid(
        cdir,
        R_grid,
        display_arr,
        eigenvalues,
        n_modes,
        candidate_params,
        figure_name='phase3_cavity_modes.png',
        title_suffix=r"Envelope probability densities $|F(R)|^2$",
    )
    
    # Also plot spectrum with modern styling
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(eigenvalues))
    vals = np.asarray(eigenvalues, dtype=float)
    scatter = ax.scatter(idx, vals, c=vals, cmap='viridis', s=80, linewidths=0.8, edgecolors='black')
    ax.set_facecolor('whitesmoke')
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\Delta\omega$')
    ax.set_title('Envelope Spectrum')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label(r'$\Delta\omega$ scale')
    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase3_spectrum.png', dpi=220)
    plt.close()


def plot_phase4_bandstructure(cdir, distances, band_values, ticks):
    """Plot Δω versus Bloch-path distance for Phase 4 validation."""
    cdir = Path(cdir)
    fig, ax = plt.subplots(figsize=(8, 5))
    n_modes = band_values.shape[1]
    for mode in range(n_modes):
        label = f"Mode {mode}" if mode < 3 else None
        ax.plot(distances, band_values[:, mode], label=label)

    tick_list = []
    for label, pos in ticks:
        if not tick_list or abs(tick_list[-1][1] - pos) > 1e-9:
            tick_list.append((label, pos))

    if tick_list:
        ax.set_xticks([pos for _, pos in tick_list])
        ax.set_xticklabels([label for label, _ in tick_list])

    ax.set_xlabel(r"Bloch-path distance |k|")
    ax.set_ylabel(r"Δω (a/c units)")
    ax.set_title("Phase 4: EA minibands along high-symmetry path")
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    fig.tight_layout()
    fig.savefig(cdir / 'phase4_bandstructure.png', dpi=150)
    plt.close(fig)


def plot_phase4_mode_profiles(
    cdir,
    R_grid,
    fields,
    eigenvalues,
    n_modes,
    candidate_params=None,
):
    """Render Γ-point mode densities using the Phase 3 layout."""
    display = []
    for field in fields[:n_modes]:
        prob = np.abs(field) ** 2
        max_val = float(prob.max())
        if max_val > 0:
            prob /= max_val
        display.append(prob)
    display_arr = np.asarray(display)
    _render_mode_grid(
        cdir,
        R_grid,
        display_arr,
        eigenvalues,
        n_modes,
        candidate_params,
        figure_name='phase4_gamma_modes.png',
        title_suffix=r"Phase 4 Γ-point envelopes $|F(R)|^2$",
    )


def plot_phase4_mode_differences(
    cdir,
    R_grid,
    diff_fields,
    eigenvalues,
    n_modes,
    candidate_params=None,
):
    """Render Phase 4 minus Phase 3 probability differences."""
    _render_mode_grid(
        cdir,
        R_grid,
        diff_fields,
        eigenvalues,
        n_modes,
        candidate_params,
        figure_name='phase4_gamma_minus_phase3.png',
        title_suffix=r"Phase 4 − Phase 3 probability difference",
        cmap='RdBu_r',
        symmetric=True,
    )


def make_phase1_plots(cdir, R_grid, V, vg, M_inv, candidate_params=None, moire_meta=None):
    """Wrapper for Phase 1 plotting"""
    plot_phase1_fields(cdir, R_grid, V, vg, M_inv, candidate_params, moire_meta)
    plot_phase1_lattice_panels(cdir, candidate_params, moire_meta)


def plot_phase2_fields(cdir, R_grid, V, M_inv, vg=None):
    """Visualize Phase 2 inputs (potential, curvature, and optional |v_g|)."""
    x_coords = R_grid[:, 0, 0]
    y_coords = R_grid[0, :, 1]
    extent = [float(x_coords.min()), float(x_coords.max()),
              float(y_coords.min()), float(y_coords.max())]
    eigvals = np.linalg.eigvalsh(M_inv)

    n_panels = 4 if vg is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4))
    axes = np.atleast_1d(axes)

    im = axes[0].imshow(V.T, origin='lower', cmap='RdBu_r', extent=extent, aspect='equal')
    axes[0].set_title('V(R) potential')
    axes[0].set_xlabel('R_x')
    axes[0].set_ylabel('R_y')
    plt.colorbar(im, ax=axes[0], shrink=0.8, label='Frequency shift')

    im = axes[1].imshow(eigvals[..., 0].T, origin='lower', cmap='plasma', extent=extent, aspect='equal')
    axes[1].set_title('M⁻¹ eigenvalue 1 (min)')
    axes[1].set_xlabel('R_x')
    axes[1].set_ylabel('R_y')
    plt.colorbar(im, ax=axes[1], shrink=0.8, label='Curvature')

    im = axes[2].imshow(eigvals[..., 1].T, origin='lower', cmap='plasma', extent=extent, aspect='equal')
    axes[2].set_title('M⁻¹ eigenvalue 2 (max)')
    axes[2].set_xlabel('R_x')
    axes[2].set_ylabel('R_y')
    plt.colorbar(im, ax=axes[2], shrink=0.8, label='Curvature')

    if vg is not None:
        vg_norm = np.linalg.norm(np.asarray(vg)[..., :2], axis=-1)
        im = axes[3].imshow(vg_norm.T, origin='lower', cmap='viridis', extent=extent, aspect='equal')
        axes[3].set_title('|v_g(R)|')
        axes[3].set_xlabel('R_x')
        axes[3].set_ylabel('R_y')
        plt.colorbar(im, ax=axes[3], shrink=0.8, label='Group velocity (a/c)')

    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase2_fields_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase4_comparison(cdir, ea_eigs, moire_bands, comparison):
    """
    Plot Phase 4 validation comparison
    
    Args:
        cdir: Candidate directory path
        ea_eigs: EA eigenvalues DataFrame
        moire_bands: Full moiré band structure
        comparison: Comparison DataFrame
    """
    # Placeholder - will be implemented in Phase 4
    pass


def plot_band_structure(bands, candidate_row, save_path=None, ax=None):
    """
    Plot band structure with highlighted candidate k-point and band
    
    Args:
        bands: Band structure data from compute_bandstructure
        candidate_row: Candidate parameters (dict or DataFrame row)
        save_path: Path to save the figure
    """
    freqs = bands['frequencies']  # Shape: (n_k, n_bands)
    k_labels = bands.get('k_labels', []) or []
    k_path = bands.get('k_path')
    label_positions = bands.get('k_label_positions')
    
    created_fig = False
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure
    
    # Plot all bands
    n_k, n_bands = freqs.shape
    if k_path is not None:
        x = np.asarray(k_path)
    else:
        x = np.arange(n_k)
    
    for band_idx in range(n_bands):
        ax.plot(x, freqs[:, band_idx], 'b-', alpha=0.6, linewidth=1)
    
    # Highlight the candidate band and k-point
    target_band = int(candidate_row.get('band_index', 0))
    k_label = candidate_row.get('k_label', 'Γ')
    
    if target_band < n_bands:
        ax.plot(x, freqs[:, target_band], 'r-', linewidth=2, label=f'Band {target_band}')
    
    # Mark high symmetry points with vertical lines
    if k_labels:
        if label_positions is not None and len(label_positions) == len(k_labels):
            x_positions = label_positions
        else:
            num_segments = max(1, len(k_labels) - 1)
            x_positions = np.linspace(x[0], x[-1], len(k_labels))

        y_min, y_max = freqs.min(), freqs.max()
        y_range = y_max - y_min
        y_label = y_max + 0.04 * (y_range if y_range > 0 else 1.0)

        for pos, label in zip(x_positions, k_labels):
            ax.axvline(pos, color='k', linestyle='-', alpha=0.4, linewidth=1.0)
            ax.text(pos, y_label, label, ha='center', va='bottom', fontsize=10, weight='bold')

        # Highlight the target k-point marker
        if target_band < n_bands and k_label in k_labels:
            label_idx = k_labels.index(k_label)
            x_pos = x_positions[label_idx]
            ax.plot(x_pos, np.interp(x_pos, x, freqs[:, target_band]), 'ro',
                    markersize=8, zorder=10,
                    label=f'{k_label}, ω={np.interp(x_pos, x, freqs[:, target_band]):.4f}')
    
    ax.set_xlabel('k-path', fontsize=10)
    ax.set_ylabel('Frequency (c/a)', fontsize=10)
    ax.set_title(f"{candidate_row.get('lattice_type', 'lattice')}, r/a={candidate_row.get('r_over_a', 0):.2f}, ε={candidate_row.get('eps_bg', 0):.1f}", 
                fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x[0], x[-1])

    # Add margins to y-limits for readability
    y_min, y_max = freqs.min(), freqs.max()
    y_range = y_max - y_min
    y_margin = 0.05 * (y_range if y_range > 0 else 1.0)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    if created_fig:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
    elif save_path is not None and fig is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')


def _bg_color_from_eps(eps_bg: float):
    try:
        eps = float(eps_bg)
    except (TypeError, ValueError):
        eps = 0.0
    if eps <= 0:
        return (1.0, 1.0, 1.0)
    eps_clamped = min(13.0, eps)
    t = eps_clamped / 13.0
    dark = np.array([6, 78, 59], dtype=float) / 255.0  # Deep green tone
    white = np.ones(3)
    color = white + t * (dark - white)
    return tuple(np.clip(color, 0.0, 1.0))


def plot_monolayer_unit_cell(geom_params, candidate_row=None, save_path=None, ax=None):
    """Plot a 4x4 chunk of the monolayer crystal using base vectors and hole radius."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        created_fig = True
    else:
        fig = ax.figure

    a1 = np.array(geom_params['a1'][:2], dtype=float)
    a2 = np.array(geom_params['a2'][:2], dtype=float)
    radius = float(geom_params['radius'])
    eps_bg = geom_params.get('eps_bg', 0.0)
    bg_color = _bg_color_from_eps(eps_bg)
    ax.set_facecolor(bg_color)

    grid_size = 4
    half = (grid_size - 1) / 2.0

    all_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            base_origin = (i - half) * a1 + (j - half) * a2
            corners = np.array([
                base_origin,
                base_origin + a1,
                base_origin + a1 + a2,
                base_origin + a2,
            ])
            all_points.append(corners)
            cell_patch = Polygon(corners, closed=True, fill=False, linewidth=0.6, edgecolor='#0f172a', alpha=0.5)
            ax.add_patch(cell_patch)

            circle_center = base_origin
            hole = Circle((float(circle_center[0]), float(circle_center[1])), radius,
                          facecolor='#e2e8f0', edgecolor='#94a3b8', linewidth=0.9)
            ax.add_patch(hole)

    all_points = np.vstack(all_points) if all_points else np.zeros((1, 2))
    extent_min = all_points.min(axis=0) - radius * 1.8
    extent_max = all_points.max(axis=0) + radius * 1.8

    ax.plot([0.0, float(a1[0])], [0.0, float(a1[1])], color='#dc2626', linewidth=1.0)
    ax.plot([0.0, float(a2[0])], [0.0, float(a2[1])], color='#f97316', linewidth=1.0)
    a1_label = a1 * 0.6
    a2_label = a2 * 0.6
    ax.text(float(a1_label[0]), float(a1_label[1]), 'a1', color='#dc2626', fontsize=9)
    ax.text(float(a2_label[0]), float(a2_label[1]), 'a2', color='#f97316', fontsize=9)

    ax.set_xlim(float(extent_min[0]), float(extent_max[0]))
    ax.set_ylim(float(extent_min[1]), float(extent_max[1]))
    ax.set_aspect('equal')
    ax.set_xlabel('x (a units)')
    ax.set_ylabel('y (a units)')
    title = 'Monolayer crystal chunk'
    if candidate_row is not None:
        title += f"\n r/a={candidate_row.get('r_over_a', 0):.3f}, ε={candidate_row.get('eps_bg', 0):.2f}"
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.35, alpha=0.35)

    if created_fig:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
    elif save_path is not None and fig is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')


def plot_optimizer_candidate_summary(geom_params, bands, candidate_row, save_path):
    """Create a side-by-side plot of band diagram and monolayer geometry."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_band_structure(bands, candidate_row, ax=axes[0])
    plot_monolayer_unit_cell(geom_params, candidate_row, ax=axes[1])
    fig.suptitle(
        f"Optimizer pick: {candidate_row.get('lattice_type', '')} | "
        f"band {int(candidate_row.get('band_index', 0))} at {candidate_row.get('optimization_k_label', candidate_row.get('k_label', '?'))}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def plot_top_candidates_grid(top_candidates, bands_list, save_path, n_cols=4):
    """
    Create a grid of band diagrams for top candidates
    
    Args:
        top_candidates: DataFrame of top candidates
        bands_list: List of band structure data for each candidate
        save_path: Path to save the figure
        n_cols: Number of columns in the grid
    """
    n_candidates = len(top_candidates)
    n_rows = int(np.ceil(n_candidates / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_candidates > 1 else [axes]
    
    for idx, (_, row) in enumerate(top_candidates.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        bands = bands_list[idx]
        freqs = bands['frequencies']
        k_labels = bands.get('k_labels', []) or []
        k_path = bands.get('k_path')
        label_positions = bands.get('k_label_positions')
        
        # Plot bands
        n_k, n_bands = freqs.shape
        if k_path is not None:
            x = np.asarray(k_path)
        else:
            x = np.arange(n_k)
        
        for band_idx in range(n_bands):
            ax.plot(x, freqs[:, band_idx], 'b-', alpha=0.5, linewidth=0.8)
        
        # Highlight target band
        target_band = int(row['band_index'])
        k_label = row['k_label']
        
        if target_band < n_bands:
            ax.plot(x, freqs[:, target_band], 'r-', linewidth=1.5)
        
        # Mark k-point
        if k_labels and k_label in k_labels:
            if label_positions is not None and len(label_positions) == len(k_labels):
                x_pos = label_positions[k_labels.index(k_label)]
            else:
                num_segments = max(1, len(k_labels) - 1)
                x_pos = np.linspace(x[0], x[-1], len(k_labels))[k_labels.index(k_label)]
            if target_band < n_bands:
                y_val = np.interp(x_pos, x, freqs[:, target_band])
                ax.plot(x_pos, y_val, 'ro', markersize=6)
        
        # Mark high symmetry points with vertical lines
        if k_labels:
            if label_positions is not None and len(label_positions) == len(k_labels):
                x_positions = label_positions
            else:
                x_positions = np.linspace(x[0], x[-1], len(k_labels))

            y_min, y_max = freqs.min(), freqs.max()
            y_range = y_max - y_min
            y_label = y_max + 0.03 * (y_range if y_range > 0 else 1.0)

            for pos, label in zip(x_positions, k_labels):
                ax.axvline(pos, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
                # Only label first/last to avoid crowding in grid
                if label == k_labels[0] or label == k_labels[-1]:
                    ax.text(pos, y_label, label, ha='center', va='bottom', fontsize=7)
        
        # Title with key info
        title = (f"#{row['candidate_id']}: {row['lattice_type']}\n"
            f"r/a={row['r_over_a']:.2f}, ε={row['eps_bg']:.1f}, {k_label}-band{target_band}\n"
                f"Score={row['S_total']:.3f}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('k-path', fontsize=8)
        ax.set_ylabel('ω (c/a)', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlim(x[0], x[-1])
        
        # Set y-limits to show all bands clearly
        y_margin = 0.05 * (freqs.max() - freqs.min())
        ax.set_ylim(freqs.min() - y_margin, freqs.max() + y_margin)
    
    # Hide unused subplots
    for idx in range(n_candidates, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved candidate grid plot to: {save_path}")
