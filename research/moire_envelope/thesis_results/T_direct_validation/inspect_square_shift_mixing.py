#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MEEP_NUM_THREADS', '1')

import meep as mp

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from common.geometry import high_symmetry_points
from phasesV3.phase1_mpb_v3 import create_mpb_geometry, create_mpb_solver


def get_bz_path_vertices(lattice_type):
    if lattice_type in ('honeycomb', 'hex', 'triangular'):
        return [
            ('Γ', np.array([0.0, 0.0, 0.0], dtype=float)),
            ('K', np.array([2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float)),
            ('M', np.array([0.5, 0.0, 0.0], dtype=float)),
            ('Γ', np.array([0.0, 0.0, 0.0], dtype=float)),
        ]
    return high_symmetry_points(lattice_type)


def get_probe_k_point(lattice_type):
    if lattice_type in ('honeycomb', 'hex', 'triangular'):
        return 'K', np.array([2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float)
    return 'M', np.array([0.5, 0.5, 0.0], dtype=float)


def run_mpb_at_k(mode_solver, k_point, polarization):
    """Run MPB for a single k-point while suppressing noisy solver output."""
    mode_solver.k_points = [mp.Vector3(*k_point)]
    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        if polarization.upper() == 'TM':
            mode_solver.run_tm()
        elif polarization.upper() == 'TE':
            mode_solver.run_te()
        else:
            raise ValueError(f'Unsupported polarization: {polarization}')
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
    return np.array(mode_solver.all_freqs[0], dtype=float)


def interpolate_path(point_list, k_interp):
    k_points = [mp.Vector3(*point) for point in point_list]
    interpolated = mp.interpolate(k_interp, k_points)
    return np.array([[kp.x, kp.y, kp.z] for kp in interpolated], dtype=float)


def build_band_path(lattice_type, k_interp):
    hs = get_bz_path_vertices(lattice_type)
    labels = [label for label, _ in hs]
    points = [vec for _, vec in hs]
    interpolated = interpolate_path(points, k_interp)

    k_path = np.zeros(len(interpolated), dtype=float)
    for idx in range(1, len(interpolated)):
        k_path[idx] = k_path[idx - 1] + np.linalg.norm(interpolated[idx] - interpolated[idx - 1])

    k_label_positions = [0.0]
    segment_end_idx = [0]
    cumulative = 0.0
    for idx in range(len(points) - 1):
        cumulative += np.linalg.norm(points[idx + 1] - points[idx])
        nearest = int(np.argmin(np.abs(k_path - cumulative)))
        segment_end_idx.append(nearest)
        k_label_positions.append(float(k_path[nearest]))

    return {
        'labels': labels,
        'points': np.array(points, dtype=float),
        'k_points': interpolated,
        'k_path': k_path,
        'k_label_positions': np.array(k_label_positions, dtype=float),
        'segment_end_idx': segment_end_idx,
    }


def compute_band_diagram(lattice_type, r_over_a, eps_bg, eps_hole, resolution, num_bands, k_interp, delta_frac=None):
    path = build_band_path(lattice_type, k_interp)
    if delta_frac is None:
        delta_frac = np.array([0.0, 0.0], dtype=float)
    geometry, lattice, bg_eps = create_mpb_geometry(
        lattice_type=lattice_type,
        r_over_a=r_over_a,
        eps_bg=eps_bg,
        eps_hole=eps_hole,
        delta_frac=np.array(delta_frac, dtype=float),
    )

    band_data = {}
    for polarization in ('TE', 'TM'):
        solver = create_mpb_solver(
            geometry=geometry,
            lattice=lattice,
            eps_bg=bg_eps,
            num_bands=num_bands,
            resolution=resolution,
            polarization=polarization,
        )
        freqs = np.zeros((len(path['k_points']), num_bands), dtype=float)
        for idx, k_point in enumerate(path['k_points']):
            values = run_mpb_at_k(solver, k_point, polarization)
            freqs[idx, :] = values[:num_bands]
        band_data[polarization] = freqs

    return path, band_data


def build_triangle_shift_path(samples_per_segment):
    vertices = [
        np.array([0.0, 0.0], dtype=float),
        np.array([0.0, 0.5], dtype=float),
        np.array([0.5, 0.5], dtype=float),
        np.array([0.0, 0.0], dtype=float),
    ]
    points = []
    segment_ids = []
    for seg_idx in range(len(vertices) - 1):
        start = vertices[seg_idx]
        end = vertices[seg_idx + 1]
        endpoint = seg_idx == len(vertices) - 2
        ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=endpoint)
        for t in ts:
            points.append((1.0 - t) * start + t * end)
            segment_ids.append(seg_idx)
    return np.array(points, dtype=float), np.array(vertices, dtype=float), np.array(segment_ids, dtype=int)


def render_triangle_gif_frame(path_data, band_data, triangle_vertices, shift_points, frame_idx, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_band, ax_shift = axes

    te_colors = plt.cm.Blues(np.linspace(0.5, 0.9, band_data['TE'].shape[1]))
    tm_colors = plt.cm.Oranges(np.linspace(0.5, 0.9, band_data['TM'].shape[1]))
    for band_idx in range(band_data['TE'].shape[1]):
        ax_band.plot(path_data['k_path'], band_data['TE'][:, band_idx], color=te_colors[band_idx], lw=2.0)
        ax_band.plot(
            path_data['k_path'],
            band_data['TM'][:, band_idx],
            color=tm_colors[band_idx],
            lw=2.0,
            linestyle='--',
        )
    for xpos in path_data['k_label_positions']:
        ax_band.axvline(xpos, color='0.82', lw=0.8)
    ax_band.set_xticks(path_data['k_label_positions'])
    ax_band.set_xticklabels(path_data['labels'])
    ax_band.set_ylabel('Frequency $\\omega a / 2\\pi c$')
    current = shift_points[frame_idx]
    ax_band.set_title(f'Band diagram at shift $\\delta=({current[0]:.3f},{current[1]:.3f})$')
    ax_band.grid(True, axis='y', alpha=0.2)
    handles = [
        plt.Line2D([0], [0], color=te_colors[0], lw=2.0, label='TE'),
        plt.Line2D([0], [0], color=tm_colors[0], lw=2.0, linestyle='--', label='TM'),
    ]
    ax_band.legend(handles=handles, frameon=False, loc='upper left')

    ax_shift.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color='0.75', lw=1.5)
    ax_shift.scatter(triangle_vertices[:-1, 0], triangle_vertices[:-1, 1], color='0.45', s=30)
    traversed = shift_points[: frame_idx + 1]
    ax_shift.plot(traversed[:, 0], traversed[:, 1], color='#2ca02c', lw=2.5)
    ax_shift.scatter(current[0], current[1], color='#d62728', s=80, zorder=5)
    ax_shift.set_xlim(-0.03, 0.53)
    ax_shift.set_ylim(-0.03, 0.53)
    ax_shift.set_aspect('equal', adjustable='box')
    ax_shift.set_xlabel('$\\delta_x$')
    ax_shift.set_ylabel('$\\delta_y$')
    ax_shift.set_title('Triangle walk in stacking-shift space')
    ax_shift.grid(True, alpha=0.2)
    ax_shift.text(0.02, 0.96, f'frame {frame_idx + 1}/{len(shift_points)}', transform=ax_shift.transAxes, va='top')

    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def create_triangle_band_gif(
    lattice_type,
    r_over_a,
    eps_bg,
    eps_hole,
    resolution,
    num_bands,
    k_interp,
    samples_per_segment,
    fps,
    output_dir,
):
    shift_points, triangle_vertices, _segment_ids = build_triangle_shift_path(samples_per_segment)
    frames_dir = output_dir / 'triangle_gif_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    gif_band_data = []
    frame_paths = []
    path_data = None
    for frame_idx, delta_frac in enumerate(shift_points):
        path_data, band_data = compute_band_diagram(
            lattice_type=lattice_type,
            r_over_a=r_over_a,
            eps_bg=eps_bg,
            eps_hole=eps_hole,
            resolution=resolution,
            num_bands=num_bands,
            k_interp=k_interp,
            delta_frac=delta_frac,
        )
        gif_band_data.append({
            'delta_frac': delta_frac.copy(),
            'TE': band_data['TE'].copy(),
            'TM': band_data['TM'].copy(),
        })
        frame_path = frames_dir / f'frame_{frame_idx:03d}.png'
        render_triangle_gif_frame(path_data, band_data, triangle_vertices, shift_points, frame_idx, frame_path)
        frame_paths.append(frame_path)

    gif_path = output_dir / f'{lattice_type.lower()}_shift_triangle_band_diagram.gif'
    images = [Image.open(frame_path) for frame_path in frame_paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000 / fps)),
        loop=0,
    )
    for image in images:
        image.close()

    gif_npz = output_dir / 'triangle_gif_band_data.npz'
    np.savez_compressed(
        gif_npz,
        k_path=path_data['k_path'],
        k_points=path_data['k_points'],
        k_label_positions=path_data['k_label_positions'],
        shift_points=shift_points,
        te_freqs=np.stack([entry['TE'] for entry in gif_band_data], axis=0),
        tm_freqs=np.stack([entry['TM'] for entry in gif_band_data], axis=0),
    )
    return gif_path, gif_npz, frames_dir, shift_points


def compute_probe_point_shift_scan(lattice_type, r_over_a, eps_bg, eps_hole, resolution, shift_grid_n):
    probe_label, probe_k = get_probe_k_point(lattice_type)
    rows = []
    sample_index = 0

    for ix in range(shift_grid_n):
        for iy in range(shift_grid_n):
            delta_frac = np.array([ix / shift_grid_n, iy / shift_grid_n], dtype=float)
            geometry, lattice, bg_eps = create_mpb_geometry(
                lattice_type=lattice_type,
                r_over_a=r_over_a,
                eps_bg=eps_bg,
                eps_hole=eps_hole,
                delta_frac=delta_frac,
            )

            for polarization in ('TE', 'TM'):
                solver = create_mpb_solver(
                    geometry=geometry,
                    lattice=lattice,
                    eps_bg=bg_eps,
                    num_bands=2,
                    resolution=resolution,
                    polarization=polarization,
                )
                freqs = run_mpb_at_k(solver, probe_k, polarization)[:2]
                for local_band_idx, omega in enumerate(freqs, start=1):
                    band_label = f'{polarization}{local_band_idx}'
                    rows.append(
                        {
                            'sample_index': sample_index,
                            'ix': ix,
                            'iy': iy,
                            'delta_x': float(delta_frac[0]),
                            'delta_y': float(delta_frac[1]),
                            'probe_k_label': probe_label,
                            'polarization': polarization,
                            'local_band_index': local_band_idx,
                            'band_label': band_label,
                            'omega': float(omega),
                        }
                    )
            sample_index += 1

    return pd.DataFrame(rows)


def plot_band_diagram(path_data, band_data, output_path, lattice_type):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    te_colors = plt.cm.Blues(np.linspace(0.45, 0.9, band_data['TE'].shape[1]))
    tm_colors = plt.cm.Oranges(np.linspace(0.45, 0.9, band_data['TM'].shape[1]))

    for band_idx in range(band_data['TE'].shape[1]):
        ax.plot(
            path_data['k_path'],
            band_data['TE'][:, band_idx],
            color=te_colors[band_idx],
            lw=1.8,
            label=f'TE{band_idx + 1}',
        )
        ax.plot(
            path_data['k_path'],
            band_data['TM'][:, band_idx],
            color=tm_colors[band_idx],
            lw=1.8,
            linestyle='--',
            label=f'TM{band_idx + 1}',
        )

    for xpos in path_data['k_label_positions']:
        ax.axvline(xpos, color='0.8', lw=0.8)

    ax.set_xticks(path_data['k_label_positions'])
    ax.set_xticklabels(path_data['labels'])
    ax.set_ylabel('Frequency $\\omega a / 2\\pi c$')
    ax.set_title(f'{lattice_type.capitalize()} bilayer reference band diagram at zero stacking shift')
    ax.grid(True, axis='y', alpha=0.2)
    ax.legend(ncol=2, fontsize=9, frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_shift_eigenvalues(scan_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    probe_label = scan_df['probe_k_label'].iloc[0]
    palette = {
        'TE1': '#1f77b4',
        'TE2': '#5fa2dd',
        'TM1': '#d95f02',
        'TM2': '#f2a65a',
    }

    for band_label, group in scan_df.groupby('band_label', sort=False):
        ax.scatter(
            group['sample_index'],
            group['omega'],
            s=28,
            alpha=0.85,
            color=palette[band_label],
            label=band_label,
            edgecolors='none',
        )

    ax.set_xlabel('Stacking-shift sample index (row-major over 10×10 grid)')
    ax.set_ylabel(f'Frequency at {probe_label}')
    ax.set_title(f'First two TE and TM eigenvalues across the stacking-shift grid at {probe_label}')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Inspect square-lattice band mixing under stacking shifts.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--lattice-type', default='square')
    parser.add_argument('--r-over-a', type=float, default=0.2)
    parser.add_argument('--eps-bg', type=float, default=1.0)
    parser.add_argument('--eps-hole', type=float, default=8.9)
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--num-bands', type=int, default=4)
    parser.add_argument('--k-interp', type=int, default=19)
    parser.add_argument('--shift-grid', type=int, default=10)
    parser.add_argument('--make-triangle-gif', action='store_true')
    parser.add_argument('--triangle-samples-per-segment', type=int, default=20)
    parser.add_argument('--gif-fps', type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_data, band_data = compute_band_diagram(
        lattice_type=args.lattice_type,
        r_over_a=args.r_over_a,
        eps_bg=args.eps_bg,
        eps_hole=args.eps_hole,
        resolution=args.resolution,
        num_bands=args.num_bands,
        k_interp=args.k_interp,
    )

    scan_df = compute_probe_point_shift_scan(
        lattice_type=args.lattice_type,
        r_over_a=args.r_over_a,
        eps_bg=args.eps_bg,
        eps_hole=args.eps_hole,
        resolution=args.resolution,
        shift_grid_n=args.shift_grid,
    )

    prefix = args.lattice_type.lower()
    band_plot = args.output_dir / f'{prefix}_bilayer_band_diagram_te_tm.png'
    shift_plot = args.output_dir / f'{prefix}_probe_shift_scan_colored_by_band.png'
    csv_path = args.output_dir / f'{prefix}_probe_shift_scan.csv'
    metadata_path = args.output_dir / 'inspection_metadata.json'
    npz_path = args.output_dir / 'band_diagram_data.npz'
    gif_path = None
    gif_npz = None
    frames_dir = None
    triangle_shift_points = None

    plot_band_diagram(path_data, band_data, band_plot, args.lattice_type)
    plot_shift_eigenvalues(scan_df, shift_plot)
    scan_df.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        k_path=path_data['k_path'],
        k_points=path_data['k_points'],
        k_label_positions=path_data['k_label_positions'],
        te_freqs=band_data['TE'],
        tm_freqs=band_data['TM'],
    )

    if args.make_triangle_gif:
        gif_path, gif_npz, frames_dir, triangle_shift_points = create_triangle_band_gif(
            lattice_type=args.lattice_type,
            r_over_a=args.r_over_a,
            eps_bg=args.eps_bg,
            eps_hole=args.eps_hole,
            resolution=args.resolution,
            num_bands=args.num_bands,
            k_interp=args.k_interp,
            samples_per_segment=args.triangle_samples_per_segment,
            fps=args.gif_fps,
            output_dir=args.output_dir,
        )

    metadata = {
        'lattice_type': args.lattice_type,
        'r_over_a': args.r_over_a,
        'eps_bg': args.eps_bg,
        'eps_hole': args.eps_hole,
        'resolution': args.resolution,
        'num_bands_per_polarization': args.num_bands,
        'k_interp': args.k_interp,
        'shift_grid': args.shift_grid,
        'k_labels': path_data['labels'],
        'probe_k': {
            'label': scan_df['probe_k_label'].iloc[0],
            'vector': get_probe_k_point(args.lattice_type)[1].tolist(),
        },
        'stacking_shift_rule': 'delta_frac = [ix / N, iy / N] with N = shift_grid, matching phasesV3 row-major registry sampling',
        'triangle_gif': {
            'enabled': bool(args.make_triangle_gif),
            'samples_per_segment': args.triangle_samples_per_segment,
            'fps': args.gif_fps,
            'path_vertices': [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]],
            'num_frames': None if triangle_shift_points is None else int(len(triangle_shift_points)),
        },
        'files': {
            'band_diagram_plot': str(band_plot),
            'shift_scan_plot': str(shift_plot),
            'shift_scan_csv': str(csv_path),
            'band_diagram_npz': str(npz_path),
        },
    }
    if gif_path is not None:
        metadata['files']['triangle_gif'] = str(gif_path)
        metadata['files']['triangle_gif_band_data'] = str(gif_npz)
        metadata['files']['triangle_gif_frames_dir'] = str(frames_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f'Saved band diagram to {band_plot}')
    print(f'Saved shift scan plot to {shift_plot}')
    print(f'Saved shift scan table to {csv_path}')
    if gif_path is not None:
        print(f'Saved triangle-path GIF to {gif_path}')


if __name__ == '__main__':
    main()