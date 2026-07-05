from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Deque, Optional, cast
import time

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .axes import GeometryIndices, build_axes
from .config import load_scan_config
from .mpb_runner import MPBUnavailableError, SolverName, get_band_solver
from .storage import (
    Status,
    ensure_scan,
    list_pending_indices,
    mark_status,
    summarize_status,
    write_frequency,
)

console = Console()
app = typer.Typer(help="Utilities for managing the band-diagram scan library.")


def _format_duration(seconds: float) -> str:
    if seconds == float("inf") or np.isnan(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_scan_command(
    config_path: Path | str,
    h5_path: Path | str = Path("data/band_library.h5"),
    limit: Optional[int] = None,
    dry_run: bool = False,
    progress: bool = True,
    solver: SolverName = SolverName.AUTO,
) -> None:
    config, raw_yaml, config_hash = load_scan_config(config_path)
    axes = build_axes(config)
    handle = ensure_scan(h5_path, config, axes, raw_yaml, config_hash)
    solver_fn, resolved_solver = get_band_solver(solver)

    with h5py.File(handle.file_path, "r+") as h5:
        scans_root = h5["scans"]
        if not isinstance(scans_root, h5py.Group):  # pragma: no cover - defensive
            raise TypeError("/scans must be an HDF5 group")
        scan_obj = scans_root[handle.scan_id]
        if not isinstance(scan_obj, h5py.Group):  # pragma: no cover - defensive
            raise TypeError("Scan entry must be an HDF5 group")
        scan_grp = cast(h5py.Group, scan_obj)
        pending_indices, status_snapshot = list_pending_indices(scan_grp)
        total_geometries = status_snapshot.size
        completed_count = int(np.count_nonzero(status_snapshot == Status.COMPLETE))
        pending = pending_indices
        if limit is not None:
            pending = pending[:limit]

        if dry_run:
            console.print(
                f"[bold cyan]Scan {handle.scan_id}[/] would process {len(pending)} geometries (limit={limit}) using solver={resolved_solver.value}."
            )
            counts = summarize_status(scan_grp)
            for status, count in counts.items():
                console.print(f"  {status.name.title():<12}: {count}")
            return

        if pending.size == 0:
            console.print(f"[green]No pending geometries for scan {handle.scan_id}.[/]")
            return

        status_grp = scan_grp["status"]
        data_grp = scan_grp["data"]
        if not isinstance(status_grp, h5py.Group) or not isinstance(data_grp, h5py.Group):  # pragma: no cover
            raise TypeError("Scan storage layout corrupted (status/data groups missing)")
        status_ds_obj = status_grp["geom_status"]
        freq_ds_obj = data_grp["freq"]
        if not isinstance(status_ds_obj, h5py.Dataset) or not isinstance(freq_ds_obj, h5py.Dataset):  # pragma: no cover
            raise TypeError("Scan storage layout corrupted (datasets missing)")
        status_ds = cast(h5py.Dataset, status_ds_obj)
        freq_ds = cast(h5py.Dataset, freq_ds_obj)
        console.print(
            f"[cyan]Using solver '{resolved_solver.value}' for scan {handle.scan_id} "
            f"(scheduled={len(pending)}, total={total_geometries}, already_complete={completed_count}).[/]"
        )

        progress_bar = None
        eta_placeholder = "ETA --:--:--"
        durations: Deque[float] | None = deque(maxlen=10) if progress else None
        last_timestamp = time.perf_counter()
        if progress:
            progress_bar = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                TaskProgressColumn(show_speed=True),
                TextColumn("{task.completed:,}/{task.total:,}", justify="right"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[eta]}", justify="left"),
                console=console,
                transient=False,
            )

        processed = 0
        cm = progress_bar if progress_bar is not None else nullcontext()
        with cm as active_progress:
            task_id = None
            if progress_bar is not None:
                task_id = progress_bar.add_task(
                    f"{handle.scan_id}",
                    total=total_geometries,
                    completed=completed_count,
                    eta=eta_placeholder,
                )

            for idx_array in pending:
                idx = GeometryIndices(*map(int, idx_array))
                params = axes.indices_to_params(idx)
                mark_status(status_ds, idx, Status.IN_PROGRESS)
                try:
                    k_path = axes.k_path_for(params.lattice_type)
                    bands = solver_fn(params, config.mpb, k_path)
                    write_frequency(freq_ds, idx, bands)
                    mark_status(status_ds, idx, Status.COMPLETE)
                    processed += 1
                    if progress_bar is not None and task_id is not None:
                        now = time.perf_counter()
                        if durations is not None:
                            durations.append(now - last_timestamp)
                        last_timestamp = now
                        remaining = max(0, total_geometries - (completed_count + processed))
                        if durations and len(durations) > 0:
                            avg_duration = sum(durations) / len(durations)
                            eta_seconds = avg_duration * remaining
                            eta_value = f"ETA {_format_duration(eta_seconds)}"
                        else:
                            eta_value = eta_placeholder
                        progress_bar.update(
                            task_id,
                            advance=1,
                            eta=eta_value,
                        )
                except Exception:
                    mark_status(status_ds, idx, Status.FAILED)
                    raise
                finally:
                    h5.flush()

        final_completed = completed_count + processed
        console.print(
            f"[green]Completed {processed} geometries for scan {handle.scan_id} "
            f"({final_completed}/{total_geometries} total complete).[/]"
        )


@app.command(name="run")
def run_command(
    config_path: Path = typer.Argument(..., help="Path to the YAML scan configuration."),
    h5_path: Path = typer.Option(Path("data/band_library.h5"), help="Output HDF5 file."),
    limit: Optional[int] = typer.Option(None, help="Optional cap on the number of geometries."),
    dry_run: bool = typer.Option(False, help="Only report pending work without running simulations."),
    no_progress: bool = typer.Option(False, help="Disable tqdm progress bars."),
    solver: SolverName = typer.Option(
        SolverName.AUTO,
        case_sensitive=False,
        help="Band solver backend (auto, mpb, synthetic).",
    ),
) -> None:
    """Run (or preview) a scan defined by a YAML configuration."""

    try:
        run_scan_command(
            config_path,
            h5_path,
            limit,
            dry_run,
            progress=not no_progress,
            solver=solver,
        )
    except MPBUnavailableError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


@app.command(name="inspect")
def inspect_command(
    h5_path: Path = typer.Option(Path("data/band_library.h5"), help="Band library file."),
    scan_id: Optional[str] = typer.Option(None, help="Specific scan_id to inspect."),
) -> None:
    """Print a status summary for one or all scans."""

    h5_path = Path(h5_path)
    if not h5_path.exists():
        console.print(f"[red]File {h5_path} does not exist.[/]")
        raise typer.Exit(code=1)

    with h5py.File(h5_path, "r") as h5:
        scans_node = h5.get("scans")
        if scans_node is None:
            console.print("[yellow]No scans recorded yet.[/]")
            return
        if not isinstance(scans_node, h5py.Group):  # pragma: no cover - defensive
            raise TypeError("/scans must be a group")

        scans = cast(h5py.Group, scans_node)
        targets = [scan_id] if scan_id else list(scans.keys())
        table = Table("scan_id", "pending", "complete", "in_progress", "failed")
        for sid in targets:
            if sid not in scans:
                console.print(f"[yellow]Scan {sid} not found.[/]")
                continue
            scan_node = scans[sid]
            if not isinstance(scan_node, h5py.Group):  # pragma: no cover
                raise TypeError("Scan entry must be a group")
            counts = summarize_status(scan_node)
            table.add_row(
                sid,
                str(counts.get(Status.PENDING, 0)),
                str(counts.get(Status.COMPLETE, 0)),
                str(counts.get(Status.IN_PROGRESS, 0)),
                str(counts.get(Status.FAILED, 0)),
            )
        console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
