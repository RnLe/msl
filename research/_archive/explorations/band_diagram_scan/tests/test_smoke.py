from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from bandlib.cli import run_scan_command
from bandlib.mpb_runner import SolverName
from bandlib.storage import Status


def test_smoke_scan(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config/scan_tiny_debug.yml"
    h5_path = tmp_path / "bandlib_smoke.h5"

    run_scan_command(
        config_path=config_path,
        h5_path=h5_path,
        limit=2,
        progress=False,
        solver=SolverName.SYNTHETIC,
    )

    assert h5_path.exists(), "Scan should create the HDF5 library file"

    with h5py.File(h5_path, "r") as h5:
        scan = h5["scans"]["tiny_debug"]
        status = scan["status"]["geom_status"][...]
        completed = np.argwhere(status == Status.COMPLETE)
        assert completed.shape[0] == 2, "Limit should mark two geometries complete"

        freq = scan["data"]["freq"]
        for idx in completed:
            bands = freq[
                idx[0],
                idx[1],
                idx[2],
                idx[3],
                idx[4],
                :, :,
            ]
            assert np.all(np.isfinite(bands)), "Completed entries should contain real data"
