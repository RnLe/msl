#!/usr/bin/env python3
"""
Non-commensurate FDFD study at X-point (TM, square lattice).

Uses build_moire_supercell_eps to construct a SINGLE moiré cell at
exact twist angles (not commensurate approximations).  The FDFD solver
applies Bloch-periodic boundary conditions as usual.

This module provides helpers for generating eps grids and running solves.
"""

import os
import sys
import numpy as np

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.abspath(os.path.join(
    STUDY_DIR, '..', '..', '..', 'research',
    'moire_envelope', 'thesis_results'))
sys.path.insert(0, THESIS_RESULTS)

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_moire_supercell_eps

# ── physical parameters (same as commensurate study) ────────────────
R_OVER_A = 0.2
EPS_ROD  = 8.9
EPS_BG   = 1.0
A        = 1.0  # lattice constant

# Exact twist angles in degrees
ANGLES = {
    '8deg': 8.0,
    '4deg': 4.0,
    '2deg': 2.0,
    '1deg': 1.0,
}

# Monolayer X-point in Cartesian k-space  (a = 1 ⇒ q_X = π/a = π)
Q_X = np.array([np.pi, 0.0])
