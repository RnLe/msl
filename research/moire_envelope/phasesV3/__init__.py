"""
phasesV3 - Multi-Band Envelope Approximation Pipeline (MPB-based)

This package implements the V3 multi-band envelope approximation
for moiré photonic crystals using MPB (MIT Photonic Bands), including:

- Berry connection (non-Abelian gauge field)
- Born-Huang potential
- Multi-band subspace tracking
- Gauge-covariant derivatives

Theory reference: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md

Phases:
    phase0_library_v3: Band library search with multi-band subspace info
    phase1_mpb_v3: Multi-band local Bloch problems (MPB)
    phase2_mpb_v3: Berry connection and Born-Huang potential
    phase3_mpb_v3: Multi-band envelope Hamiltonian solver
    phase5_meep_v3: Meep FDTD validation & Q-factor analysis

Usage:
    python -m phasesV3.phase0_library_v3
    python -m phasesV3.phase1_mpb_v3
    python -m phasesV3.phase2_mpb_v3
    python -m phasesV3.phase3_mpb_v3
    python -m phasesV3.phase5_meep_v3 --test
"""

from .phase0_library_v3 import run_phase0_library_v3
from .phase1_mpb_v3 import run_phase1_v3
from .phase2_mpb_v3 import run_phase2_v3
from .phase3_mpb_v3 import run_phase3_v3

try:
    from .phase5_meep_v3 import run_phase5_v3, run_test_mode
except (ImportError, SystemExit):
    # meep may not be installed — skip phase5 exports
    pass

__version__ = "3.0.0"
__all__ = [
    "run_phase0_library_v3",
    "run_phase1_v3", 
    "run_phase2_v3",
    "run_phase3_v3",
]
