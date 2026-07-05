"""
blaze_phasesV3 - Multi-Band Envelope Approximation Pipeline

This package implements the V3 multi-band envelope approximation
for moiré photonic crystals, including:

- Berry connection (non-Abelian gauge field)
- Born-Huang potential
- Multi-band subspace tracking
- Gauge-covariant derivatives

Theory reference: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md

Phases:
    phase0_library_v3: Band library search with multi-band subspace info
    phase1_blaze_v3: Multi-band local Bloch problems (BLAZE)
    phase2_blaze_v3: Berry connection and Born-Huang potential
    phase3_blaze_v3: Multi-band envelope Hamiltonian solver

Usage:
    python -m blaze_phasesV3.phase0_library_v3
    python -m blaze_phasesV3.phase1_blaze_v3
    python -m blaze_phasesV3.phase2_blaze_v3
    python -m blaze_phasesV3.phase3_blaze_v3
"""

from .phase0_library_v3 import run_phase0_library_v3
from .phase1_blaze_v3 import run_phase1_v3
from .phase2_blaze_v3 import run_phase2_v3
from .phase3_blaze_v3 import run_phase3_v3

__version__ = "3.0.0"
__all__ = [
    "run_phase0_library_v3",
    "run_phase1_v3", 
    "run_phase2_v3",
    "run_phase3_v3",
]
