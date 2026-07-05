"""
Master runner for all Blaze2D vs MPB validation steps.
Run: python run_all.py [step1|step2|step3|step4|all]
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'

import sys
import time


def run_step(name, module_name):
    print(f"\n{'#' * 80}")
    print(f"# Running: {name}")
    print(f"{'#' * 80}\n")
    t0 = time.time()
    mod = __import__(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n>>> {name} completed in {elapsed:.1f}s\n")


def main():
    steps = {
        "step1": ("Step 1: Band Diagrams", "validate_bands"),
        "step2": ("Step 2: K-stencil / Velocity / Mass Tensor", "validate_stencil"),
        "step3": ("Step 3: Eigenfunctions", "validate_eigenfunctions"),
        "step4": ("Step 4: Derived Quantities", "validate_derived"),
    }

    requested = sys.argv[1] if len(sys.argv) > 1 else "all"

    if requested == "all":
        for key in steps:
            run_step(steps[key][0], steps[key][1])
    elif requested in steps:
        run_step(steps[requested][0], steps[requested][1])
    else:
        print(f"Unknown step: {requested}")
        print(f"Usage: python run_all.py [{' | '.join(list(steps.keys()) + ['all'])}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
