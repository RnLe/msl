#!/usr/bin/env python3

import subprocess
import sys
import os

# Quick test to see debug output from Rust
os.chdir('/home/renlephy/msl')

# Run a simple test that will trigger the debug output
test_cmd = ['cargo', 'test', '--package', 'moire-lattice', 'unified_api_uses_stored_twist_angle', '--', '--nocapture']

result = subprocess.run(test_cmd, capture_output=True, text=True)
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
