import meep as mp
import inspect
import sys

# Get the source file of Simulation
src_file = inspect.getfile(mp.Simulation)
print(f"Source file: {src_file}")

with open(src_file, 'r') as f:
    lines = f.readlines()
    
# Search for _create_grid_volume
for i, line in enumerate(lines):
    if "def _create_grid_volume" in line:
        print(f"Found _create_grid_volume at {i+1}")
        start = i
        end = i + 20
        for j in range(start, end):
            print(f"{j+1}: {lines[j].rstrip()}")
        break
