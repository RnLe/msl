
import meep as mp
import h5py
import numpy as np
import os
import glob

# Ensure clean slate
for f in glob.glob("*.h5"):
    if "check_meep" in f:
        os.remove(f)

cell = mp.Vector3(10,10,0)
geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf,mp.inf), material=mp.Medium(index=1.5))]
sim = mp.Simulation(cell_size=cell, resolution=10, geometry=geometry)
sim.init_sim()

# Use standard output function
# note: filename prefix will be 'check_meep'
sim.run(mp.at_beginning(mp.output_efield), until=0)

# Check files
files = glob.glob("check_meep-e-*.h5")
print("Found files:", files)

for fn in files:
    with h5py.File(fn, 'r') as hf:
        print(f"File: {fn}")
        print("Keys:", list(hf.keys()))
