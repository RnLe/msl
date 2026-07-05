
import meep as mp
import h5py
import numpy as np
import os
import glob

# Use single process
cell = mp.Vector3(10,10,0)
geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf,mp.inf), material=mp.Medium(index=1.0))]
sources = [mp.Source(mp.GaussianSource(1.0), component=mp.Ex, center=mp.Vector3(0,0))]
sim = mp.Simulation(cell_size=cell, resolution=5, geometry=geometry, sources=sources)
sim.init_sim()
# Run a bit to populate fields
sim.run(until=1.0)

fn = "ref_fields"
for f in glob.glob(fn + "*.h5"):
    os.remove(f)

# dump fields to h5
sim.dump_fields(fn)

files = glob.glob(fn + "*.h5")
print("Found:", files)
if files:
    with h5py.File(files[0], 'r') as hf:
        print("Keys:", list(hf.keys()))
        for k in hf.keys():
            print(f"  {k}: {hf[k].shape}")
            
