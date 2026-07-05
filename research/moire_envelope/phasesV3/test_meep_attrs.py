import meep as mp
sim = mp.Simulation(cell_size=mp.Vector3(1,1), resolution=10)
print("Has set_array?", hasattr(sim, 'set_array'))
print("Dir sim:", dir(sim))
