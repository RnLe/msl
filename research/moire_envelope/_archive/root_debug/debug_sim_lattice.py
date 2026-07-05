import meep as mp
sim = mp.Simulation(cell_size=mp.Vector3(1,1,0), resolution=10)
print("Keys in sim:")
for k in dir(sim):
    if "lat" in k.lower():
        print(f"  {k}")
