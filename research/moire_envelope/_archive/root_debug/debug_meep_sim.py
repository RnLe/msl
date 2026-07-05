import meep as mp
import inspect
print(f"Meep version: {mp.__version__}")
print(inspect.signature(mp.Simulation.__init__))
