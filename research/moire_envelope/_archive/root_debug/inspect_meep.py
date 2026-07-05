import meep as mp
import inspect

print("Meep Version:", mp.__version__)
print("\nSimulation.__init__ args:")
sig = inspect.signature(mp.Simulation.__init__)
for name in sig.parameters:
    print(name)

print("\nSimulation docstring:")
print(mp.Simulation.__init__.__doc__)
