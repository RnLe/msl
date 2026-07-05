import meep as mp
import inspect

print("Meep Version:", mp.__version__)
print("Symmetries in meep:")
for name in dir(mp):
    if "Symmetry" in name or "Rotate" in name or "Mirror" in name:
        print(name)
