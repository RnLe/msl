# Bloch Phases of Neighboring Registries

Crucial: The bloch phase of u_n of neighboring registries in the moire cell have essentially random phases!

The envelope F_n(R) from phase 3 is computed with a Hamiltonian that uses overlaps which absorb the phase information.

[Phase4]   Loaded Bloch fields: shape (64, 64, 10, 64, 64, 3)
Band indices in all_bands array: [0, 1, 2]
sub_bands: [0 1 2], all_bands: [0 1 2 3 4 5 6 7 8 9]
Dominant band (subspace band 1) -> all_band index 1

Phase alignment between neighboring registry points:
  center-right overlap: |<u_c|u_r>| = 0.707582, phase = 29.98 deg
  center-up     overlap: |<u_c|u_u>| = 0.994106, phase = 1.60 deg

Phase jumps across ALL neighboring registry pairs:
  s1-direction: mean=90.55 deg, max=180.00 deg
                std=104.90 deg
  s2-direction: mean=68.55 deg, max=180.00 deg
                std=102.11 deg

Phase jump histogram (degrees):
    0- 20: s1=  465, s2= 1988
   20- 40: s1=  474, s2=  169
   40- 60: s1=  436, s2=  154
   60- 80: s1=  412, s2=  134
   80-100: s1=  413, s2=  131
  100-120: s1=  407, s2=  147
  120-140: s1=  472, s2=  135
  140-160: s1=  470, s2=  134
  160-180: s1=  483, s2= 1040
