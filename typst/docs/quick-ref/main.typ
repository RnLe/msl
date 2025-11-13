// docs/quick-ref/main.typ
// Short cheat sheet / quick reference.

#import "/shared/layout.typ": msl-article-default
#import "/shared/math-msl.typ": *

#show: msl-article-default.with(
  title: "MSL Quick Reference",
  subtitle: "Key formulas and definitions",
  authors: ("Rene",),
  date: none,
)

= Key scales

- Moiré length scale $ L_m approx a / theta $
- Twist angle $ theta $ (typically $ 0.5 degree $ to $ 10 degree $)
- Interlayer coupling parameters $ t_perp $, $ t_parallel $
- Energy scales: $ epsilon_"Dirac" approx v_F |bold(k)| $, $ epsilon_"moiré" approx v_F / L_m $

= Core objects

- Real-space coordinate: $ r in #RR^2 $
- Momentum-space coordinate: $ k in #RR^2 $
- State vectors: #ket($psi$), #bra($psi$)
- Operators: #op($H$), #op($T$)
- Inner products: #braket($phi$, $psi$)

= Lattice vectors

For triangular monolayer lattice:
$ bold(a)_1 = a vec(1, 0), quad bold(a)_2 = a vec(1/2, sqrt(3)/2) $

Reciprocal lattice:
$ bold(b)_1 = (2pi) / a vec(1, -1/sqrt(3)), quad bold(b)_2 = (2pi) / a vec(0, 2/sqrt(3)) $

Moiré reciprocal lattice vectors (small angle approximation):
$ bold(G)_m approx theta |bold(G)_"mono"| $

= Common Hamiltonians

== Single-layer Dirac Hamiltonian
$ H_0(bold(k)) = v_F bold(sigma) dot.c bold(k) $

where $ bold(sigma) = (sigma_x, sigma_y) $ are Pauli matrices for sublattice.

== Interlayer coupling (BM model)
$ T(bold(r)) = sum_(j=0)^2 T_j e^(i bold(q)_j dot.c bold(r)) $

with $ bold(q)_j $ the moiré reciprocal vectors.

= MSL Framework components

== Rust core modules
- `lattice`: Bravais lattice construction and operations
- `reciprocal`: Reciprocal space calculations
- `symmetry`: Point group and space group operations
- `bloch`: Bloch eigensolvers and band structures

== Python bindings
Import via: `from msl import Lattice, ReciprocalLattice, ...`

== Key functions
- `Lattice::new(a1, a2)`: Create 2D Bravais lattice
- `generate_moire_lattice(theta)`: Construct moiré superlattice
- `compute_band_structure(kpath)`: Calculate energy bands

= Numerical considerations

- *k-point sampling*: Use $ N_k approx 100 $ points per moiré BZ edge for convergence
- *Truncation*: Include $ approx 3 $ shells of moiré reciprocal vectors
- *Energy cutoffs*: Typically $ E_"cut" approx 10 times t_perp $

= References

For detailed theory, see the main MSL theory notes and relevant literature
in the shared bibliography.

#bibliography("/assets/exampleBib.bib")
