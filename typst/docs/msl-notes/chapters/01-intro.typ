// docs/msl-notes/chapters/01-intro.typ
// Conceptual introduction.

#import "/shared/math-msl.typ": *

= Motivation and context <sec:intro>

The MSL framework targets problems where a twisted bilayer
(or multilayer) structure leads to emergent long-wavelength
degrees of freedom. The effective description often lives
on a moiré superlattice in real or reciprocal space.

A prototypical example is the continuum model for twisted
bilayer graphene (cite relevant papers from exampleBib.bib as needed).

We often work in a Hilbert space of layer, sublattice, and
valley degrees of freedom and use operators like $ H $
acting on k-dependent spinors.

Example usage of the shared math helpers:

- Real-space coordinate $ r in #RR^2 $
- Bloch wavevector $ k in #RR^2 $
- State vector #ket($psi$)
- Overlap #braket($phi$, $psi$)

The actual BM model details are in @sec:bm-model.
