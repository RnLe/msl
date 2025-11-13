// docs/msl-notes/chapters/02-bistritzer-bm.typ
// Bistritzer-MacDonald continuum model.

#import "/shared/math-msl.typ": *

= Bistritzer–MacDonald model <sec:bm-model>

The Bistritzer–MacDonald (BM) model describes twisted bilayer graphene
in the continuum limit. The Hamiltonian acts on a four-component spinor
representing layer (top/bottom) and sublattice (A/B) degrees of freedom.

== Model structure

The model consists of:
- Intralayer Dirac cones for each layer
- Interlayer tunneling terms with moiré modulation
- Valley degree of freedom (K, K')

== Key parameters

Using our shared macros:

#moire-params($theta$, $a$, $t_perp$)

Where:
- $ theta $: twist angle (typically $ approx 1 degree $ for magic angle)
- $ a $: graphene lattice constant ($ approx 2.46 space Å $)
- $ t_perp $: interlayer hopping amplitude

== Hamiltonian structure

The continuum Hamiltonian in momentum space:

$
  H(bold(k)) = mat(
    H_0(bold(k)), T(bold(k));
    T^dagger(bold(k)), H_0(R_theta bold(k))
  )
$

where $ H_0 $ is the single-layer Dirac Hamiltonian and $ T $ encodes
the moiré-modulated interlayer coupling.

== Envelope approximation connection

This model connects to the envelope approximation framework discussed
in @sec:envelope, where we project onto low-energy subspaces.
