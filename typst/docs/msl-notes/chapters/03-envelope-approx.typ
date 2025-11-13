// docs/msl-notes/chapters/03-envelope-approx.typ
// Envelope approximation framework.

#import "/shared/math-msl.typ": *

= Envelope approximation <sec:envelope>

The envelope approximation (EA) is a systematic method to derive
effective low-energy models for moiré systems by projecting the
full Hamiltonian onto slowly-varying envelope functions.

== Physical picture

In a moiré system with length scale $ L_m $, the physics separates into:
- *Fast* oscillations at the monolayer lattice scale $ a $
- *Slow* modulations at the moiré scale $ L_m approx a / theta $

The EA systematically captures the slow physics while averaging
over fast oscillations.

== Mathematical framework

Starting from the full tight-binding Hamiltonian $ H_"TB" $, we:

1. Expand wavefunctions as Bloch-modulated envelopes:
  $ psi(bold(r)) = sum_n phi_n(bold(r)) u_(bold(k)_n)(bold(r)) $

2. Project onto a low-energy subspace (e.g., states near Dirac points)

3. Derive an effective Hamiltonian #op($H_"eff"$) for the envelopes

== Implementation in MSL

The MSL framework implements EA through several stages:

- *Lattice setup*: Define monolayer and moiré lattices
- *Local Bloch functions*: Compute monolayer eigenstates at each moiré site
- *EA operator construction*: Build effective coupling matrices
- *Envelope solver*: Solve the effective eigenvalue problem

For details on the numerical pipeline, see the `research/envelope_approximation/`
directory in the MSL repository.

== Validity regime

The EA is valid when:
- $ theta $ is small ($ theta lt.double 10 degree $)
- Interlayer coupling $ t_perp $ is moderate
- Energy scales separate: $ epsilon_"moiré" lt.double epsilon_"monolayer" $

These conditions ensure clean separation between fast and slow degrees of freedom.
