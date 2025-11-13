// docs/msl-notes/main.typ
// Main longform theory backup for the MSL framework.

#import "/shared/layout.typ": msl-article-default
#import "/shared/math-msl.typ": *

#show: msl-article-default.with(
  title: "Moiré Superlattice Theory Notes",
  subtitle: "Theory backup for the MSL Rust framework",
  authors: ("Rene",),
  date: datetime.today().display(),
)

// From here on, normal Typst markup.

// Top-level intro section
= Overview <sec:overview>

These notes collect the theoretical background for the
_moiré superlattice (MSL)_ framework implemented in Rust.

- Geometry and lattice construction
- Continuum models (e.g. Bistritzer–MacDonald)
- Envelope approximations and effective models
- Numerical aspects and scaling limits

For quick jumps, see @sec:bm-model and @sec:envelope.

#pagebreak()

// Split content into chapter files in ./chapters.
#include "chapters/01-intro.typ"
#include "chapters/02-bistritzer-bm.typ"
#include "chapters/03-envelope-approx.typ"

// Bibliography heading
= References

// Typst will render the full bibliography from exampleBib.bib
#bibliography("/assets/exampleBib.bib")
