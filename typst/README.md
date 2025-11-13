# MSL Typst Documentation

Clean, multi-document Typst setup for the **MSL (Moiré Superlattice)** Rust framework—a mini-knowledge-base for moiré lattice research theory and implementation notes.

## Overview

This `typst/` folder serves as a standalone documentation project containing:

- **Multiple documents** (theory notes, quick references, etc.)
- **Shared layout & math macros** for consistent MSL-specific notation
- **Unified bibliography** (`assets/exampleBib.bib`) used across all documents
- Integration with **Typst CLI** and **VSCode (Tinymist extension)**

## Directory Structure

```
typst/
├── assets/
│   ├── exampleBib.bib       # Shared bibliography (BibTeX)
│   └── figures/             # Images for all documents
├── shared/
│   ├── layout.typ           # Global layout template (msl-article)
│   └── math-msl.typ         # MSL-specific math macros
├── docs/
│   ├── msl-notes/           # Main longform theory document
│   │   ├── main.typ
│   │   └── chapters/
│   │       ├── 01-intro.typ
│   │       ├── 02-bistritzer-bm.typ
│   │       └── 03-envelope-approx.typ
│   └── quick-ref/           # Quick reference cheat sheet
│       └── main.typ
├── target/                  # Compiled PDFs (auto-generated)
├── Makefile                 # Build automation
└── README.md                # This file
```

## Quick Start

### Prerequisites

- **Typst CLI** installed ([install guide](https://github.com/typst/typst))
- **WSL/Ubuntu** (or any Linux/macOS environment)
- **VSCode + Tinymist extension** (optional, for live preview)

### Compilation

Navigate to the `typst/` directory:

```bash
cd typst
```

**Build all documents:**
```bash
make
```

**Build specific documents:**
```bash
make msl-notes    # Main theory notes
make quick-ref    # Quick reference
```

**Watch mode** (auto-recompile on save):
```bash
make watch-notes  # Watch theory notes
make watch-ref    # Watch quick reference
```

**Clean compiled PDFs:**
```bash
make clean
```

### Manual Compilation

If you prefer direct `typst` commands:

```bash
# From typst/ directory
typst compile --root . docs/msl-notes/main.typ target/msl-notes.pdf
typst compile --root . docs/quick-ref/main.typ target/quick-ref.pdf
```

The `--root .` flag ensures absolute Typst paths (like `"/shared/layout.typ"`) resolve correctly from any subfolder.

## VSCode Integration (Recommended)

For the best editing experience:

1. **Open repo in WSL:**
   ```bash
   cd /path/to/msl
   code .
   ```

2. **Install Tinymist extension:**
   - Extension ID: `myriad-dreamin.tinymist`
   - Provides live preview, diagnostics, auto-completion, and jump-to-definition

3. **Open and preview:**
   - Open `typst/docs/msl-notes/main.typ`
   - Click the Tinymist preview icon or run command: **"Typst Preview: Preview current file"**
   - Edit any file (`main.typ` or `chapters/*.typ`)—preview updates in real-time

Tinymist automatically handles imports and bibliography resolution when the main file is active.

## Key Features

### Shared Layout (`shared/layout.typ`)

All documents use the `msl-article` template for consistent styling:

```typst
#import "/shared/layout.typ": msl-article-default
#import "/shared/math-msl.typ": *

#show: msl-article-default.with(
  title: "Document Title",
  subtitle: "Optional Subtitle",
  authors: ("Rene",),
  date: datetime.today().display(),
)
```

Features:
- A4 paper, readable margins
- Numbered headings
- Clean title block with author/date

### Math Macros (`shared/math-msl.typ`)

Common MSL/condensed-matter notation:

| Macro | Output | Description |
|-------|--------|-------------|
| `#RR`, `#CC`, `#ZZ` | **R**, **C**, **Z** | Number sets |
| `#v(vec)` | **vec** | Bold vectors |
| `#ket(psi)` | \|ψ⟩ | Ket notation |
| `#bra(psi)` | ⟨ψ\| | Bra notation |
| `#braket(a, b)` | ⟨a\|b⟩ | Inner product |
| `#op(H)` | Ĥ | Operator (hat) |
| `#dby(f, x)` | ∂f/∂x | Partial derivative |
| `#moire-params(θ, a, t)` | Moiré parameters: θ = ..., a = ..., t = ... | Parameter display |

Extend this file with additional physics/math macros as needed.

### Bibliography Management

All documents share `assets/exampleBib.bib`. To cite:

```typst
// In your document
@citation-key

// At the end of main.typ
#bibliography("/assets/exampleBib.bib")
```

Typst automatically formats citations and generates the bibliography.

## Adding New Documents

To create a new document (e.g., "Envelope Approximation Details"):

1. **Create folder:**
   ```bash
   mkdir -p docs/envelope-notes
   ```

2. **Create `main.typ`:**
   ```typst
   // docs/envelope-notes/main.typ
   #import "/shared/layout.typ": msl-article-default
   #import "/shared/math-msl.typ": *

   #show: msl-article-default.with(
     title: "Envelope Approximation Notes",
     authors: ("Rene",),
   )

   = Introduction
   Content goes here...

   #bibliography("/assets/exampleBib.bib")
   ```

3. **Add to Makefile** (optional):
   ```makefile
   ENVELOPE_NOTES = $(TARGET)/envelope-notes.pdf
   ENVELOPE_NOTES_SRC = docs/envelope-notes/main.typ

   envelope-notes: $(ENVELOPE_NOTES)

   $(ENVELOPE_NOTES): $(ENVELOPE_NOTES_SRC) shared/*.typ
       $(TYPST) compile --root $(ROOT) $(ENVELOPE_NOTES_SRC) $(ENVELOPE_NOTES)
   ```

4. **Compile:**
   ```bash
   make envelope-notes
   # or
   typst compile --root . docs/envelope-notes/main.typ target/envelope-notes.pdf
   ```

## Path Resolution

This setup uses **absolute Typst paths** (starting with `/`) to ensure imports work from any subfolder:

- `/shared/layout.typ` → always resolves to `typst/shared/layout.typ`
- `/assets/exampleBib.bib` → always resolves to `typst/assets/exampleBib.bib`

This requires compiling with `--root .` from the `typst/` directory, which the Makefile handles automatically.

## Best Practices

1. **Keep shared code in `shared/`**: Layout, macros, and reusable definitions
2. **One `main.typ` per document**: Each document is self-contained with its own `main.typ`
3. **Use `#include` for chapters**: Break long documents into chapter files
4. **Assets in `assets/`**: Bibliography, figures, data files
5. **Compile to `target/`**: Keep compiled PDFs separate from source
6. **Use watch mode while editing**: `make watch-notes` for instant feedback

## Troubleshooting

**Import errors (file not found):**
- Ensure you're compiling with `--root .` from `typst/`
- Check that paths start with `/` (e.g., `"/shared/layout.typ"`)

**Bibliography not rendering:**
- Verify `exampleBib.bib` exists in `assets/`
- Ensure you have citations (e.g., `@key`) in your document before `#bibliography()`

**VSCode preview issues:**
- Make sure Tinymist extension is installed in WSL (not Windows)
- Open the main file (`main.typ`), not chapter files
- Check Typst version: `typst --version` (should be ≥ 0.11.0)

## Resources

- **Typst Documentation**: https://typst.app/docs/
- **Typst CLI**: https://github.com/typst/typst
- **Tinymist Extension**: https://github.com/Myriad-Dreamin/tinymist
- **Typst Universe** (packages): https://typst.app/universe/

## License

This documentation setup is part of the MSL framework project. See repository root for license information.

---

**Happy typesetting!** For questions or improvements, reach out via the main MSL repository issues.
