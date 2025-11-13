# Makefile for MSL Typst documentation
# Compiles theory notes and quick reference using Typst CLI

TYPST = typst
ROOT  = .
TARGET = target

# Output PDFs
MSL_NOTES = $(TARGET)/msl-notes.pdf
QUICK_REF = $(TARGET)/quick-ref.pdf

# Source files
MSL_NOTES_SRC = docs/msl-notes/main.typ
QUICK_REF_SRC = docs/quick-ref/main.typ

.PHONY: all msl-notes quick-ref clean watch-notes watch-ref help

# Default target: build all documents
all: msl-notes quick-ref

# Compile main theory notes
msl-notes: $(MSL_NOTES)

$(MSL_NOTES): $(MSL_NOTES_SRC) shared/*.typ docs/msl-notes/chapters/*.typ assets/exampleBib.bib
	@echo "Compiling MSL theory notes..."
	$(TYPST) compile --root $(ROOT) $(MSL_NOTES_SRC) $(MSL_NOTES)
	@echo "✓ MSL notes compiled to $(MSL_NOTES)"

# Compile quick reference
quick-ref: $(QUICK_REF)

$(QUICK_REF): $(QUICK_REF_SRC) shared/*.typ assets/exampleBib.bib
	@echo "Compiling quick reference..."
	$(TYPST) compile --root $(ROOT) $(QUICK_REF_SRC) $(QUICK_REF)
	@echo "✓ Quick reference compiled to $(QUICK_REF)"

# Watch mode for continuous compilation during editing
watch-notes:
	@echo "Watching MSL notes for changes (Ctrl+C to stop)..."
	$(TYPST) watch --root $(ROOT) $(MSL_NOTES_SRC) $(MSL_NOTES)

watch-ref:
	@echo "Watching quick reference for changes (Ctrl+C to stop)..."
	$(TYPST) watch --root $(ROOT) $(QUICK_REF_SRC) $(QUICK_REF)

# Clean compiled PDFs
clean:
	@echo "Cleaning compiled PDFs..."
	rm -f $(TARGET)/*.pdf
	@echo "✓ Cleaned target directory"

# Display help
help:
	@echo "MSL Typst Documentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Compile all documents (default)"
	@echo "  msl-notes   - Compile main theory notes"
	@echo "  quick-ref   - Compile quick reference"
	@echo "  watch-notes - Watch and auto-recompile theory notes"
	@echo "  watch-ref   - Watch and auto-recompile quick reference"
	@echo "  clean       - Remove compiled PDFs"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Compile everything"
	@echo "  make msl-notes    # Compile just the main notes"
	@echo "  make watch-notes  # Auto-recompile on changes"
	@echo "  make clean        # Remove PDFs"
