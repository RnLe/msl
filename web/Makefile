# Makefile for cleaning up the repo and set it to a fresh-clone state
# Run in root of the repo with sudo:

# sudo make clean

# Run 'pnpm i' to install dependencies again
# Run 'pnpm build' to build the project again

# Folders to blow away
CLEAN_DIRS := \
  node_modules \
  .turbo \
  out \
  **/**/node_modules \
  **/**/.next \
  **/**/out \
  **/**/dist \
  **/**/.turbo \
  **/**/pkg \
  **/**/wasm/target \
  apps/website/wasm

# Files to blow away
CLEAN_FILES := \
  **/**/*.tsbuildinfo \

.PHONY: clean
clean:
	@echo "🧹 Cleaning all build artifacts and installs…"
	# Remove directories
	@for d in $(CLEAN_DIRS); do \
	  sudo rm -rf $$d; \
	done
	# Remove files
	@for f in $(CLEAN_FILES); do \
	  sudo rm -f $$f; \
	done
	@echo "✅  Done. Repo is now back to a fresh-clone state."

# Make your life easier connecting to the container

# Run the container
run:
	docker compose up msl_website_dev
# Run and build the container
runb:
	docker compose up msl_website_dev --build
# Stop the container and remove the volumes
down:
	docker compose down -v

# Drop into a shell in the dev container
shell:
	docker compose exec msl_website_dev sh

# Serve the website’s out/ folder on 3001
#
#    Assumes both:
#      • a package.json script like "serve:out": "serve out -l 3001"
#      • AND that 'serve' is in the dependencies of the package.json
serve:
	pnpm --filter msl-website run serve:out

# Completely clean the repo, install dependencies, build, and serve (reproducing the gh actions workflow - on the host)
dev: clean
	pnpm i && pnpm run build && pnpm --filter msl-website run serve:out

# Create a production build of the website (in the container) and serve it (on the host). This is exactly what the gh actions workflow does.
prod: clean down
	docker compose up msl_website_prod --build && \
	npx serve out -l 3001

# WASM Build Pipeline
# ===================

# Build WASM module and copy to website
.PHONY: build-wasm
build-wasm:
	@echo "🦀 Building WASM module..."
	cd ../rust-wasm && wasm-pack build --target web --out-dir pkg
	@echo "📦 Copying WASM files to website..."
	mkdir -p apps/website/wasm
	cp ../rust-wasm/pkg/moire_lattice_wasm.js apps/website/wasm/wasm.js
	cp ../rust-wasm/pkg/moire_lattice_wasm.d.ts apps/website/wasm/wasm.d.ts
	cp ../rust-wasm/pkg/moire_lattice_wasm_bg.wasm apps/website/wasm/wasm_bg.wasm
	cp ../rust-wasm/pkg/moire_lattice_wasm_bg.wasm.d.ts apps/website/wasm/wasm_bg.wasm.d.ts
	@echo "🧹 Cleaning up WASM build artifacts..."
	cd ../rust-wasm && rm -rf target
	@echo "✅ WASM module built and copied successfully!"

# Build WASM and install dependencies
.PHONY: build-wasm-dev
build-wasm-dev: build-wasm
	@echo "📦 Installing dependencies..."
	pnpm install
	@echo "✅ WASM build and dependencies ready for development!"

# Clean WASM artifacts from website
.PHONY: clean-wasm
clean-wasm:
	@echo "🧹 Cleaning WASM artifacts from website..."
	rm -rf apps/website/wasm/*
	@echo "✅ WASM artifacts cleaned!"

# Full development build including WASM
dev-with-wasm: clean build-wasm-dev
	pnpm run build && pnpm --filter msl-website run serve:out