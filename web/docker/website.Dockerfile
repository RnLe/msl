# 1) Dev stage
FROM node:22-alpine AS developer

# NOTE: Rust is installed into the dev stage to allow for builds
# Effectively, a separate slim rust image could be used as service that runs parallel, which minimizes the size and complexity of the dev image
# But this needs careful wiring. For now, this is the simplest solution
# Prepare rust installation
RUN apk add --no-cache \
    build-base \
    curl \
    git \
    openssl-dev \
    musl-dev \
    llvm \
    clang \
    cmake

# install rustup (non-interactive), set up PATH
ENV CARGO_HOME=/root/.cargo
ENV PATH="${CARGO_HOME}/bin:/usr/local/bin:${PATH}"
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path \
    && rustup default stable \
    && rustup target add wasm32-unknown-unknown \
    && cargo install wasm-pack

WORKDIR /repo
ENV NODE_ENV=development

# Copy the entire repo to the container and remove the .gitignore file
# This mimics the behavior of gh pages and prevents setup mismatches between development and production
COPY . .
RUN rm .gitignore

RUN corepack enable \
    && pnpm install
# && pnpm install --frozen-lockfile         (add this back when modules are stable)

# For now, no wasm build
# # build the WASM bundle into public/wasm
# WORKDIR /repo
# # wasm-pack emits into public/wasm
# RUN wasm-pack build \
#       --target bundler \
#       --out-dir public/wasm \
#       --dev

# switch back to website root for dev server
WORKDIR /repo

EXPOSE 3000

# Note: The container internally runs on port 3000.
# Port mapping in docker-compose.yml determines the host port.

# 2) Build stage (Node only)
FROM node:22-alpine AS builder
WORKDIR /repo
COPY --from=developer /repo ./

# Build the website
RUN corepack enable && pnpm run build