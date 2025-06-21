# Moire Lattice Suite (MSL)

A high-performance monorepo for lattice and moire lattice calculations, providing Rust, Python, and WebAssembly interfaces with comprehensive documentation and interactive tools.

## 🚀 Overview

This project combines three main components:

1. **High-Performance Rust Core** (`rust-core/`): Comprehensive lattice algorithms and data structures
2. **Python Bindings** (`rust-python/`): PyO3-based Python package for rapid prototyping and analysis
3. **WebAssembly Interface** (`rust-wasm/`): Browser-compatible high-performance calculations
4. **Web Application** (`web/`): Next.js-based documentation, theory, and interactive tools

## 📁 Project Structure

```
msl/
├── Cargo.toml             # Workspace configuration
├── rust-core/             # Pure Rust lattice library
├── rust-python/           # Python bindings (PyO3 + Maturin)
├── rust-wasm/             # WebAssembly bindings
├── python-example/        # Python usage examples
├── web/                   # Next.js web application
└── docs/                  # Additional documentation
```

## 🛠️ Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [Python](https://www.python.org/) 3.8+
- [Node.js](https://nodejs.org/) 16+ (for web components)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) (for WASM)

### Building the Rust Core

```bash
# Build and test the core library
cd rust-core
cargo build --release
cargo test
```

### Python Bindings

```bash
# Install and build Python bindings
cd rust-python
pip install maturin
maturin develop

# Run Python examples
cd ../python-example
python example.py
```

### WebAssembly

```bash
# Build WASM package
cd rust-wasm
wasm-pack build --target web --out-dir pkg
```

### Web Application

```bash
# Set up web application (copy your existing Next.js app here)
cd web
# npm install && npm run dev
```

## 🎯 Features

### Core Library (`rust-core`)
- Multiple Bravais lattice types (square, rectangular, hexagonal, oblique)
- High-performance point generation and lattice operations
- Reciprocal space calculations
- Coordination number analysis
- Voronoi cell construction
- Symmetry operations and high-symmetry points

### Python Interface (`rust-python`)
- Pythonic API for all core functionality
- Seamless numpy integration
- Visualization helpers
- Jupyter notebook support
- Performance benchmarking tools

### WebAssembly Interface (`rust-wasm`)
- Browser-compatible lattice calculations
- SVG generation for visualization
- TypeScript definitions
- React component integration
- Real-time interactive applications

### Web Application (`web`)
- Comprehensive API documentation
- Physics and mathematics theory articles
- Interactive lattice visualization tools
- Educational demonstrations
- Performance comparisons

## 📖 Documentation

- **API Documentation**: Available at [GitHub Pages](https://rnle.github.io/msl) (when deployed)
- **Theory Articles**: Physics and mathematics background
- **Examples**: Practical usage examples for all interfaces
- **Performance**: Benchmarks and optimization guides

## 🔬 Use Cases

### Research Applications
- Condensed matter physics simulations
- Materials science calculations
- Crystallography analysis
- Electronic structure studies

### Educational Tools
- Interactive lattice visualization
- Physics concept demonstrations
- Mathematical lattice theory
- Computational physics examples

### Development
- High-performance algorithm prototyping
- Cross-platform lattice libraries
- Web-based scientific applications

## 🚀 Performance

The Rust core provides exceptional performance:
- **Memory efficiency**: Optimized data structures
- **Parallel processing**: Rayon-based parallelization
- **SIMD optimization**: Vectorized calculations where applicable
- **Zero-cost abstractions**: High-level API with minimal overhead

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/rnle/msl.git
cd msl

# Build all components
cargo build --release
cd rust-python && maturin develop && cd ..
cd rust-wasm && wasm-pack build --target web && cd ..
```

## 📝 License

This project is licensed under [MIT License](LICENSE-MIT)

## 🙏 Acknowledgments

- Built with [Rust](https://www.rust-lang.org/)
- Python bindings via [PyO3](https://pyo3.rs/)
- WebAssembly via [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)
- Web framework: [Next.js](https://nextjs.org/)
- Documentation: [Nextra](https://nextra.site/)

## 📬 Contact

Rene-Marcel Lehner - rene.lehner@tu-dortmund.de

Project Link: [https://github.com/rnle/msl](https://github.com/rnle/msl)

---

*This project is part of ongoing research in condensed matter physics and computational materials science.*
