# Moire Lattice WASM Bindings

WebAssembly bindings for the `moire-lattice` Rust library, enabling high-performance lattice calculations in web browsers and Node.js environments.

## Features

- **High Performance**: Rust-powered calculations compiled to WebAssembly
- **Web Compatible**: Works in all modern browsers and Node.js
- **Multiple Lattice Types**: Support for square, rectangular, hexagonal, and oblique lattices
- **Interactive Visualization**: Generate SVG representations and raw point data
- **TypeScript Support**: Includes TypeScript definitions for better development experience

## Installation

### Using npm (when published)
```bash
npm install moire-lattice-wasm
```

### Building from Source

1. Install [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
2. Build the package:
   ```bash
   wasm-pack build --target web --out-dir pkg
   ```

## Usage

### In a Web Browser

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import init, { WasmLattice2D, create_square_lattice } from './pkg/moire_lattice_wasm.js';
        
        async function run() {
            // Initialize the WASM module
            await init();
            
            // Create a square lattice
            const lattice = create_square_lattice(1.0);
            
            // Generate points within radius 5
            const points = lattice.generate_points(5.0, 0.0, 0.0);
            console.log(`Generated ${points.length} lattice points`);
            
            // Get lattice properties
            console.log('Unit cell area:', lattice.unit_cell_area());
            console.log('Parameters:', lattice.get_parameters());
            
            // Generate SVG visualization
            const svg = lattice.to_svg(400, 400, 5.0);
            document.getElementById('visualization').innerHTML = svg;
        }
        
        run();
    </script>
</head>
<body>
    <div id="visualization"></div>
</body>
</html>
```

### Using Custom Parameters

```javascript
import init, { WasmLattice2D } from './pkg/moire_lattice_wasm.js';

await init();

// Create a custom lattice
const lattice = new WasmLattice2D({
    lattice_type: "hexagonal",
    a: 1.0,
    b: null,  // Will default to a
    angle: null  // Will default to appropriate angle for lattice type
});

// Generate points
const points = lattice.generate_points(3.0, 0.0, 0.0);

// Get lattice vectors
const vectors = lattice.lattice_vectors();
console.log('Lattice vectors:', vectors);

// Get reciprocal vectors
const reciprocal = lattice.reciprocal_vectors();
console.log('Reciprocal vectors:', reciprocal);
```

### With Bundlers (Webpack, Vite, etc.)

```javascript
import init, { WasmLattice2D, create_hexagonal_lattice } from 'moire-lattice-wasm';

async function setupLattice() {
    await init();
    
    const lattice = create_hexagonal_lattice(1.0);
    return lattice;
}
```

## API Reference

### Classes

#### `WasmLattice2D`

**Constructor:**
- `new WasmLattice2D(params)`: Create lattice from parameters object
  - `params.lattice_type`: "square", "rectangular", "hexagonal", or "oblique"
  - `params.a`: First lattice parameter
  - `params.b`: Second lattice parameter (optional)
  - `params.angle`: Lattice angle in degrees (optional)

**Methods:**
- `generate_points(radius, center_x, center_y)`: Generate lattice points within radius
- `get_parameters()`: Get lattice parameters as object
- `unit_cell_area()`: Calculate unit cell area
- `lattice_vectors()`: Get lattice vectors as object
- `reciprocal_vectors()`: Get reciprocal lattice vectors
- `to_svg(width, height, radius)`: Generate SVG representation

### Utility Functions

- `create_square_lattice(a)`: Create square lattice with parameter `a`
- `create_hexagonal_lattice(a)`: Create hexagonal lattice with parameter `a`
- `create_rectangular_lattice(a, b)`: Create rectangular lattice with parameters `a`, `b`
- `version()`: Get library version

## Development

### Building

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web --out-dir pkg

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg-node

# Build for bundlers
wasm-pack build --target bundler --out-dir pkg-bundler
```

### Testing

```bash
# Test in browser
wasm-pack test --headless --firefox

# Test in Node.js
wasm-pack test --node
```

## Performance

The WASM module provides near-native performance for lattice calculations while maintaining JavaScript's ease of use. For intensive calculations involving thousands of lattice points, expect significant performance improvements over pure JavaScript implementations.

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support  
- Safari: Full support (iOS 11.3+)
- Internet Explorer: Not supported

## License

This project is licensed under: MIT License