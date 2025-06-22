'use client';

import { useEffect, useState } from 'react';
import { initWasm } from '../providers/wasmLoader';

interface WignerSeitzCell2DProps {
  latticeType: 'oblique' | 'rectangular' | 'centered_rectangular' | 'square' | 'hexagonal';
  width?: number;
  height?: number;
  showNeighbors?: boolean;
  showConstruction?: boolean;
}

export function WignerSeitzCell2D({ 
  latticeType, 
  width = 400, 
  height = 400,
  showNeighbors = true,
  showConstruction = false
}: WignerSeitzCell2DProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initWasm()
      .then(() => setIsLoaded(true))
      .catch((err: any) => setError(err.message));
  }, []);

  if (error) {
    return <div className="error">Error loading WASM: {error}</div>;
  }

  if (!isLoaded) {
    return <div className="loading">Loading Wigner-Seitz cell visualization...</div>;
  }

  return (
    <div className="wigner-seitz-cell-2d">
      <div className="placeholder" style={{ width, height, border: '2px dashed #ccc', padding: '20px' }}>
        <h4>Wigner-Seitz Cell - {latticeType}</h4>
        <p>Interactive Wigner-Seitz cell (Voronoi cell) visualization will be implemented here.</p>
        <ul>
          <li>Central lattice point</li>
          <li>Wigner-Seitz cell boundary</li>
          {showNeighbors && <li>Nearest neighbor lattice points</li>}
          {showConstruction && <li>Voronoi construction lines</li>}
        </ul>
      </div>
    </div>
  );
}
