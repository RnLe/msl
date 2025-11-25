'use client'

import { useEffect, useState } from 'react';
import { initWasm } from '../providers/wasmLoader';

interface ReciprocalLatticeVisualization2DProps {
  latticeType: 'oblique' | 'rectangular' | 'centered_rectangular' | 'square' | 'hexagonal';
  width?: number;
  height?: number;
  showBrillouinZone?: boolean;
  showHighSymmetryPoints?: boolean;
}

export function ReciprocalLatticeVisualization2D({ 
  latticeType, 
  width = 400, 
  height = 400,
  showBrillouinZone = true,
  showHighSymmetryPoints = true
}: ReciprocalLatticeVisualization2DProps) {
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
    return <div className="loading">Loading reciprocal lattice visualization...</div>;
  }

  return (
    <div className="reciprocal-lattice-visualization-2d">
      <div className="placeholder" style={{ width, height, border: '2px dashed #ccc', padding: '20px' }}>
        <h4>Reciprocal Lattice - {latticeType}</h4>
        <p>Interactive reciprocal lattice and Brillouin zone visualization will be implemented here.</p>
        <ul>
          <li>Reciprocal lattice points</li>
          {showBrillouinZone && <li>First Brillouin zone</li>}
          {showHighSymmetryPoints && <li>High symmetry points (Γ, K, M, etc.)</li>}
        </ul>
      </div>
    </div>
  );
}
