'use client';

import { MoireBuilder2D } from './MoireBuilder2D';
import { useMoireLatticeState } from './MDXMoireStateProvider';

interface ContextMoireBuilder2DProps {
  // Display options
  width?: number;
  height?: number;
  showUnitCell?: boolean;
  showLatticeVectors?: boolean;
  showGrid?: boolean;
  showAxes?: boolean;
  showPoints?: boolean;
  showLattice1?: boolean;
  showLattice2?: boolean;
  showMoireLattice?: boolean;
  is_debug?: boolean;
  
  // Styling
  pointRadius?: number;
  vectorWidth?: number;
  gridOpacity?: number;
}

export function ContextMoireBuilder2D(props: ContextMoireBuilder2DProps) {
  const { setMoireLattice, setBaseLattice } = useMoireLatticeState();

  return (
    <MoireBuilder2D
      {...props}
      onMoireLatticeChange={setMoireLattice}
      onBaseLatticeChange={setBaseLattice}
    />
  );
}
