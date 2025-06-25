'use client';

import { HighSymmetryVisualization2D } from './HighSymmetryVisualization2D';
import { useMoireLatticeState } from './MDXMoireStateProvider';

interface ContextHighSymmetryVisualization2DProps {
  // Display options
  width?: number;
  height?: number;
  showLatticeVectors?: boolean;
  showGrid?: boolean;
  showAxes?: boolean;
  showPoints?: boolean;
  shells?: number; // Number of shells to display
  
  // Styling
  pointRadius?: number;
  vectorWidth?: number;
  gridOpacity?: number;
  
  // Optional: Use base lattice or moiré lattice
  useBaseLattice?: boolean; // If true, use base lattice instead of moiré lattice
}

export function ContextHighSymmetryVisualization2D({
  useBaseLattice = false,
  ...props
}: ContextHighSymmetryVisualization2DProps) {
  const { moireLattice, baseLattice } = useMoireLatticeState();

  // Choose which lattice to use
  const latticeToUse = useBaseLattice ? baseLattice : moireLattice;

  if (!latticeToUse) {
    return (
      <div 
        className="flex items-center justify-center border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-800"
        style={{ height: props.height || 400 }}
      >
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-gray-500 dark:text-gray-400">
            Waiting for {useBaseLattice ? 'base' : 'moiré'} lattice data...
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            Adjust the parameters in the builder above
          </p>
        </div>
      </div>
    );
  }

  return (
    <HighSymmetryVisualization2D
      {...props}
      customLattice={latticeToUse}
    />
  );
}
