'use client';

import { useMoireLatticeState } from './MDXMoireStateProvider';
import { useMemo } from 'react';

interface MoirePropertiesDisplayProps {
  height?: number;
  showAdvancedProperties?: boolean;
}

export function MoirePropertiesDisplay({ 
  height = 300, 
  showAdvancedProperties = false 
}: MoirePropertiesDisplayProps) {
  const { moireLattice, baseLattice } = useMoireLatticeState();

  const properties = useMemo(() => {
    if (!moireLattice || !baseLattice) {
      return null;
    }

    try {
      const periodRatio = moireLattice.moire_period_ratio();
      const twistAngle = moireLattice.twist_angle_degrees();
      
      return {
        periodRatio,
        twistAngle,
        isValid: true
      };
    } catch (error) {
      console.error('Error calculating moiré properties:', error);
      return {
        periodRatio: 0,
        twistAngle: 0,
        isValid: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }, [moireLattice, baseLattice]);

  if (!moireLattice || !baseLattice) {
    return (
      <div 
        className="flex items-center justify-center border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-800"
        style={{ height }}
      >
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-gray-500 dark:text-gray-400">
            Waiting for moiré lattice data...
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
            Adjust the parameters in the builder above
          </p>
        </div>
      </div>
    );
  }

  if (!properties?.isValid) {
    return (
      <div 
        className="flex items-center justify-center border border-red-300 dark:border-red-600 rounded-lg bg-red-50 dark:bg-red-900/20"
        style={{ height }}
      >
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 font-medium">
            Error calculating moiré properties
          </p>
          {properties?.error && (
            <p className="text-xs text-red-500 dark:text-red-400 mt-1">
              {properties.error}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div 
      className="border border-gray-300 dark:border-gray-600 rounded-lg p-6 bg-white dark:bg-gray-900"
      style={{ minHeight: height }}
    >
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Moiré Lattice Properties
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Period Ratio:</span>
          <span className="text-sm bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 px-3 py-1 rounded-full font-mono">
            {properties.periodRatio.toFixed(3)}
          </span>
        </div>
        
        <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Twist Angle:</span>
          <span className="text-sm bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-3 py-1 rounded-full font-mono">
            {properties.twistAngle.toFixed(2)}°
          </span>
        </div>
      </div>

      {showAdvancedProperties && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <h4 className="text-md font-medium mb-3 text-gray-800 dark:text-gray-200">
            Advanced Properties
          </h4>
          <div className="space-y-2">
            <div className="text-xs text-gray-600 dark:text-gray-400">
              <span className="font-medium">Moiré Period:</span> λ = a₁ × {properties.periodRatio.toFixed(3)}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">
              <span className="font-medium">Magic Angle Range:</span> {properties.twistAngle < 2 ? '✓ Near magic angle' : '✗ Not in magic angle range'}
            </div>
          </div>
        </div>
      )}
      
      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <p className="text-xs text-blue-700 dark:text-blue-300">
          💡 This component automatically updates when you change parameters in the builder above, 
          demonstrating shared state across multiple visualizations in the same MDX document.
        </p>
      </div>
    </div>
  );
}
