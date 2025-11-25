'use client';

import { useState, useEffect, useMemo } from 'react';
import { LatticeVisualization2D } from './LatticeVisualization2D';
import { getWasmModule } from '../providers/wasmLoader';
import type { WasmLattice2D } from '../../public/wasm/moire_lattice_wasm';
import { 
  ChevronDown, 
  Info, 
  FlipHorizontal2, 
  Grid3x3, 
  Maximize2,
  Activity,
  Ruler,
  Triangle
} from 'lucide-react';

type BravaisType = 'square' | 'hexagonal' | 'rectangular' | 'centered_rectangular' | 'oblique' | 'custom';

interface LatticeParameters {
  type: BravaisType;
  a: number;
  b: number;
  gamma: number; // angle in degrees
}

const PRESET_LATTICES: Record<Exclude<BravaisType, 'custom'>, { name: string; a: number; b: number; gamma: number }> = {
  square: { name: 'Square', a: 1, b: 1, gamma: 90 },
  hexagonal: { name: 'Hexagonal', a: 1, b: 1, gamma: 120 },
  rectangular: { name: 'Rectangular', a: 1, b: 1.5, gamma: 90 },
  centered_rectangular: { name: 'Centered Rectangular', a: 1, b: 1.5, gamma: 90 },
  oblique: { name: 'Oblique', a: 1, b: 1.2, gamma: 70 },
};

export function LatticeBuilder2D() {
  const [parameters, setParameters] = useState<LatticeParameters>({
    type: 'square',
    a: 1,
    b: 1,
    gamma: 90
  });
  
  const [isReciprocal, setIsReciprocal] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [latticeStats, setLatticeStats] = useState<any>(null);
  const [detectedLatticeType, setDetectedLatticeType] = useState<string>('');
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);
  
  // Initialize WASM
  useEffect(() => {
    getWasmModule().then(() => {
      setIsWasmLoaded(true);
    });
  }, []);
  
  // Calculate lattice statistics
  useEffect(() => {
    if (!isWasmLoaded) return;
    
    async function calculateStats() {
      try {
        const wasm = await getWasmModule();
        let lattice: WasmLattice2D | null = null;
        
        // Create lattice based on type
        if (parameters.type === 'custom') {
          // Create custom lattice using oblique type with custom parameters
          const params = {
            lattice_type: 'oblique',
            a: parameters.a,
            b: parameters.b,
            angle: parameters.gamma // angle in degrees
          };
          lattice = new wasm.WasmLattice2D(params);
        } else {
          // Use preset lattice
          switch (parameters.type) {
            case 'square':
              lattice = wasm.create_square_lattice(parameters.a);
              break;
            case 'hexagonal':
              lattice = wasm.create_hexagonal_lattice(parameters.a);
              break;
            case 'rectangular':
              lattice = wasm.create_rectangular_lattice(parameters.a, parameters.b);
              break;
            case 'centered_rectangular':
              lattice = wasm.create_centered_rectangular_lattice(parameters.a, parameters.b);
              break;
            case 'oblique':
              lattice = wasm.create_oblique_lattice(parameters.a, parameters.b, parameters.gamma);
              break;
          }
        }
        
        if (lattice) {
          // Get lattice vectors
          const vectors = lattice.lattice_vectors();
          const reciprocalVectors = lattice.reciprocal_vectors();
          
          // Get Wigner-Seitz cell
          const wignerSeitz = lattice.wigner_seitz_cell();
          const wignerSeitzData = wignerSeitz ? wignerSeitz.get_data() : null;
          
          // Get Brillouin zone
          const brillouinZone = lattice.brillouin_zone();
          const brillouinData = brillouinZone ? brillouinZone.get_data() : null;
          
          // Get coordination analysis
          const coordination = lattice.coordination_analysis();
          
          // Calculate metric tensor (G = A^T * A)
          const a1 = vectors.a ? [vectors.a.x, vectors.a.y] : [1, 0];
          const a2 = vectors.b ? [vectors.b.x, vectors.b.y] : [0, 1];
          
          const g11 = a1[0] * a1[0] + a1[1] * a1[1];
          const g12 = a1[0] * a2[0] + a1[1] * a2[1];
          const g22 = a2[0] * a2[0] + a2[1] * a2[1];
          
          // Calculate angles between vectors
          const dot = a1[0] * a2[0] + a1[1] * a2[1];
          const mag1 = Math.sqrt(g11);
          const mag2 = Math.sqrt(g22);
          const angle = Math.acos(dot / (mag1 * mag2)) * 180 / Math.PI;
          
          setLatticeStats({
            unitCellArea: lattice.unit_cell_area(),
            vectors: {
              a1: a1,
              a2: a2,
              magnitudes: [mag1, mag2],
              angle: angle
            },
            reciprocal: {
              b1: reciprocalVectors.a ? [reciprocalVectors.a.x, reciprocalVectors.a.y] : null,
              b2: reciprocalVectors.b ? [reciprocalVectors.b.x, reciprocalVectors.b.y] : null,
            },
            metricTensor: [[g11, g12], [g12, g22]],
            wignerSeitz: {
              area: wignerSeitzData?.measure || 0,
              vertices: wignerSeitzData?.vertices?.length || 0
            },
            brillouinZone: {
              area: brillouinData?.measure || 0,
              vertices: brillouinData?.vertices?.length || 0
            },
            coordination: coordination,
            bravaisType: wasm.bravais_type_to_string(lattice.bravais_type())
          });
          
          // Detect lattice type using validation functions
          const detectedType = wasm.determine_lattice_type_2d(lattice);
          const detectedTypeString = wasm.bravais_type_to_string(detectedType);
          setDetectedLatticeType(detectedTypeString);
          
          // Clean up
          lattice.free();
          if (wignerSeitz) wignerSeitz.free();
          if (brillouinZone) brillouinZone.free();
        }
      } catch (error) {
        console.error('Error calculating lattice stats:', error);
      }
    }
    
    calculateStats();
  }, [parameters, isWasmLoaded]);
  
  // Handle lattice type change
  const handleTypeChange = (type: BravaisType) => {
    if (type === 'custom') {
      setParameters(prev => ({ ...prev, type }));
    } else {
      const preset = PRESET_LATTICES[type];
      setParameters({
        type,
        a: preset.a,
        b: preset.b,
        gamma: preset.gamma
      });
    }
    setDropdownOpen(false);
  };
  
  // Get basis vectors for visualization
  const basisVectors = useMemo((): [[number, number], [number, number]] | undefined => {
    if (parameters.type === 'custom') {
      const angleRad = (parameters.gamma * Math.PI) / 180;
      return [
        [parameters.a, 0],
        [parameters.b * Math.cos(angleRad), parameters.b * Math.sin(angleRad)]
      ];
    }
    return undefined;
  }, [parameters]);
  
  return (
    <>
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .slider::-webkit-slider-track {
          height: 8px;
          background: linear-gradient(to right, #e5e7eb, #9ca3af);
          border-radius: 4px;
        }
        
        .slider::-moz-range-track {
          height: 8px;
          background: linear-gradient(to right, #e5e7eb, #9ca3af);
          border-radius: 4px;
        }
        
        .dark .slider::-webkit-slider-track {
          background: linear-gradient(to right, #374151, #6b7280);
        }
        
        .dark .slider::-moz-range-track {
          background: linear-gradient(to right, #374151, #6b7280);
        }
      `}</style>
      <div className="lattice-builder-2d w-full space-y-6">
      {/* Control Panel */}
      <div className="bg-transparent rounded-lg p-6">
        {/* Detected Lattice Type Status */}
        {detectedLatticeType && (
          <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <div className="flex items-center gap-2">
              <Grid3x3 className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Detected Lattice Type:
              </span>
              <span className="text-sm font-mono text-blue-900 dark:text-blue-100 capitalize">
                {detectedLatticeType.replace('_', ' ')}
              </span>
            </div>
          </div>
        )}
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column: Controls */}
          <div className="space-y-4">
            {/* Lattice Type Selector */}
            <div className="relative">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Lattice Type
              </label>
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="w-full px-4 py-2 text-left bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
              >
                <div className="flex items-center justify-between">
                  <span className="capitalize">
                    {parameters.type === 'custom' ? 'Custom' : PRESET_LATTICES[parameters.type].name}
                  </span>
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </div>
              </button>
              
              {dropdownOpen && (
                <div className="absolute z-10 mt-1 w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg">
                  {Object.entries(PRESET_LATTICES).map(([key, preset]) => (
                    <button
                      key={key}
                      onClick={() => handleTypeChange(key as Exclude<BravaisType, 'custom'>)}
                      className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                    >
                      {preset.name}
                    </button>
                  ))}
                  <button
                    onClick={() => handleTypeChange('custom')}
                    className="w-full px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors border-t border-gray-200 dark:border-gray-600"
                  >
                    Custom
                  </button>
                </div>
              )}
            </div>
            
            {/* Representation Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Space Representation
              </label>
              <button
                onClick={() => setIsReciprocal(!isReciprocal)}
                className={`w-full px-4 py-2 rounded-md transition-colors ${
                  isReciprocal
                    ? 'bg-orange-500 text-white hover:bg-orange-600'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                <FlipHorizontal2 className="w-4 h-4 inline mr-2" />
                {isReciprocal ? 'Reciprocal Space' : 'Direct Space'}
              </button>
            </div>
            
            {/* Debug Toggle */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Debug Information
              </label>
              <button
                onClick={() => setShowDebug(!showDebug)}
                className={`w-full px-4 py-2 rounded-md transition-colors ${
                  showDebug
                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                <Info className="w-4 h-4 inline mr-2" />
                {showDebug ? 'Hide Debug' : 'Show Debug'}
              </button>
            </div>
          </div>
          
          {/* Right Column: Sliders */}
          <div className="space-y-6">
            {/* Parameter a Slider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Parameter a: {parameters.a.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.01"
                max="10"
                step="0.01"
                value={parameters.a}
                onChange={(e) => setParameters(prev => ({ ...prev, a: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>0.01</span>
                <span>10.00</span>
              </div>
            </div>
            
            {/* Parameter b Slider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Parameter b: {parameters.b.toFixed(2)}
                {(parameters.type === 'square' || parameters.type === 'hexagonal') && (
                  <span className="text-xs text-gray-500 ml-2">(locked to a)</span>
                )}
              </label>
              <input
                type="range"
                min="0.01"
                max="10"
                step="0.01"
                value={parameters.b}
                onChange={(e) => setParameters(prev => ({ ...prev, b: parseFloat(e.target.value) }))}
                disabled={parameters.type === 'square' || parameters.type === 'hexagonal'}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>0.01</span>
                <span>10.00</span>
              </div>
            </div>
            
            {/* Angle gamma Slider */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Angle γ: {parameters.gamma.toFixed(0)}°
                {(parameters.type !== 'oblique' && parameters.type !== 'custom') && (
                  <span className="text-xs text-gray-500 ml-2">(constrained by lattice type)</span>
                )}
              </label>
              <input
                type="range"
                min="0"
                max="360"
                step="1"
                value={parameters.gamma}
                onChange={(e) => setParameters(prev => ({ ...prev, gamma: parseFloat(e.target.value) }))}
                disabled={parameters.type !== 'oblique' && parameters.type !== 'custom'}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>0°</span>
                <span>360°</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Visualization */}
      <div className="bg-transparent rounded-lg p-6">
        <LatticeVisualization2D
          latticeType={parameters.type === 'custom' ? 'oblique' : parameters.type}
          basisVectors={parameters.type === 'custom' ? undefined : basisVectors}
          a={parameters.a}
          b={parameters.b}
          gamma={parameters.gamma}
          is_reciprocal={isReciprocal}
          is_debug={showDebug}
          height={500}
          shells={4}
        />
      </div>
      
      {/* Statistics Panel */}
      {latticeStats && (
        <div className="bg-transparent rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
            Lattice Analysis
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Basic Properties */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Grid3x3 className="w-4 h-4" />
                Basic Properties
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Bravais Type:</span>
                  <span className="font-mono text-blue-600 dark:text-blue-400">{latticeStats.bravaisType}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Unit Cell Area:</span>
                  <span className="font-mono">{latticeStats.unitCellArea.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">|a₁|:</span>
                  <span className="font-mono">{latticeStats.vectors.magnitudes[0].toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">|a₂|:</span>
                  <span className="font-mono">{latticeStats.vectors.magnitudes[1].toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">∠(a₁, a₂):</span>
                  <span className="font-mono">{latticeStats.vectors.angle.toFixed(2)}°</span>
                </div>
              </div>
            </div>
            
            {/* Voronoi Cells */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Maximize2 className="w-4 h-4" />
                Voronoi Cells
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Wigner-Seitz Area:</span>
                  <span className="font-mono">{latticeStats.wignerSeitz.area.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">WS Vertices:</span>
                  <span className="font-mono">{latticeStats.wignerSeitz.vertices}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Brillouin Zone Area:</span>
                  <span className="font-mono">{latticeStats.brillouinZone.area.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">BZ Vertices:</span>
                  <span className="font-mono">{latticeStats.brillouinZone.vertices}</span>
                </div>
              </div>
            </div>
            
            {/* Vectors & Matrices */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Activity className="w-4 h-4" />
                Vectors & Matrices
              </div>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-600 dark:text-gray-400">a₁:</span>
                  <span className="font-mono ml-2">[{latticeStats.vectors.a1[0].toFixed(3)}, {latticeStats.vectors.a1[1].toFixed(3)}]</span>
                </div>
                <div>
                  <span className="text-gray-600 dark:text-gray-400">a₂:</span>
                  <span className="font-mono ml-2">[{latticeStats.vectors.a2[0].toFixed(3)}, {latticeStats.vectors.a2[1].toFixed(3)}]</span>
                </div>
                {latticeStats.reciprocal.b1 && (
                  <>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">b₁:</span>
                      <span className="font-mono ml-2">[{latticeStats.reciprocal.b1[0].toFixed(3)}, {latticeStats.reciprocal.b1[1].toFixed(3)}]</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">b₂:</span>
                      <span className="font-mono ml-2">[{latticeStats.reciprocal.b2[0].toFixed(3)}, {latticeStats.reciprocal.b2[1].toFixed(3)}]</span>
                    </div>
                  </>
                )}
              </div>
            </div>
            
            {/* Metric Tensor */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Triangle className="w-4 h-4" />
                Metric Tensor
              </div>
              <div className="font-mono text-sm bg-gray-100 dark:bg-gray-900 p-3 rounded">
                <div className="grid grid-cols-2 gap-2">
                  <div>[{latticeStats.metricTensor[0][0].toFixed(3)}</div>
                  <div>{latticeStats.metricTensor[0][1].toFixed(3)}]</div>
                  <div>[{latticeStats.metricTensor[1][0].toFixed(3)}</div>
                  <div>{latticeStats.metricTensor[1][1].toFixed(3)}]</div>
                </div>
              </div>
            </div>
            
            {/* Coordination */}
            {latticeStats.coordination && (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  <Ruler className="w-4 h-4" />
                  Coordination
                </div>
                <div className="space-y-2 text-sm">
                  {latticeStats.coordination.shells && latticeStats.coordination.shells.slice(0, 3).map((shell: any, index: number) => (
                    <div key={index} className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Shell {index + 1}:</span>
                      <span className="font-mono">{shell.count} at {shell.distance.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
    </>
  );
}
