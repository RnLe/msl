'use client';

import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Stage, Layer, Circle, Line, Text, Arrow, Shape, Group } from 'react-konva';
import { getWasmModule } from '../providers/wasmLoader';
import type { Lattice2D, Moire2D } from '../../public/wasm/moire_lattice_wasm';
import { Square, SquareCheck } from 'lucide-react';

interface LatticeVisualization2DProps {
  // Option 1: Use predefined lattice types
  latticeType?: 'oblique' | 'rectangular' | 'centered_rectangular' | 'square' | 'hexagonal';
  
  // Option 2: Pass custom basis vectors [a1, a2] where each vector is [x, y]
  basisVectors?: [[number, number], [number, number]];
  
  // Lattice parameters for predefined types
  a?: number; // First lattice parameter
  b?: number; // Second lattice parameter (for rectangular, centered_rectangular, oblique)
  gamma?: number; // Angle in degrees (for oblique)
  
  // Display options
  width?: number;
  height?: number;
  showUnitCell?: boolean;
  showLatticeVectors?: boolean;
  showGrid?: boolean;
  showAxes?: boolean;
  showPoints?: boolean;
  shells?: number; // Number of shells to display
  is_reciprocal?: boolean; // Toggle for reciprocal lattice representation
  is_debug?: boolean; // Show debug statistics
  
  // Styling
  pointRadius?: number;
  vectorWidth?: number;
  gridOpacity?: number;
}

// Vintage blue color palette that works in both light and dark modes
const COLORS = {
  latticePoint: '#4682B4', // SteelBlue
  vector: '#191970', // MidnightBlue
  unitCell: '#6495ED', // CornflowerBlue
  grid: '#B0C4DE', // LightSteelBlue
  axes: '#696969', // DimGray
  text: '#2F4F4F', // DarkSlateGray
  background: 'transparent',
  voronoiCell: '#4682B4', // SteelBlue for Wigner-Seitz cell
  voronoiFill: 'rgba(70, 130, 180, 0.15)', // Semi-transparent SteelBlue
  // Orange counterparts for reciprocal lattice
  reciprocalLatticePoint: '#FF8C00', // DarkOrange
  reciprocalVector: '#CC5500', // Burnt Orange
  reciprocalUnitCell: '#FFA500', // Orange
  reciprocalGrid: '#FFE4B5', // Moccasin
  reciprocalVoronoiCell: '#FF8C00', // DarkOrange for Brillouin zone
  reciprocalVoronoiFill: 'rgba(255, 140, 0, 0.15)', // Semi-transparent DarkOrange
};

export function LatticeVisualization2D({ 
  latticeType,
  basisVectors,
  a = 1,
  b = 1,
  gamma = 90,
  width,
  height = 400,
  showUnitCell = true,
  showLatticeVectors = true,
  showGrid = true,
  showAxes = true,
  showPoints = true,
  shells = 3,
  is_reciprocal = false,
  is_debug = false,
  pointRadius = 4,
  vectorWidth = 3,
  gridOpacity = 0.2
}: LatticeVisualization2DProps) {
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);
  const [isLatticeReady, setIsLatticeReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lattice, setLattice] = useState<Lattice2D | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(width || 800);
  
  // Local state for toggle buttons
  const [localShowGrid, setLocalShowGrid] = useState(showGrid);
  const [localShowAxes, setLocalShowAxes] = useState(showAxes);
  const [localShowVectors, setLocalShowVectors] = useState(showLatticeVectors);
  const [localShowUnitCell, setLocalShowUnitCell] = useState(showUnitCell);
  const [localShowPoints, setLocalShowPoints] = useState(showPoints);
  
  // Voronoi cell state
  const [showVoronoiCell, setShowVoronoiCell] = useState(false);
  const [fillVoronoiSpace, setFillVoronoiSpace] = useState(false);
  
  // Debug timing and stats
  const [debugStats, setDebugStats] = useState({
    vectorsTime: 0,
    voronoiTime: 0,
    latticePointsTime: 0,
    gridLinesTime: 0,
    latticeCreationTime: 0,
    renderStartTime: 0
  });
  
  // Update local state when props change
  useEffect(() => {
    setLocalShowGrid(showGrid);
    setLocalShowAxes(showAxes);
    setLocalShowVectors(showLatticeVectors);
    setLocalShowUnitCell(showUnitCell);
    setLocalShowPoints(showPoints);
  }, [showGrid, showAxes, showLatticeVectors, showUnitCell, showPoints]);
  
  // Get colors based on representation type
  const currentColors = is_reciprocal ? {
    latticePoint: COLORS.reciprocalLatticePoint,
    vector: COLORS.reciprocalVector,
    unitCell: COLORS.reciprocalUnitCell,
    grid: COLORS.reciprocalGrid,
    voronoiCell: COLORS.reciprocalVoronoiCell,
    voronoiFill: COLORS.reciprocalVoronoiFill,
    axes: COLORS.axes,
    text: COLORS.text,
    background: COLORS.background
  } : {
    latticePoint: COLORS.latticePoint,
    vector: COLORS.vector,
    unitCell: COLORS.unitCell,
    grid: COLORS.grid,
    voronoiCell: COLORS.voronoiCell,
    voronoiFill: COLORS.voronoiFill,
    axes: COLORS.axes,
    text: COLORS.text,
    background: COLORS.background
  };
  
  // Handle responsive width
  useEffect(() => {
    if (!width && containerRef.current) {
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          setContainerWidth(entry.contentRect.width);
        }
      });
      
      resizeObserver.observe(containerRef.current);
      setContainerWidth(containerRef.current.offsetWidth);
      
      return () => {
        resizeObserver.disconnect();
      };
    }
  }, [width]);
  
  const canvasWidth = width || containerWidth;
  
  // Get lattice vectors first to calculate proper scale
  const vectors = useMemo(() => {
    const startTime = performance.now();
    
    // Wait for both WASM and lattice to be ready
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      // Get basis matrix (9 elements: column-major [a1x, a1y, a1z, a2x, a2y, a2z, ...]) 
      const basisFlat = is_reciprocal 
        ? lattice.getReciprocalBasis()
        : lattice.getDirectBasis();
      
      // Extract 2D vectors from the matrix (first two columns)
      const result = {
        a1: [basisFlat[0], basisFlat[1]], // First column
        a2: [basisFlat[3], basisFlat[4]]  // Second column
      };
      
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, vectorsTime: endTime - startTime }));
      
      return result;
    } catch (err) {
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, vectorsTime: endTime - startTime }));
      return { a1: [1, 0], a2: [0, 1] };
    }
  }, [lattice, isWasmLoaded, isLatticeReady, is_reciprocal]);
  
  // Get Voronoi cell data (Wigner-Seitz for direct, Brillouin zone for reciprocal)
  const voronoiCellData = useMemo(() => {
    const startTime = performance.now();
    
    // Wait for both WASM and lattice to be ready
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      // Get vertices as flat array [x, y, z, x, y, z, ...]
      const verticesFlat = is_reciprocal 
        ? lattice.getBrillouinZoneVertices()
        : lattice.getWignerSeitzVertices();
      
      if (!verticesFlat || verticesFlat.length === 0) {
        const endTime = performance.now();
        setDebugStats(prev => ({ ...prev, voronoiTime: endTime - startTime }));
        return null;
      }
      
      // Convert flat array to [x, y] pairs, ignoring z-component
      const vertices = [];
      for (let i = 0; i < verticesFlat.length; i += 3) {
        vertices.push([verticesFlat[i], verticesFlat[i + 1]]);
      }
      
      // Validate vertices for NaN/Infinity
      const validVertices = vertices.filter((vertex: number[]) => 
        vertex.length >= 2 && isFinite(vertex[0]) && isFinite(vertex[1])
      );
      
      if (is_debug && validVertices.length !== vertices.length) {
        console.warn(`Filtered out ${vertices.length - validVertices.length} invalid vertices`);
      }
      
      const result = {
        vertices: validVertices,
        edges: [], // Edge info not available from new API
        measure: 0  // Measure not available from new API
      };
      
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, voronoiTime: endTime - startTime }));
      
      return result;
    } catch (err) {
      console.warn('Failed to get Voronoi cell data:', err);
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, voronoiTime: endTime - startTime }));
      return null;
    }
  }, [lattice, isWasmLoaded, isLatticeReady, is_reciprocal]);

  // Canvas center and scale for coordinate transformation
  const centerX = canvasWidth / 2;
  const centerY = height / 2;
  
  // Calculate scale based on lattice type and vectors
  const scale = useMemo(() => {
    if (!vectors) {
      return Math.min(canvasWidth, height) / (2 * shells * 1.5);
    }
    
    // Calculate the maximum extent of the lattice vectors
    const maxExtent = Math.max(
      Math.sqrt(vectors.a1[0] ** 2 + vectors.a1[1] ** 2),
      Math.sqrt(vectors.a2[0] ** 2 + vectors.a2[1] ** 2),
      Math.sqrt((vectors.a1[0] + vectors.a2[0]) ** 2 + (vectors.a1[1] + vectors.a2[1]) ** 2)
    );
    
    // For reciprocal lattice, we need to account for the typical 2π factor
    const scaleFactor = is_reciprocal ? 1.5 : 1.5;
    
    return Math.min(canvasWidth, height) / (2 * shells * maxExtent * scaleFactor);
  }, [canvasWidth, height, shells, vectors, is_reciprocal]);
  
  // Grid spacing in canvas pixels (should align with lattice unit)
  const gridSpacing = scale;
  
  // (latticeCenter is computed later after latticePoints declaration)

  // Coordinate transformation functions
  const latticeToCanvas = (x: number, y: number): [number, number] => {
    return [
      centerX + x * scale,
      centerY - y * scale // Flip y-axis for standard math coordinates
    ];
  };

  const canvasToLattice = (canvasX: number, canvasY: number): [number, number] => {
    return [
      (canvasX - centerX) / scale,
      (centerY - canvasY) / scale
    ];
  };

  // Initialize WASM and create lattice with proper async handling
  useEffect(() => {
    let mounted = true;
    let currentLattice: Lattice2D | null = null;
    
    async function initializeAndCreateLattice() {
      try {
        // Reset states
        setError(null);
        // Don't set isLatticeReady to false to prevent canvas blanking
        
        const latticeCreateStartTime = performance.now();
        
        const wasm = await getWasmModule();
        if (!mounted) return;
        
        setIsWasmLoaded(true);
        
        // Small delay to ensure WASM is fully ready
        await new Promise(resolve => setTimeout(resolve, 100));
        if (!mounted) return;
        
        // Create lattice based on type or vectors
        if (basisVectors) {
          // Create lattice from custom basis vectors using new Lattice2D constructor
          const directMatrix = new Float64Array([
            basisVectors[0][0], basisVectors[0][1], 0,
            basisVectors[1][0], basisVectors[1][1], 0,
            0, 0, 1
          ]);
          currentLattice = new wasm.Lattice2D(directMatrix);
        } else if (latticeType) {
          // Create predefined lattice type
          switch (latticeType) {
            case 'square':
              currentLattice = wasm.create_square_lattice(a);
              break;
            case 'hexagonal':
              currentLattice = wasm.create_hexagonal_lattice(a);
              break;
            case 'rectangular':
              currentLattice = wasm.create_rectangular_lattice(a, b);
              break;
            case 'centered_rectangular':
              currentLattice = wasm.create_centered_rectangular_lattice(a, b);
              break;
            case 'oblique':
              currentLattice = wasm.create_oblique_lattice(a, b, gamma);
              break;
            default:
              throw new Error(`Unknown lattice type: ${latticeType}`);
          }
        } else {
          // Default to square lattice
          currentLattice = wasm.create_square_lattice(1);
        }
        
        // Validate that lattice was created successfully
        if (!currentLattice || typeof currentLattice !== 'object' || Object.keys(currentLattice).length === 0) {
          throw new Error('Failed to create lattice - WASM returned invalid object');
        }
        
        const latticeCreateEndTime = performance.now();
        
        if (mounted) {
          setLattice(currentLattice);
          setIsLatticeReady(true);
          setDebugStats(prev => ({ ...prev, latticeCreationTime: latticeCreateEndTime - latticeCreateStartTime }));
        } else if (currentLattice) {
          currentLattice.free();
        }
      } catch (err: any) {
        if (mounted) {
          setError(err.message);
        }
      }
    }
    
    initializeAndCreateLattice();
    
    return () => {
      mounted = false;
      if (currentLattice) {
        currentLattice.free();
      }
    };
  }, [latticeType, basisVectors, a, b, gamma]);

  const latticePoints = useMemo(() => {
    const startTime = performance.now();
    
    // Wait for both WASM and lattice to be ready
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return [];
    }
    
    try {
      // Calculate the lattice coordinate bounds based on canvas size and scale
      // We want to show lattice points that fit within the visible canvas area
      const latticeWidth = canvasWidth / scale;
      const latticeHeight = height / scale;
      
      // Validate the input parameters
      if (!isFinite(latticeWidth) || !isFinite(latticeHeight) || latticeWidth <= 0 || latticeHeight <= 0) {
        const endTime = performance.now();
        setDebugStats(prev => ({ ...prev, latticePointsTime: endTime - startTime }));
        return [];
      }
      
      // Get points as flat array [x, y, z, x, y, z, ...]
      const pointsFlat = is_reciprocal 
        ? lattice.getReciprocalLatticePoints(latticeWidth, latticeHeight)
        : lattice.getDirectLatticePoints(latticeWidth, latticeHeight);
      
      // Convert flat array to [x, y] pairs, ignoring z-component
      const result = [];
      for (let i = 0; i < pointsFlat.length; i += 3) {
        result.push([pointsFlat[i], pointsFlat[i + 1]]);
      }
      
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, latticePointsTime: endTime - startTime }));
      
      return result;
    } catch (err) {
      const endTime = performance.now();
      setDebugStats(prev => ({ ...prev, latticePointsTime: endTime - startTime }));
      return [];
    }
  }, [lattice, canvasWidth, height, scale, isWasmLoaded, isLatticeReady, is_reciprocal]);

  // Compute lattice bounding-box center (in lattice coordinates)
  const latticeCenter = useMemo(() => {
    if (!latticePoints || latticePoints.length === 0) {
      return [0, 0];
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of latticePoints) {
      if (!p || !Array.isArray(p) || p.length < 2) continue;
      const x = p[0];
      const y = p[1];
      if (!isFinite(x) || !isFinite(y)) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
      return [0, 0];
    }

    return [(minX + maxX) / 2, (minY + maxY) / 2];
  }, [latticePoints]);

  // Generate grid lines aligned with coordinate system
  const gridLines = useMemo(() => {
    const startTime = performance.now();
    
    const lines = [];
    
    // Calculate grid bounds with some padding
    const minX = -Math.ceil(canvasWidth / (2 * gridSpacing)) - 1;
    const maxX = Math.ceil(canvasWidth / (2 * gridSpacing)) + 1;
    const minY = -Math.ceil(height / (2 * gridSpacing)) - 1;
    const maxY = Math.ceil(height / (2 * gridSpacing)) + 1;
    
    // Vertical lines
    for (let i = minX; i <= maxX; i++) {
      const [x] = latticeToCanvas(i, 0);
      if (x >= -1 && x <= canvasWidth + 1) { // Small margin to show edge lines
        lines.push({
          key: `v-${i}`,
          points: [x, 0, x, height],
          stroke: currentColors.grid,
          strokeWidth: 1
        });
      }
    }
    
    // Horizontal lines
    for (let j = minY; j <= maxY; j++) {
      const [, y] = latticeToCanvas(0, j);
      if (y >= -1 && y <= height + 1) { // Small margin to show edge lines
        lines.push({
          key: `h-${j}`,
          points: [0, y, canvasWidth, y],
          stroke: currentColors.grid,
          strokeWidth: 1
        });
      }
    }
    
    const endTime = performance.now();
    setDebugStats(prev => ({ ...prev, gridLinesTime: endTime - startTime }));
    
    return lines;
  }, [canvasWidth, height, gridSpacing, centerX, centerY, scale, currentColors.grid]);

  // Track render performance for Voronoi space filling
  useEffect(() => {
    if (is_debug && fillVoronoiSpace) {
      const startTime = performance.now();
      
      // Use requestAnimationFrame to measure after render is complete
      requestAnimationFrame(() => {
        const endTime = performance.now();
        const renderTime = endTime - startTime;
        console.log(`Voronoi space fill render time: ${renderTime.toFixed(2)}ms`);
        setDebugStats(prev => ({ ...prev, renderStartTime: renderTime }));
      });
    }
  }, [fillVoronoiSpace, is_debug, latticePoints.length, voronoiCellData]);

  if (error) {
    return (
      <div className="text-red-500 p-4 border border-red-300 rounded">
        <strong>Error:</strong> {error}
      </div>
    );
  }

  if (!isWasmLoaded) {
    return (
      <div className="text-gray-500 p-4 border border-gray-300 rounded animate-pulse">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin"></div>
          <span>Loading WASM module...</span>
        </div>
      </div>
    );
  }

  if (!isLatticeReady || !lattice) {
    return (
      <div ref={containerRef} className="lattice-visualization-2d w-full flex flex-col items-center">
        <div style={{ width: canvasWidth, height: height }} className="relative">
          {/* Preserve canvas dimensions during loading */}
          <Stage width={canvasWidth} height={height}>
            <Layer>
              {/* Show a minimal loading indicator overlay */}
              <Text
                x={canvasWidth / 2 - 60}
                y={height / 2}
                text="Updating..."
                fontSize={14}
                fill="#6b7280"
                fontFamily="Arial"
              />
            </Layer>
          </Stage>
          {/* Optional: subtle loading overlay */}
          <div className="absolute top-2 right-2 flex items-center space-x-2 text-xs text-gray-500">
            <div className="w-3 h-3 border border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="lattice-visualization-2d w-full flex flex-col items-center">
      <Stage width={canvasWidth} height={height}>
        <Layer>
          {/* Background grid */}
          {localShowGrid && (
            <Group opacity={gridOpacity}>
              {gridLines.map((line) => {
                const { key, ...lineProps } = line;
                return <Line key={key} {...lineProps} />;
              })}
            </Group>
          )}
          
          {/* Coordinate axes */}
          {localShowAxes && (
            <Group>
              {/* X-axis */}
              <Arrow
                points={[20, height - 20, 80, height - 20]}
                stroke={currentColors.axes}
                strokeWidth={2}
                fill={currentColors.axes}
                pointerLength={8}
                pointerWidth={8}
              />
              {isFinite(height) && (
                <Text
                  x={85}
                  y={height - 25}
                  text="x"
                  fontSize={14}
                  fill={currentColors.text}
                />
              )}
              
              {/* Y-axis */}
              <Arrow
                points={[20, height - 20, 20, height - 80]}
                stroke={currentColors.axes}
                strokeWidth={2}
                fill={currentColors.axes}
                pointerLength={8}
                pointerWidth={8}
              />
              {isFinite(height) && (
                <Text
                  x={25}
                  y={height - 90}
                  text="y"
                  fontSize={14}
                  fill={currentColors.text}
                />
              )}
            </Group>
          )}
          
          {/* Unit cell */}
          {localShowUnitCell && vectors && (
            <Line
              points={[
                ...latticeToCanvas(0, 0),
                ...latticeToCanvas(vectors.a1[0], vectors.a1[1]),
                ...latticeToCanvas(vectors.a1[0] + vectors.a2[0], vectors.a1[1] + vectors.a2[1]),
                ...latticeToCanvas(vectors.a2[0], vectors.a2[1]),
                ...latticeToCanvas(0, 0)
              ]}
              stroke={currentColors.unitCell}
              strokeWidth={2}
              dash={[5, 5]}
            />
          )}
          
          {/* Voronoi cell visualization */}
          {(showVoronoiCell || fillVoronoiSpace) && voronoiCellData && voronoiCellData.vertices && (
            <Group>
              {/* Single Voronoi cell at origin */}
              {showVoronoiCell && voronoiCellData && voronoiCellData.vertices && voronoiCellData.vertices.length > 0 && (
                <Group>
                  {/* Filled cell */}
                  {(() => {
                    // Validate vertices and convert to canvas coordinates
                    // Single cell at origin - no centering offset
                    const canvasPoints = voronoiCellData.vertices
                      .filter((vertex: number[]) => vertex && Array.isArray(vertex) && vertex.length >= 2 && isFinite(vertex[0]) && isFinite(vertex[1]))
                      .flatMap((vertex: number[]) => {
                        const [canvasX, canvasY] = latticeToCanvas(vertex[0], vertex[1]);
                        
                        if (!isFinite(canvasX) || !isFinite(canvasY)) {
                          if (is_debug) {
                            console.warn('Invalid canvas coordinates in single cell:', vertex, '->', [canvasX, canvasY]);
                          }
                          return []; // Filter out invalid coordinates
                        }
                        
                        return [canvasX, canvasY];
                      });
                    
                    // Only render if we have enough valid points
                    if (canvasPoints.length < 6) {
                      if (is_debug) {
                        console.warn('Insufficient valid points for single Voronoi cell:', canvasPoints.length);
                      }
                      return null;
                    }
                    
                    // Validate all coordinates are finite
                    const allValid = canvasPoints.every((coord: number) => isFinite(coord));
                    if (!allValid) {
                      if (is_debug) {
                        console.warn('Invalid coordinates found in single Voronoi cell canvas points:', canvasPoints);
                      }
                      return null;
                    }
                    
                    return (
                      <Line
                        points={canvasPoints}
                        closed
                        fill={currentColors.voronoiFill}
                        stroke={currentColors.voronoiCell}
                        strokeWidth={2}
                      />
                    );
                  })()}
                </Group>
              )}
              
              {/* Fill space with Voronoi cells - reuse existing lattice points */}
              {fillVoronoiSpace &&
               latticePoints.length > 0 &&
               voronoiCellData &&
               voronoiCellData.vertices && (() => {
                 // Generate <Line> elements first
                 const cellElements = latticePoints
                   .map((point: [number, number] | number[], index: number) => {
                     // Validate input point
                     if (!point || !Array.isArray(point)) {
                       if (is_debug) {
                         console.warn(`Invalid point structure at index ${index}:`, point);
                       }
                       return null;
                     }
                     
                     const [siteX, siteY] = point.length >= 2 ? [point[0], point[1]] : [0, 0];
                     
                     // Validate lattice point coordinates
                     if (!isFinite(siteX) || !isFinite(siteY)) {
                       if (is_debug) {
                         console.warn(`Invalid lattice point at index ${index}:`, [siteX, siteY]);
                       }
                       return null;
                     }
                     
                     // Apply centering offset
                     const [cx, cy] = latticeCenter;
                     
                     // Translate vertices to this lattice site
                      const translatedVertices = voronoiCellData.vertices
                       .filter((vertex: number[]) => vertex && Array.isArray(vertex) && vertex.length >= 2)
                       .map((vertex: number[]) => [
                         vertex[0] + siteX - cx,
                         vertex[1] + siteY - cy
                       ])
                       .filter((vertex: number[]) => isFinite(vertex[0]) && isFinite(vertex[1])); // Filter out NaN/Infinity
                      
                      // Skip if we don't have enough valid vertices
                      if (translatedVertices.length < 3) {
                        if (is_debug) {
                          console.warn(`Insufficient valid vertices for cell ${index}:`, translatedVertices.length);
                        }
                        return null;
                      }
                      
                      // Convert to canvas coordinates and validate
                      const canvasPoints = translatedVertices.flatMap((vertex: number[]) => {
                        const [canvasX, canvasY] = latticeToCanvas(vertex[0], vertex[1]);
                        
                        // Validate canvas coordinates
                        if (!isFinite(canvasX) || !isFinite(canvasY)) {
                          if (is_debug) {
                            console.warn(`Invalid canvas coordinates for vertex:`, vertex, '->', [canvasX, canvasY]);
                          }
                          return []; // Return empty array to be filtered out by flatMap
                        }
                        
                        return [canvasX, canvasY];
                      });
                      
                      // Skip if we don't have enough valid canvas points
                      if (canvasPoints.length < 6) { // Need at least 3 vertices * 2 coordinates
                        if (is_debug) {
                          console.warn(`Insufficient valid canvas points for cell ${index}:`, canvasPoints.length);
                        }
                        return null;
                      }
                      
                      // Validate that canvasPoints contains only finite numbers
                      const allPointsValid = canvasPoints.every((coord: number) => isFinite(coord));
                      if (!allPointsValid) {
                        if (is_debug) {
                          console.warn(`Canvas points contain invalid values for cell ${index}:`, canvasPoints);
                        }
                        return null;
                      }
                      
                      return (
                        <Line
                          key={`voronoi-fill-${index}`}
                          points={canvasPoints}
                          closed
                          fill={currentColors.voronoiFill}
                          stroke={currentColors.voronoiCell}
                          strokeWidth={1}
                        />
                      );
                    })
                    .filter((el: any) => el != null);

                 // Render <Group> only when we actually have children
                 return cellElements.length > 0 ? <Group>{cellElements}</Group> : null;
               })()}
            </Group>
          )}
          
          {/* Lattice points */}
          {localShowPoints && latticePoints.length > 0 ? (
            latticePoints
              .map((point: [number, number] | number[], index: number) => {
                // Validate point structure
                if (!point || !Array.isArray(point) || point.length < 2) {
                  return null;
                }
                
                const [x, y] = [point[0], point[1]];
                
                // Validate coordinates
                if (!isFinite(x) || !isFinite(y)) {
                  return null;
                }
                
                // Apply centering offset for lattice points only
                const [cx, cy] = latticeCenter;
                const [canvasX, canvasY] = latticeToCanvas(x - cx, y - cy);
                
                // Validate canvas coordinates
                if (!isFinite(canvasX) || !isFinite(canvasY)) {
                  return null;
                }
                
                // Only render points that are visible on canvas
                if (canvasX >= -pointRadius && canvasX <= canvasWidth + pointRadius && 
                    canvasY >= -pointRadius && canvasY <= height + pointRadius) {
                  return (
                    <Circle
                      key={`point-${index}`}
                      x={canvasX}
                      y={canvasY}
                      radius={pointRadius}
                      fill={currentColors.latticePoint}
                      stroke={currentColors.vector}
                      strokeWidth={1}
                    />
                  );
                }
                return null;
              })
              .filter((element: any) => element !== null && element !== undefined)
          ) : (
            localShowPoints && isFinite(centerX) && isFinite(centerY) && (
              <Text
                x={centerX - 50}
                y={centerY - 10}
                text="No points generated"
                fontSize={12}
                fill={currentColors.text}
              />
            )
          )}
          
          {/* Lattice vectors */}
          {localShowVectors && vectors && (
            <Group>
              {/* a1/b1 vector */}
              <Arrow
                points={[
                  ...latticeToCanvas(0, 0),
                  ...latticeToCanvas(vectors.a1[0], vectors.a1[1])
                ]}
                stroke={currentColors.vector}
                strokeWidth={vectorWidth}
                fill={currentColors.vector}
                pointerLength={10}
                pointerWidth={10}
              />
              {(() => {
                // Calculate midpoint of a1 vector
                const midX = vectors.a1[0] * 0.5;
                const midY = vectors.a1[1] * 0.5;
                
                // Calculate perpendicular vector (rotate 90° counterclockwise)
                const perpX = -vectors.a1[1];
                const perpY = vectors.a1[0];
                
                // Normalize perpendicular vector and scale for offset
                const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                const offsetScale = is_reciprocal ? 1.2 : 0.2; 
                const normalizedPerpX = (perpX / perpLength) * offsetScale;
                const normalizedPerpY = (perpY / perpLength) * offsetScale;
                
                // Calculate label position
                const labelX = midX + normalizedPerpX;
                const labelY = midY + normalizedPerpY;
                
                const [canvasX, canvasY] = latticeToCanvas(labelX, labelY);
                
                const labelText = is_reciprocal ? "b₁" : "a₁";
                
                // Validate text content and canvas coordinates
                if (!labelText || labelText.trim() === "" || !isFinite(canvasX) || !isFinite(canvasY)) {
                  return null;
                }
                
                return (
                  <Text
                    x={canvasX - 10}
                    y={canvasY - 10}
                    text={labelText}
                    fontSize={18}
                    fill={currentColors.axes}
                    fontStyle="bold"
                  />
                );
              })()}
              
              {/* a2/b2 vector */}
              <Arrow
                points={[
                  ...latticeToCanvas(0, 0),
                  ...latticeToCanvas(vectors.a2[0], vectors.a2[1])
                ]}
                stroke={currentColors.vector}
                strokeWidth={vectorWidth}
                fill={currentColors.vector}
                pointerLength={10}
                pointerWidth={10}
              />
              {(() => {
                // Calculate midpoint of a2 vector
                const midX = vectors.a2[0] * 0.5;
                const midY = vectors.a2[1] * 0.5;
                
                // Calculate perpendicular vector (rotate 90° counterclockwise)
                const perpX = -vectors.a2[1];
                const perpY = vectors.a2[0];
                
                // Normalize perpendicular vector and scale for offset
                const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                const offsetScale = is_reciprocal ? 1.2 : 0.2; 
                const normalizedPerpX = (perpX / perpLength) * offsetScale;
                const normalizedPerpY = (perpY / perpLength) * offsetScale;
                
                // Calculate label position
                const labelX = midX + normalizedPerpX;
                const labelY = midY + normalizedPerpY;
                
                const [canvasX, canvasY] = latticeToCanvas(labelX, labelY);
                
                const labelText = is_reciprocal ? "b₂" : "a₂";
                
                // Validate text content and canvas coordinates
                if (!labelText || labelText.trim() === "" || !isFinite(canvasX) || !isFinite(canvasY)) {
                  return null;
                }
                
                return (
                  <Text
                    x={canvasX - 10}
                    y={canvasY - 10}
                    text={labelText}
                    fontSize={18}
                    fill={currentColors.axes}
                    fontStyle="bold"
                  />
                );
              })()}
            </Group>
          )}
        </Layer>
      </Stage>
      
      {/* Debug Statistics */}
      {is_debug && (
        <div className="w-full max-w-4xl mx-auto mt-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs font-mono">
            {/* Performance Metrics */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">⏱️ Performance</h4>
              <div>Lattice Creation: <span className="text-blue-600 dark:text-blue-400">{debugStats.latticeCreationTime.toFixed(2)}ms</span></div>
              <div>Vectors: <span className="text-blue-600 dark:text-blue-400">{debugStats.vectorsTime.toFixed(2)}ms</span></div>
              <div>Voronoi Cell: <span className="text-blue-600 dark:text-blue-400">{debugStats.voronoiTime.toFixed(2)}ms</span></div>
              <div>Lattice Points: <span className="text-blue-600 dark:text-blue-400">{debugStats.latticePointsTime.toFixed(2)}ms</span></div>
              <div>Grid Lines: <span className="text-blue-600 dark:text-blue-400">{debugStats.gridLinesTime.toFixed(2)}ms</span></div>
              <div>Fill Render: <span className="text-blue-600 dark:text-blue-400">{debugStats.renderStartTime.toFixed(2)}ms</span></div>
            </div>
            
            {/* Lattice Information */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Lattice Info</h4>
              <div>Type: <span className="text-green-600 dark:text-green-400">{latticeType || 'square'}</span></div>
              <div>Mode: <span className="text-green-600 dark:text-green-400">{is_reciprocal ? 'reciprocal' : 'direct'}</span></div>
              <div>Points Count: <span className="text-green-600 dark:text-green-400">{latticePoints.length}</span></div>
              <div>Grid Lines: <span className="text-green-600 dark:text-green-400">{gridLines.length}</span></div>
              <div>Canvas: <span className="text-green-600 dark:text-green-400">{canvasWidth}×{height}px</span></div>
            </div>
            
            {/* Lattice Vectors */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">📐 Vectors</h4>
              {vectors ? (
                <>
                  <div>a₁: <span className="text-purple-600 dark:text-purple-400">[{vectors.a1[0].toFixed(3)}, {vectors.a1[1].toFixed(3)}]</span></div>
                  <div>a₂: <span className="text-purple-600 dark:text-purple-400">[{vectors.a2[0].toFixed(3)}, {vectors.a2[1].toFixed(3)}]</span></div>
                  <div>|a₁|: <span className="text-purple-600 dark:text-purple-400">{Math.sqrt(vectors.a1[0]**2 + vectors.a1[1]**2).toFixed(3)}</span></div>
                  <div>|a₂|: <span className="text-purple-600 dark:text-purple-400">{Math.sqrt(vectors.a2[0]**2 + vectors.a2[1]**2).toFixed(3)}</span></div>
                </>
              ) : (
                <div className="text-gray-400">Loading...</div>
              )}
            </div>
            
            {/* Rectangle Dimensions */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">📏 Rectangle</h4>
              <div>Width: <span className="text-orange-600 dark:text-orange-400">{(canvasWidth / scale).toFixed(3)}</span></div>
              <div>Height: <span className="text-orange-600 dark:text-orange-400">{(height / scale).toFixed(3)}</span></div>
              <div>Scale: <span className="text-orange-600 dark:text-orange-400">{scale.toFixed(3)}</span></div>
              <div>Shells: <span className="text-orange-600 dark:text-orange-400">{shells}</span></div>
            </div>
            
            {/* Polyhedron Stats */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">🔶 Polyhedron</h4>
              {voronoiCellData ? (
                <>
                  <div>Vertices: <span className="text-red-600 dark:text-red-400">{voronoiCellData.vertices.length}</span></div>
                  <div>Edges: <span className="text-red-600 dark:text-red-400">{voronoiCellData.edges.length}</span></div>
                  <div>Measure: <span className="text-red-600 dark:text-red-400">{voronoiCellData.measure.toFixed(6)}</span></div>
                  {fillVoronoiSpace && (
                    <div>Rendered Cells: <span className="text-red-600 dark:text-red-400">{latticePoints.length}</span></div>
                  )}
                </>
              ) : (
                <div className="text-gray-400">Loading...</div>
              )}
            </div>
            
            {/* Polyhedron Points */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">📍 Polyhedron Points</h4>
              {voronoiCellData ? (
                <div className="max-h-32 overflow-y-auto text-xs">
                  {voronoiCellData.vertices.map((vertex: number[], index: number) => (
                    <div key={index}>
                      v{index}: <span className="text-cyan-600 dark:text-cyan-400">[{vertex[0].toFixed(3)}, {vertex[1].toFixed(3)}]</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-400">Loading...</div>
              )}
            </div>
            
            {/* State Information */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">🔄 State</h4>
              <div>WASM: <span className={isWasmLoaded ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{isWasmLoaded ? 'loaded' : 'loading'}</span></div>
              <div>Lattice: <span className={isLatticeReady ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{isLatticeReady ? 'ready' : 'creating'}</span></div>
              <div>Show Voronoi: <span className={showVoronoiCell ? "text-green-600 dark:text-green-400" : "text-gray-600 dark:text-gray-400"}>{showVoronoiCell ? 'yes' : 'no'}</span></div>
              <div>Fill Space: <span className={fillVoronoiSpace ? "text-green-600 dark:text-green-400" : "text-gray-600 dark:text-gray-400"}>{fillVoronoiSpace ? 'yes' : 'no'}</span></div>
            </div>
            
            {/* Validation Information */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">🔍 Validation</h4>
              {(() => {
                const invalidLatticePoints = latticePoints.filter((point: [number, number] | number[]) => {
                  const [x, y] = Array.isArray(point) ? point : [0, 0];
                  return !isFinite(x) || !isFinite(y);
                });
                
                const invalidVertices = voronoiCellData ? voronoiCellData.vertices.filter((vertex: number[]) => 
                  !isFinite(vertex[0]) || !isFinite(vertex[1])
                ) : [];
                
                const scaleValid = isFinite(scale) && scale > 0;
                
                return (
                  <>
                    <div>Scale Valid: <span className={scaleValid ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{scaleValid ? 'yes' : 'no'}</span></div>
                    <div>Invalid Points: <span className={invalidLatticePoints.length === 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{invalidLatticePoints.length}</span></div>
                    <div>Invalid Vertices: <span className={invalidVertices.length === 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{invalidVertices.length}</span></div>
                    <div>Canvas Size Valid: <span className={isFinite(canvasWidth) && isFinite(height) ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"}>{isFinite(canvasWidth) && isFinite(height) ? 'yes' : 'no'}</span></div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      )}
      
      {/* Toggle buttons with subtle styling */}
      <div className="flex justify-evenly w-full gap-2 mt-4">
        <div
          onClick={() => setLocalShowGrid(!localShowGrid)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Grid</span>
          {localShowGrid ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowAxes(!localShowAxes)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Axes</span>
          {localShowAxes ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowVectors(!localShowVectors)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Vectors</span>
          {localShowVectors ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowUnitCell(!localShowUnitCell)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Unit Cell</span>
          {localShowUnitCell ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowPoints(!localShowPoints)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Points</span>
          {localShowPoints ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
      
      {/* Second row: Voronoi cell controls */}
      <div className="flex justify-center w-full gap-2 mt-2">
        <div
          onClick={() => setShowVoronoiCell(!showVoronoiCell)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1 max-w-xs"
        >
          <span className="text-sm font-medium mr-2">
            {is_reciprocal ? 'Show Brillouin Zone' : 'Show Wigner-Seitz Cell'}
          </span>
          {showVoronoiCell ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setFillVoronoiSpace(!fillVoronoiSpace)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1 max-w-xs"
        >
          <span className="text-sm font-medium mr-2">
            {is_reciprocal ? 'Fill with Brillouin Zones' : 'Fill with Wigner-Seitz Cells'}
          </span>
          {fillVoronoiSpace ? (
            <SquareCheck 
              className={`w-4 h-4 flex-shrink-0 ${
                is_reciprocal ? 'text-orange-600 dark:text-orange-400' : 'text-blue-600 dark:text-blue-400'
              }`} 
            />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
    </div>
  );
}