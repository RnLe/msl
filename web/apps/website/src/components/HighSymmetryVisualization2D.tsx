'use client';

import { useEffect, useState, useMemo, useRef } from 'react';
import { Stage, Layer, Circle, Line, Arrow, Text, Group } from 'react-konva';
import { getWasmModule } from '../providers/wasmLoader';
import type { WasmLattice2D, WasmMoire2D } from '../../public/wasm/moire_lattice_wasm';
import { Square, SquareCheck } from 'lucide-react';

interface HighSymmetryVisualization2DProps {
  // Option 1: Use predefined lattice types
  latticeType?: 'oblique' | 'rectangular' | 'centered_rectangular' | 'square' | 'hexagonal';
  
  // Option 2: Pass custom basis vectors [a1, a2] where each vector is [x, y]
  basisVectors?: [[number, number], [number, number]];
  
  // Option 3: Pass a custom lattice directly
  customLattice?: WasmLattice2D | WasmMoire2D;
  
  // Lattice parameters for predefined types
  a?: number; // First lattice parameter
  b?: number; // Second lattice parameter (for rectangular, centered_rectangular, oblique)
  gamma?: number; // Angle in degrees (for oblique)
  
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
}

// Orange color palette for reciprocal lattice with vintage high symmetry colors
const COLORS = {
  latticePoint: '#FF8C00', // DarkOrange
  vector: '#CC5500', // Burnt Orange
  grid: '#FFE4B5', // Moccasin
  axes: '#696969', // DimGray
  text: '#2F4F4F', // DarkSlateGray
  background: 'transparent',
  brillouinZone: '#FF8C00', // DarkOrange for Brillouin zone
  brillouinZoneFill: 'rgba(255, 140, 0, 0.15)', // Semi-transparent DarkOrange
  // Vintage colors for high symmetry elements
  highSymmetryPoint: '#9f80b9', // SaddleBrown - vintage brown
  highSymmetryPath: '#9f80b9', // DarkSlateGray - vintage dark gray
  highSymmetryLabel: '#9f80b9', // SaddleBrown for labels
};

export function HighSymmetryVisualization2D({ 
  latticeType,
  basisVectors,
  customLattice,
  a = 1,
  b = 1,
  gamma = 90,
  width,
  height = 400,
  showLatticeVectors = false,
  showGrid = true,
  showAxes = true,
  showPoints = true,
  shells = 3,
  pointRadius = 4,
  vectorWidth = 3,
  gridOpacity = 0.2
}: HighSymmetryVisualization2DProps) {
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);
  const [isLatticeReady, setIsLatticeReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lattice, setLattice] = useState<WasmLattice2D | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(width || 800);
  
  // Local state for toggle buttons
  const [localShowGrid, setLocalShowGrid] = useState(showGrid);
  const [localShowAxes, setLocalShowAxes] = useState(showAxes);
  const [localShowVectors, setLocalShowVectors] = useState(showLatticeVectors);
  const [localShowPoints, setLocalShowPoints] = useState(showPoints);
  
  // Update local state when props change
  useEffect(() => {
    setLocalShowGrid(showGrid);
    setLocalShowAxes(showAxes);
    setLocalShowVectors(showLatticeVectors);
    setLocalShowPoints(showPoints);
  }, [showGrid, showAxes, showLatticeVectors, showPoints]);
  
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
  
  // Helper function to convert any lattice to WasmLattice2D
  const convertToLattice2D = (inputLattice: WasmLattice2D | WasmMoire2D): WasmLattice2D => {
    // Check if it's already a WasmLattice2D or if it has as_lattice2d method
    if ('as_lattice2d' in inputLattice && typeof inputLattice.as_lattice2d === 'function') {
      // It's a WasmMoire2D, convert it
      return inputLattice.as_lattice2d();
    } else {
      // It's already a WasmLattice2D
      return inputLattice as WasmLattice2D;
    }
  };
  
  // Get reciprocal lattice vectors
  const vectors = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      const vectorResult = lattice.reciprocal_vectors();
      
      // Handle the Result format from WASM
      const result = vectorResult && vectorResult.a && vectorResult.b ? {
        a1: [vectorResult.a.x, vectorResult.a.y],
        a2: [vectorResult.b.x, vectorResult.b.y]
      } : { a1: [1, 0], a2: [0, 1] };
      
      return result;
    } catch (err) {
      return { a1: [1, 0], a2: [0, 1] };
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);
  
  // Get high symmetry points
  const highSymmetryPoints = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return [];
    }
    
    try {
      const points = lattice.get_high_symmetry_points();
      return points || [];
    } catch (err) {
      console.warn('Failed to get high symmetry points:', err);
      return [];
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);
  
  // Get high symmetry path
  const highSymmetryPath = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return [];
    }
    
    try {
      const pathData = lattice.get_high_symmetry_path();
      return pathData?.points || [];
    } catch (err) {
      console.warn('Failed to get high symmetry path:', err);
      return [];
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);
  
  // Get Brillouin zone data
  const brillouinZoneData = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      const polyhedron = lattice.brillouin_zone();
      
      if (!polyhedron) {
        return null;
      }
      
      // Get polyhedron data
      const data = polyhedron.get_data();
      
      if (data && data.vertices) {
        // Extract vertices as [x, y] pairs, ignoring z-component for 2D
        const vertices = data.vertices.map((vertex: any) => [
          vertex.x !== undefined ? vertex.x : vertex[0], 
          vertex.y !== undefined ? vertex.y : vertex[1]
        ]);
        
        // Validate vertices for NaN/Infinity
        const validVertices = vertices.filter((vertex: [number, number]) => 
          isFinite(vertex[0]) && isFinite(vertex[1])
        );
        
        const result = {
          vertices: validVertices,
          edges: data.edges || [],
          measure: data.measure || 0
        };
        
        return result;
      }
      
      return null;
    } catch (err) {
      console.warn('Failed to get Brillouin zone data:', err);
      return null;
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);

  // Canvas center and scale for coordinate transformation
  const centerX = canvasWidth / 2;
  const centerY = height / 2;
  
  // Calculate scale based on lattice type, vectors, and shells parameter
  const scale = useMemo(() => {
    if (!vectors) {
      return Math.min(canvasWidth, height) / (shells * 1.5);
    }
    
    // Calculate the maximum extent of the lattice vectors
    const maxExtent = Math.max(
      Math.sqrt(vectors.a1[0] ** 2 + vectors.a1[1] ** 2),
      Math.sqrt(vectors.a2[0] ** 2 + vectors.a2[1] ** 2),
      Math.sqrt((vectors.a1[0] + vectors.a2[0]) ** 2 + (vectors.a1[1] + vectors.a2[1]) ** 2)
    );
    
    // Scale to fit the desired number of shells in the canvas
    // Zoom in 50% more by reducing the scale factor from 2.0 to 1.3
    const scaleFactor = 1.3;
    
    return Math.min(canvasWidth, height) / (shells * maxExtent * scaleFactor);
  }, [canvasWidth, height, shells, vectors]);
  
  // Grid spacing in canvas pixels (should align with lattice unit)
  const gridSpacing = scale;
  
  // Coordinate transformation functions
  const latticeToCanvas = (x: number, y: number): [number, number] => {
    return [
      centerX + x * scale,
      centerY - y * scale // Flip y-axis for standard math coordinates
    ];
  };

  // Initialize WASM and create lattice with proper async handling
  useEffect(() => {
    let mounted = true;
    let currentLattice: WasmLattice2D | null = null;
    
    async function initializeAndCreateLattice() {
      try {
        // Reset states
        setError(null);
        setIsLatticeReady(false);
        
        // If we have a custom lattice, use it directly without WASM initialization
        if (customLattice) {
          try {
            currentLattice = convertToLattice2D(customLattice);
            if (mounted) {
              setLattice(currentLattice);
              setIsLatticeReady(true);
              setIsWasmLoaded(true); // Set this for consistency
            }
            return;
          } catch (err: any) {
            if (mounted) {
              setError(`Failed to use custom lattice: ${err.message}`);
            }
            return;
          }
        }
        
        // For other cases, initialize WASM
        const wasm = await getWasmModule();
        if (!mounted) return;
        
        setIsWasmLoaded(true);
        
        // Small delay to ensure WASM is fully ready
        await new Promise(resolve => setTimeout(resolve, 100));
        if (!mounted) return;
        
        // Create lattice based on type, vectors, or custom lattice
        if (basisVectors) {
          // Create lattice from custom basis vectors
          const params = {
            a1: basisVectors[0],
            a2: basisVectors[1]
          };
          currentLattice = new wasm.WasmLattice2D(params);
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
        
        if (mounted) {
          setLattice(currentLattice);
          setIsLatticeReady(true);
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
  }, [latticeType, basisVectors, customLattice, a, b, gamma]);

  const latticePoints = useMemo(() => {
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
        return [];
      }
      
      // Get reciprocal lattice points
      const pointsResult = lattice.get_reciprocal_lattice_points_in_rectangle(latticeWidth, latticeHeight);
      
      let result = [];
      
      // Since it's a Result, the actual data should be in pointsResult directly if successful
      if (Array.isArray(pointsResult)) {
        result = pointsResult.map((point: any) => [point.x, point.y]);
      } else if (pointsResult && Array.isArray(pointsResult.points)) {
        result = pointsResult.points.map((point: any) => [point.x, point.y]);
      } else if (pointsResult && pointsResult.x !== undefined && pointsResult.y !== undefined) {
        // Single point case
        result = [[pointsResult.x, pointsResult.y]];
      }
      
      return result;
    } catch (err) {
      console.warn('Failed to generate lattice points:', err);
      return [];
    }
  }, [lattice, canvasWidth, height, scale, isWasmLoaded, isLatticeReady]);

  // Generate grid lines aligned with coordinate system
  const gridLines = useMemo(() => {
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
          stroke: COLORS.grid,
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
          stroke: COLORS.grid,
          strokeWidth: 1
        });
      }
    }
    
    return lines;
  }, [canvasWidth, height, gridSpacing, centerX, centerY, scale]);

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
          <div className="w-4 h-4 border-2 border-gray-300 border-t-orange-600 rounded-full animate-spin"></div>
          <span>Loading WASM module...</span>
        </div>
      </div>
    );
  }

  if (!isLatticeReady || !lattice) {
    return (
      <div className="text-gray-500 p-4 border border-gray-300 rounded animate-pulse">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 border-2 border-gray-300 border-t-orange-600 rounded-full animate-spin"></div>
          <span>Creating lattice...</span>
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="high-symmetry-visualization-2d w-full flex flex-col items-center">
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
                stroke={COLORS.axes}
                strokeWidth={2}
                fill={COLORS.axes}
                pointerLength={8}
                pointerWidth={8}
              />
              {isFinite(height) && (
                <Text
                  x={85}
                  y={height - 25}
                  text="x"
                  fontSize={14}
                  fill={COLORS.text}
                />
              )}
              
              {/* Y-axis */}
              <Arrow
                points={[20, height - 20, 20, height - 80]}
                stroke={COLORS.axes}
                strokeWidth={2}
                fill={COLORS.axes}
                pointerLength={8}
                pointerWidth={8}
              />
              {isFinite(height) && (
                <Text
                  x={25}
                  y={height - 90}
                  text="y"
                  fontSize={14}
                  fill={COLORS.text}
                />
              )}
            </Group>
          )}
          
          {/* Brillouin zone - always visible */}
          {brillouinZoneData && brillouinZoneData.vertices && brillouinZoneData.vertices.length > 0 && (
            <Group>
              {/* Filled Brillouin zone */}
              {(() => {
                // Validate vertices and convert to canvas coordinates
                const canvasPoints = brillouinZoneData.vertices
                  .filter((vertex: [number, number]) => vertex && Array.isArray(vertex) && vertex.length >= 2 && isFinite(vertex[0]) && isFinite(vertex[1]))
                  .flatMap((vertex: [number, number]) => {
                    const [canvasX, canvasY] = latticeToCanvas(vertex[0], vertex[1]);
                    
                    if (!isFinite(canvasX) || !isFinite(canvasY)) {
                      return []; // Filter out invalid coordinates
                    }
                    
                    return [canvasX, canvasY];
                  });
                
                // Only render if we have enough valid points
                if (canvasPoints.length < 6) {
                  return null;
                }
                
                // Validate all coordinates are finite
                const allValid = canvasPoints.every((coord: number) => isFinite(coord));
                if (!allValid) {
                  return null;
                }
                
                return (
                  <Line
                    points={canvasPoints}
                    closed
                    fill={COLORS.brillouinZoneFill}
                    stroke={COLORS.brillouinZone}
                    strokeWidth={2}
                  />
                );
              })()}
            </Group>
          )}
          
          {/* High symmetry points */}
          {highSymmetryPoints.length > 0 && (
            <Group>
              {highSymmetryPoints.map((point: any, index: number) => {
                // Validate point structure
                if (!point || typeof point.x !== 'number' || typeof point.y !== 'number' || !point.label) {
                  return null;
                }
                
                // Validate coordinates
                if (!isFinite(point.x) || !isFinite(point.y)) {
                  return null;
                }
                
                const [canvasX, canvasY] = latticeToCanvas(point.x, point.y);
                
                // Validate canvas coordinates
                if (!isFinite(canvasX) || !isFinite(canvasY)) {
                  return null;
                }
                
                // Only render points that are visible on canvas
                if (canvasX >= -20 && canvasX <= canvasWidth + 20 && 
                    canvasY >= -20 && canvasY <= height + 20) {
                  return (
                    <Group key={`hspoint-${index}`}>
                      {/* High symmetry point circle - smaller and no border */}
                      <Circle
                        x={canvasX}
                        y={canvasY}
                        radius={pointRadius}
                        fill={COLORS.highSymmetryPoint}
                      />
                      {/* Label */}
                      <Text
                        x={canvasX + pointRadius + 10}
                        y={canvasY - 8}
                        text={point.label}
                        fontSize={20}
                        fill={COLORS.highSymmetryLabel}
                        fontStyle="bold"
                      />
                    </Group>
                  );
                }
                return null;
              })}
            </Group>
          )}
          
          {/* High symmetry path */}
          {highSymmetryPath.length > 1 && highSymmetryPoints.length > 0 && (
            <Group>
              {(() => {
                // Create a map of labels to points for quick lookup
                const pointMap = new Map();
                highSymmetryPoints.forEach((point: any) => {
                  if (point && point.label && typeof point.x === 'number' && typeof point.y === 'number') {
                    pointMap.set(point.label, [point.x, point.y]);
                  }
                });
                
                // Generate path lines
                const pathLines = [];
                for (let i = 0; i < highSymmetryPath.length - 1; i++) {
                  const currentLabel = highSymmetryPath[i];
                  const nextLabel = highSymmetryPath[i + 1];
                  
                  const currentPoint = pointMap.get(currentLabel);
                  const nextPoint = pointMap.get(nextLabel);
                  
                  if (currentPoint && nextPoint) {
                    const [x1, y1] = latticeToCanvas(currentPoint[0], currentPoint[1]);
                    const [x2, y2] = latticeToCanvas(nextPoint[0], nextPoint[1]);
                    
                    if (isFinite(x1) && isFinite(y1) && isFinite(x2) && isFinite(y2)) {
                      pathLines.push(
                        <Line
                          key={`path-${i}`}
                          points={[x1, y1, x2, y2]}
                          stroke={COLORS.highSymmetryPath}
                          strokeWidth={3}
                          dash={[8, 4]}
                          opacity={0.7}
                        />
                      );
                    }
                  }
                }
                
                return pathLines;
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
                
                const [canvasX, canvasY] = latticeToCanvas(x, y);
                
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
                      fill={COLORS.latticePoint}
                      stroke={COLORS.vector}
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
                fill={COLORS.text}
              />
            )
          )}
          
          {/* Lattice vectors */}
          {localShowVectors && vectors && (
            <Group>
              {/* b1 vector */}
              <Arrow
                points={[
                  ...latticeToCanvas(0, 0),
                  ...latticeToCanvas(vectors.a1[0], vectors.a1[1])
                ]}
                stroke={COLORS.vector}
                strokeWidth={vectorWidth}
                fill={COLORS.vector}
                pointerLength={10}
                pointerWidth={10}
              />
              {(() => {
                // Calculate midpoint of b1 vector
                const midX = vectors.a1[0] * 0.5;
                const midY = vectors.a1[1] * 0.5;
                
                // Calculate perpendicular vector (rotate 90° counterclockwise)
                const perpX = -vectors.a1[1];
                const perpY = vectors.a1[0];
                
                // Normalize perpendicular vector and scale for offset
                const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                const offsetScale = 1.2; 
                const normalizedPerpX = (perpX / perpLength) * offsetScale;
                const normalizedPerpY = (perpY / perpLength) * offsetScale;
                
                // Calculate label position
                const labelX = midX + normalizedPerpX;
                const labelY = midY + normalizedPerpY;
                
                const [canvasX, canvasY] = latticeToCanvas(labelX, labelY);
                
                const labelText = "b₁";
                
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
                    fill={COLORS.axes}
                    fontStyle="bold"
                  />
                );
              })()}
              
              {/* b2 vector */}
              <Arrow
                points={[
                  ...latticeToCanvas(0, 0),
                  ...latticeToCanvas(vectors.a2[0], vectors.a2[1])
                ]}
                stroke={COLORS.vector}
                strokeWidth={vectorWidth}
                fill={COLORS.vector}
                pointerLength={10}
                pointerWidth={10}
              />
              {(() => {
                // Calculate midpoint of b2 vector
                const midX = vectors.a2[0] * 0.5;
                const midY = vectors.a2[1] * 0.5;
                
                // Calculate perpendicular vector (rotate 90° counterclockwise)
                const perpX = -vectors.a2[1];
                const perpY = vectors.a2[0];
                
                // Normalize perpendicular vector and scale for offset
                const perpLength = Math.sqrt(perpX * perpX + perpY * perpY);
                const offsetScale = 1.2; 
                const normalizedPerpX = (perpX / perpLength) * offsetScale;
                const normalizedPerpY = (perpY / perpLength) * offsetScale;
                
                // Calculate label position
                const labelX = midX + normalizedPerpX;
                const labelY = midY + normalizedPerpY;
                
                const [canvasX, canvasY] = latticeToCanvas(labelX, labelY);
                
                const labelText = "b₂";
                
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
                    fill={COLORS.axes}
                    fontStyle="bold"
                  />
                );
              })()}
            </Group>
          )}
        </Layer>
      </Stage>
      
      {/* Toggle buttons with subtle styling */}
      <div className="flex justify-evenly w-full gap-2 mt-4">
        <div
          onClick={() => setLocalShowGrid(!localShowGrid)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Grid</span>
          {localShowGrid ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-orange-600 dark:text-orange-400" />
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
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-orange-600 dark:text-orange-400" />
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
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-orange-600 dark:text-orange-400" />
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
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-orange-600 dark:text-orange-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
    </div>
  );
}