'use client';

import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Stage, Layer, Circle, Line, Text, Arrow, Shape, Group } from 'react-konva';
import { getWasmModule } from '../providers/wasmLoader';
import type { Lattice2D, Moire2D } from '../../public/wasm/moire_lattice_wasm';
import { Orbit, Search, Square, SquareCheck } from 'lucide-react';
import { useBandCoverageStore } from './band-coverage/store';
import type { LatticeType } from './band-coverage/types';
import { getFallbackBrillouinZoneFractionalVertices, getPathDefinition, getUniqueHighSymmetryPoints, interpolateFractionalCoordinate } from './band-coverage/symmetry';

interface HighSymmetryVisualization2DProps {
  // Option 1: Use predefined lattice types
  latticeType?: 'oblique' | 'rectangular' | 'centered_rectangular' | 'square' | 'hexagonal';
  
  // Option 2: Pass custom basis vectors [a1, a2] where each vector is [x, y]
  basisVectors?: [[number, number], [number, number]];
  
  // Option 3: Pass a custom lattice directly
  customLattice?: Lattice2D | Moire2D;
  
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
  hoveredKPoint?: HighlightPointInput;
  selectedKPoint?: HighlightPointInput;
  syncBandStore?: boolean;
}

// Orange color palette for reciprocal lattice with vintage high symmetry colors
type HighlightPointInput = {
  ratio: number
  color?: string
  label?: string
}

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

const DEFAULT_HOVER_COLOR = '#ffffff';
const DEFAULT_SELECTED_COLOR = '#ffd580';

const mapToBandLattice = (value?: string): LatticeType => {
  if (!value) return 'square';
  if (value === 'hexagonal') return 'hex';
  return value as LatticeType;
};

const fractionalToCartesian = (
  fractional: [number, number],
  vectors: { a1: [number, number]; a2: [number, number] }
): [number, number] => {
  const [u, v] = fractional;
  return [
    u * vectors.a1[0] + v * vectors.a2[0],
    u * vectors.a1[1] + v * vectors.a2[1],
  ];
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
  gridOpacity = 0.2,
  hoveredKPoint,
  selectedKPoint,
  syncBandStore = false,
}: HighSymmetryVisualization2DProps) {
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);
  const [isLatticeReady, setIsLatticeReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lattice, setLattice] = useState<Lattice2D | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(width || 800);
  const storeHovered = useBandCoverageStore((state) => state.bandHovered);
  const storeSelected = useBandCoverageStore((state) => state.bandSelected);
  
  // Local state for toggle buttons
  const [localShowGrid, setLocalShowGrid] = useState(showGrid);
  const [localShowAxes, setLocalShowAxes] = useState(showAxes);
  const [localShowVectors, setLocalShowVectors] = useState(showLatticeVectors);
  const [localShowPoints, setLocalShowPoints] = useState(showPoints);
  const [focusBrillouinZone, setFocusBrillouinZone] = useState(false);
  const [mirrorSymmetry, setMirrorSymmetry] = useState(false);
  const normalizedComponentLattice = mapToBandLattice(latticeType);
  const storeLatticeType = storeSelected?.lattice ?? storeHovered?.lattice;
  const allowStoreHighlight = syncBandStore && (!latticeType || !storeLatticeType || normalizedComponentLattice === storeLatticeType);
  const effectiveHoveredInput = allowStoreHighlight && storeHovered?.kRatio !== undefined
    ? { ratio: storeHovered.kRatio, color: storeHovered.color, label: storeHovered.kLabel }
    : hoveredKPoint;
  const effectiveSelectedInput = allowStoreHighlight && storeSelected?.kRatio !== undefined
    ? { ratio: storeSelected.kRatio, color: storeSelected.color, label: storeSelected.kLabel }
    : selectedKPoint;
  const highlightLatticeType: LatticeType = (allowStoreHighlight && storeLatticeType ? storeLatticeType : normalizedComponentLattice) ?? 'square';
  
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
  
  // Helper function to convert any lattice to Lattice2D
  const convertToLattice2D = (inputLattice: Lattice2D | Moire2D): Lattice2D => {
    // Check if it's a Moire2D (has getLattice1 method)
    if ('getLattice1' in inputLattice && typeof inputLattice.getLattice1 === 'function') {
      // It's a Moire2D, get the first constituent lattice
      return inputLattice.getLattice1();
    } else {
      // It's already a Lattice2D
      return inputLattice as Lattice2D;
    }
  };
  
  // Get reciprocal lattice vectors
  const vectors = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      // Get reciprocal basis matrix (9 elements: column-major)
      const basisFlat = lattice.getReciprocalBasis();
      
      // Extract 2D vectors from the matrix (first two columns)
      const result = {
        a1: [basisFlat[0], basisFlat[1]], // First column
        a2: [basisFlat[3], basisFlat[4]]  // Second column
      };
      
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
      // Note: High symmetry points API not yet implemented in WASM bindings
      // Returning empty array for now
      return [];
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
      // Note: High symmetry path API not yet implemented in WASM bindings
      // Returning empty array for now
      return [];
    } catch (err) {
      console.warn('Failed to get high symmetry path:', err);
      return [];
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);

  const fallbackHighSymmetry = useMemo(() => {
    if (!vectors) {
      return null;
    }

    const uniquePoints = getUniqueHighSymmetryPoints(highlightLatticeType).map((entry) => ({
      label: entry.label,
      fractional: entry.fractional as [number, number],
    }));
    const pathLabels = getPathDefinition(highlightLatticeType).map((stop) => stop.label);
    const points = uniquePoints.map(({ label, fractional }) => {
      const [x, y] = fractionalToCartesian(fractional, {
        a1: vectors.a1 as [number, number],
        a2: vectors.a2 as [number, number],
      });
      return { label, x, y };
    });
    return { points, pathLabels };
  }, [vectors, highlightLatticeType]);

  const derivedHighSymmetryPoints = highSymmetryPoints.length > 0
    ? highSymmetryPoints
    : fallbackHighSymmetry?.points ?? [];
  const derivedHighSymmetryPath = highSymmetryPath.length > 0
    ? highSymmetryPath
    : fallbackHighSymmetry?.pathLabels ?? [];
  
  // Get Brillouin zone data
  const brillouinZoneData = useMemo(() => {
    if (!isWasmLoaded || !isLatticeReady || !lattice) {
      return null;
    }
    
    try {
      // Get Brillouin zone vertices as flat array [x, y, z, x, y, z, ...]
      const verticesFlat = lattice.getBrillouinZoneVertices();
      
      if (!verticesFlat || verticesFlat.length === 0) {
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
      
      const result = {
        vertices: validVertices,
        edges: [], // Edge info not available from new API
        measure: 0  // Measure not available from new API
      };
      
      return result;
    } catch (err) {
      console.warn('Failed to get Brillouin zone data:', err);
      return null;
    }
  }, [lattice, isWasmLoaded, isLatticeReady]);

  // Canvas center and scale for coordinate transformation
  const centerX = canvasWidth / 2;
  const centerY = height / 2;
  
  // Calculate scale based on lattice type, vectors, and shells parameter
  const defaultScale = useMemo(() => {
    const zoomFactor = normalizedComponentLattice === 'square' ? 1.9 : 1.3;
    if (!vectors) {
      return Math.min(canvasWidth, height) / (shells * 1.5 * zoomFactor);
    }
    const maxExtent = Math.max(
      Math.sqrt(vectors.a1[0] ** 2 + vectors.a1[1] ** 2),
      Math.sqrt(vectors.a2[0] ** 2 + vectors.a2[1] ** 2),
      Math.sqrt((vectors.a1[0] + vectors.a2[0]) ** 2 + (vectors.a1[1] + vectors.a2[1]) ** 2)
    );
    return Math.min(canvasWidth, height) / (shells * maxExtent * zoomFactor);
  }, [canvasWidth, height, shells, vectors, normalizedComponentLattice]);

  const effectiveBrillouinVertices = useMemo(() => {
    if (brillouinZoneData && brillouinZoneData.vertices && brillouinZoneData.vertices.length > 0) {
      return brillouinZoneData.vertices
        .filter((vertex: number[]) => Array.isArray(vertex) && vertex.length >= 2 && isFinite(vertex[0]) && isFinite(vertex[1]));
    }
    if (!vectors) return [];
    const fallback = getFallbackBrillouinZoneFractionalVertices(highlightLatticeType);
    return fallback.map((fractional) => fractionalToCartesian(fractional, {
      a1: vectors.a1 as [number, number],
      a2: vectors.a2 as [number, number],
    }));
  }, [brillouinZoneData, vectors, highlightLatticeType]);

  const focusScale = useMemo(() => {
    if (!focusBrillouinZone || !effectiveBrillouinVertices.length) return null;
    const ys = effectiveBrillouinVertices.map((vertex) => vertex[1]).filter((value) => Number.isFinite(value));
    if (!ys.length) return null;
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const span = maxY - minY;
    if (!span) return null;
    const padding = 0.9;
    return (height * padding) / span;
  }, [focusBrillouinZone, effectiveBrillouinVertices, height]);

  const scale = focusScale ?? defaultScale;
  
  // Grid spacing in canvas pixels (should align with lattice unit)
  const gridSpacing = scale;
  
  // Coordinate transformation functions
  const latticeToCanvas = (x: number, y: number): [number, number] => {
    return [
      centerX + x * scale,
      centerY - y * scale // Flip y-axis for standard math coordinates
    ];
  };

  const cartesianToCanvas = (x: number, y: number): [number, number] => {
    return [
      centerX + x * scale,
      centerY - y * scale
    ];
  };

  const getSymmetryOrder = (type: LatticeType): number => {
    switch (type) {
      case 'hex':
      case 'hexagonal':
        return 6;
      case 'square':
        return 4;
      default:
        return 1;
    }
  };

  const rotatePoint = (x: number, y: number, angle: number): [number, number] => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [x * cos - y * sin, x * sin + y * cos];
  };

    // Initialize WASM and create lattice with proper async handling
  useEffect(() => {
    let mounted = true;
    let currentLattice: Lattice2D | null = null;
    
    async function initializeAndCreateLattice() {
      try {
        // Reset states
        setError(null);
        setIsLatticeReady(false);
        
        // If we have a custom lattice, use it directly without WASM initialization
        if (customLattice) {
          try {
            const convertedLattice = convertToLattice2D(customLattice);
            if (mounted) {
              setLattice(convertedLattice);
              setIsWasmLoaded(true);
              setIsLatticeReady(true);
            }
            return;
          } catch (err: any) {
            if (mounted) {
              setError(err.message);
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
        
        // Create lattice based on type or vectors
        if (basisVectors) {
          // Create lattice from custom basis vectors
          // Convert to column-major 3x3 matrix [a1x, a1y, 0, a2x, a2y, 0, 0, 0, 1]
          const directMatrix = new Float64Array([
            basisVectors[0][0], basisVectors[0][1], 0,
            basisVectors[1][0], basisVectors[1][1], 0,
            0, 0, 1
          ]);
          currentLattice = new wasm.Lattice2D(directMatrix);
        } else if (latticeType) {
          // Create predefined lattice type using helper functions
          switch (latticeType) {
            case 'square':
              currentLattice = wasm.create_square_lattice(a);
              break;
            case 'rectangular':
              currentLattice = wasm.create_rectangular_lattice(a, b);
              break;
            case 'hexagonal':
              currentLattice = wasm.create_hexagonal_lattice(a);
              break;
            case 'oblique':
              // Convert gamma from degrees to radians
              currentLattice = wasm.create_oblique_lattice(a, b, gamma * Math.PI / 180);
              break;
            case 'centered_rectangular':
              // For centered rectangular, use rectangular with special parameters
              currentLattice = wasm.create_rectangular_lattice(a, b);
              break;
            default:
              currentLattice = wasm.create_square_lattice(1);
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
      // When focusing on the Brillouin zone we still want to request the
      // broader lattice extent, so derive sampling width/height from the
      // unfocused (default) scale. This keeps side/top points available when
      // zoomed in tightly.
      const samplingScale = focusScale ? defaultScale : scale;

      // Calculate the lattice coordinate bounds based on canvas size and scale
      // Request a larger rectangle to ensure coverage in all directions
      // We need ~1.5x to 2x to ensure full coverage after centering
      const latticeWidth = (canvasWidth / samplingScale) * 1.5;
      const latticeHeight = (height / samplingScale) * 1.5;
      
      // Validate the input parameters
      if (!isFinite(latticeWidth) || !isFinite(latticeHeight) || latticeWidth <= 0 || latticeHeight <= 0) {
        return [];
      }
      
      // Get reciprocal lattice points as flat array [x, y, z, x, y, z, ...]
      const pointsFlat = lattice.getReciprocalLatticePoints(latticeWidth, latticeHeight);
      
      // Convert flat array to [x, y] pairs, ignoring z-component
      const result = [];
      for (let i = 0; i < pointsFlat.length; i += 3) {
        result.push([pointsFlat[i], pointsFlat[i + 1]]);
      }
      
      return result;
    } catch (err) {
      console.warn('Failed to generate lattice points:', err);
      return [];
    }
  }, [lattice, canvasWidth, height, scale, focusScale, defaultScale, isWasmLoaded, isLatticeReady]);

  // Compute lattice bounding-box center, adjusted to ensure origin point alignment
  const latticeCenter = useMemo(() => {
    if (!latticePoints || latticePoints.length === 0 || !vectors) {
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

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Round to nearest lattice point to ensure a point sits at canvas center
    // Find the lattice point closest to the computed center
    const b1 = vectors.a1;  // reciprocal basis vectors
    const b2 = vectors.a2;
    
    // Compute fractional coordinates of center in reciprocal basis
    const det = b1[0] * b2[1] - b1[1] * b2[0];
    if (Math.abs(det) < 1e-10) {
      return [centerX, centerY];
    }
    
    const frac_u = (centerX * b2[1] - centerY * b2[0]) / det;
    const frac_v = (b1[0] * centerY - b1[1] * centerX) / det;
    
    // Round to nearest integer lattice coordinates
    const u = Math.round(frac_u);
    const v = Math.round(frac_v);
    
    // Convert back to Cartesian coordinates - this is the nearest lattice point
    return [u * b1[0] + v * b2[0], u * b1[1] + v * b2[1]];
  }, [latticePoints, vectors]);

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
  }, [canvasWidth, height, gridSpacing, centerX, centerY, scale, COLORS.grid]);

  const highlightMarkers = useMemo(() => {
    if (!vectors) return [];
    const markers: Array<{ key: string; x: number; y: number; color: string; kind: 'hover' | 'selected' }> = [];

    const symmetryOrder = Math.max(1, mirrorSymmetry ? getSymmetryOrder(highlightLatticeType) : 1);

    const addMarker = (input: HighlightPointInput | undefined, kind: 'hover' | 'selected') => {
      if (!input || typeof input.ratio !== 'number') return;
      const interpolated = interpolateFractionalCoordinate(highlightLatticeType, input.ratio);
      if (!interpolated) return;
      const [cartX, cartY] = fractionalToCartesian(interpolated.fractional, {
        a1: vectors.a1 as [number, number],
        a2: vectors.a2 as [number, number],
      });
      const baseColor = input.color ?? (kind === 'selected' ? DEFAULT_SELECTED_COLOR : DEFAULT_HOVER_COLOR);

      for (let i = 0; i < symmetryOrder; i++) {
        const angle = (2 * Math.PI * i) / symmetryOrder;
        const [rx, ry] = symmetryOrder === 1 ? [cartX, cartY] : rotatePoint(cartX, cartY, angle);
        const [canvasX, canvasY] = cartesianToCanvas(rx, ry);
        if (!isFinite(canvasX) || !isFinite(canvasY)) continue;
        markers.push({
          key: `marker-${kind}-${i}`,
          x: canvasX,
          y: canvasY,
          color: baseColor,
          kind,
        });
      }
    };

    addMarker(effectiveSelectedInput, 'selected');
    addMarker(effectiveHoveredInput, 'hover');
    return markers;
  }, [vectors, effectiveHoveredInput, effectiveSelectedInput, highlightLatticeType, mirrorSymmetry, centerX, centerY, scale]);

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
          {effectiveBrillouinVertices.length > 0 && (
            <Group>
              {/* Filled Brillouin zone */}
              {(() => {
                // Validate vertices and convert to canvas coordinates
                const canvasPoints = effectiveBrillouinVertices
                  .filter((vertex: number[]) => vertex && Array.isArray(vertex) && vertex.length >= 2 && isFinite(vertex[0]) && isFinite(vertex[1]))
                  .flatMap((vertex: number[]) => {
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
          {derivedHighSymmetryPoints.length > 0 && (
            <Group>
              {derivedHighSymmetryPoints.map((point: any, index: number) => {
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
          {derivedHighSymmetryPath.length > 1 && derivedHighSymmetryPoints.length > 0 && (
            <Group>
              {(() => {
                // Create a map of labels to points for quick lookup
                const pointMap = new Map();
                derivedHighSymmetryPoints.forEach((point: any) => {
                  if (point && point.label && typeof point.x === 'number' && typeof point.y === 'number') {
                    pointMap.set(point.label, [point.x, point.y]);
                  }
                });
                
                // Generate path lines
                const pathLines = [];
                for (let i = 0; i < derivedHighSymmetryPath.length - 1; i++) {
                  const currentLabel = derivedHighSymmetryPath[i];
                  const nextLabel = derivedHighSymmetryPath[i + 1];
                  
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

          {highlightMarkers.length > 0 && (
            <Group>
              {highlightMarkers.map((marker) => (
                <Group key={marker.key}>
                  <Circle
                    x={marker.x}
                    y={marker.y}
                    radius={8}
                    stroke={marker.color}
                    strokeWidth={marker.kind === 'selected' ? 2.5 : 1.5}
                    fill="rgba(0,0,0,0.35)"
                  />
                  <Circle
                    x={marker.x}
                    y={marker.y}
                    radius={4}
                    fill={marker.color}
                    stroke="#000000"
                    strokeWidth={0.5}
                  />
                </Group>
              ))}
            </Group>
          )}
          
          {/* Lattice points */}
          {localShowPoints && latticePoints.length > 0 ? (
            latticePoints
              .map((point: number[] | [number, number], index: number) => {
                // Validate point structure
                if (!point || !Array.isArray(point) || point.length < 2) {
                  return null;
                }
                
                const [x, y] = [point[0], point[1]];
                
                // Validate coordinates
                if (!isFinite(x) || !isFinite(y)) {
                  return null;
                }
                
                // Apply centering offset for reciprocal lattice points
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
      <div className="flex flex-wrap justify-evenly w-full gap-2 mt-4">
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
        
        <button
          type="button"
          onClick={() => setMirrorSymmetry((value) => !value)}
          className="flex items-center justify-center border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm transition-colors duration-200 flex-none cursor-pointer"
          style={{
            backgroundColor: mirrorSymmetry ? 'rgba(59,130,246,0.12)' : 'transparent',
            color: mirrorSymmetry ? '#2563eb' : '#6b7280',
          }}
          aria-pressed={mirrorSymmetry}
        >
          <Orbit className="w-4 h-4" />
        </button>

        <button
          type="button"
          onClick={() => setFocusBrillouinZone((value) => !value)}
          className="flex items-center justify-center border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm transition-colors duration-200 flex-none cursor-pointer"
          style={{
            backgroundColor: focusBrillouinZone ? 'rgba(255,140,0,0.12)' : 'transparent',
            color: focusBrillouinZone ? '#c05621' : '#6b7280',
          }}
          aria-pressed={focusBrillouinZone}
        >
          <Search className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}