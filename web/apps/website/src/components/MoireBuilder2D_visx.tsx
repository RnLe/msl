// This file is an attempt to increase the performance of the moire lattice visualization.
// As it turns out, the konva canvas is already more performant than a visx implementation.
// This file is kept for any future references or usage.
'use client';

import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { getWasmModule } from '../providers/wasmLoader';
import type { WasmMoire2D, WasmLattice2D } from '../../public/wasm/moire_lattice_wasm';
import { Square, SquareCheck, ChevronDown, Zap, ZapOff } from 'lucide-react';
import { Group } from '@visx/group';
import { Circle, Line, Bar, Polygon } from '@visx/shape';
import { Text } from '@visx/text';
import { GridRows, GridColumns } from '@visx/grid';
import { scaleLinear } from '@visx/scale';

interface MoireBuilder2DProps {
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

// Vintage color palette
const COLORS = {
  lattice1: '#4682B4', // SteelBlue
  lattice2: '#FF8C00', // DarkOrange
  moireLattice: '#8B008B', // DarkMagenta
  vector: '#191970', // MidnightBlue
  unitCell: '#6495ED', // CornflowerBlue
  grid: '#B0C4DE', // LightSteelBlue
  axes: '#696969', // DimGray
  text: '#2F4F4F', // DarkSlateGray
  background: 'transparent',
  reciprocalLattice1: '#00CED1', // DarkTurquoise
  reciprocalLattice2: '#FF6347', // Tomato
  reciprocalMoire: '#9370DB', // MediumPurple
};

// Available lattice types from the Rust code
const LATTICE_TYPES = [
  { value: 'square', label: 'Square', params: ['a'] },
  { value: 'hexagonal', label: 'Hexagonal', params: ['a'] },
  { value: 'rectangular', label: 'Rectangular', params: ['a', 'b'] },
  { value: 'centered_rectangular', label: 'Centered Rectangular', params: ['a', 'b'] },
  { value: 'oblique', label: 'Oblique', params: ['a', 'b', 'gamma'] },
];

// Available transformation types
const TRANSFORMATION_TYPES = [
  { value: 'rotation', label: 'Rotation & Scale', params: ['angle', 'scale'] },
  { value: 'anisotropic', label: 'Anisotropic Scale', params: ['scale_x', 'scale_y'] },
  { value: 'shear', label: 'Shear', params: ['shear_x', 'shear_y'] },
  { value: 'general', label: 'General Matrix', params: ['m00', 'm01', 'm10', 'm11'] },
];

// Custom Arrow component for axes
const Arrow = ({ x1, y1, x2, y2, stroke, strokeWidth, fill }: any) => {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const arrowLength = 8;
  const arrowAngle = Math.PI / 6;
  
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={stroke} strokeWidth={strokeWidth} />
      <polygon
        points={`
          ${x2},${y2}
          ${x2 - arrowLength * Math.cos(angle - arrowAngle)},${y2 - arrowLength * Math.sin(angle - arrowAngle)}
          ${x2 - arrowLength * Math.cos(angle + arrowAngle)},${y2 - arrowLength * Math.sin(angle + arrowAngle)}
        `}
        fill={fill || stroke}
      />
    </g>
  );
};

// Performance thresholds
const PERFORMANCE_THRESHOLDS = {
  HIGH_QUALITY: 500,    // Below this, use full quality
  MEDIUM_QUALITY: 1500, // Below this, use medium quality
  LOW_QUALITY: 3000,    // Below this, use low quality
  CLUSTERING: 5000,     // Above this, use clustering
};

// Clustering function to reduce point count
function clusterPoints(points: any[], clusterSize: number = 10): any[] {
  if (points.length < PERFORMANCE_THRESHOLDS.CLUSTERING) return points;
  
  const clusters = new Map<string, { x: number, y: number, count: number }>();
  
  points.forEach(point => {
    const clusterX = Math.floor(point.x / clusterSize) * clusterSize;
    const clusterY = Math.floor(point.y / clusterSize) * clusterSize;
    const key = `${clusterX},${clusterY}`;
    
    if (clusters.has(key)) {
      const cluster = clusters.get(key)!;
      cluster.x = (cluster.x * cluster.count + point.x) / (cluster.count + 1);
      cluster.y = (cluster.y * cluster.count + point.y) / (cluster.count + 1);
      cluster.count++;
    } else {
      clusters.set(key, { x: point.x, y: point.y, count: 1 });
    }
  });
  
  return Array.from(clusters.values());
}

export function MoireBuilder2D_visx({ 
  width,
  height = 600,
  showUnitCell = true,
  showLatticeVectors = true,
  showGrid = true,
  showAxes = true,
  showPoints = true,
  showLattice1 = true,
  showLattice2 = true,
  showMoireLattice = true,
  is_debug = false,
  pointRadius = 3,
  vectorWidth = 2,
  gridOpacity = 0.2
}: MoireBuilder2DProps) {
  const [isWasmLoaded, setIsWasmLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(width || 800);
  
  // Lattice and transformation settings
  const [latticeType, setLatticeType] = useState('hexagonal');
  const [transformationType, setTransformationType] = useState('rotation');
  const [isReciprocal, setIsReciprocal] = useState(false);
  
  // Lattice parameters
  const [latticeParams, setLatticeParams] = useState({
    a: 1,
    b: 1.2,
    gamma: 60, // degrees
  });
  
  // Transformation parameters
  const [transformParams, setTransformParams] = useState({
    angle: 5, // degrees
    scale: 1,
    scale_x: 1.1,
    scale_y: 0.9,
    shear_x: 0.2,
    shear_y: 0,
    m00: 1, m01: 0, m10: 0, m11: 1, // Identity matrix
  });
  
  // Toggle states
  const [localShowGrid, setLocalShowGrid] = useState(showGrid);
  const [localShowAxes, setLocalShowAxes] = useState(showAxes);
  const [localShowVectors, setLocalShowVectors] = useState(showLatticeVectors);
  const [localShowUnitCell, setLocalShowUnitCell] = useState(showUnitCell);
  const [localShowPoints, setLocalShowPoints] = useState(showPoints);
  const [localShowLattice1, setLocalShowLattice1] = useState(showLattice1);
  const [localShowLattice2, setLocalShowLattice2] = useState(showLattice2);
  const [localShowMoireLattice, setLocalShowMoireLattice] = useState(showMoireLattice);
  
  // Dropdown states
  const [showLatticeDropdown, setShowLatticeDropdown] = useState(false);
  const [showTransformDropdown, setShowTransformDropdown] = useState(false);
  
  // Moire lattice state
  const [moireLattice, setMoireLattice] = useState<WasmMoire2D | null>(null);
  const [baseLattice, setBaseLattice] = useState<WasmLattice2D | null>(null);
  const [moireError, setMoireError] = useState<string | null>(null);
  
  // Performance mode state
  const [isLightweightMode, setIsLightweightMode] = useState(false);
  const [autoQuality, setAutoQuality] = useState(true);
  const [pointCount, setPointCount] = useState(0);
  
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
  const centerX = canvasWidth / 2;
  const centerY = height / 2;
  
  // Initialize WASM
  useEffect(() => {
    async function loadWasm() {
      try {
        await getWasmModule();
        setIsWasmLoaded(true);
      } catch (err: any) {
        setError(err.message);
      }
    }
    
    loadWasm();
  }, []);
  
  // Create base lattice and moiré lattice
  useEffect(() => {
    if (!isWasmLoaded) return;
    
    async function createLattices() {
      try {
        const wasm = await getWasmModule();
        
        // Create base lattice
        let lattice: WasmLattice2D;
        switch (latticeType) {
          case 'square':
            lattice = wasm.create_square_lattice(latticeParams.a);
            break;
          case 'hexagonal':
            lattice = wasm.create_hexagonal_lattice(latticeParams.a);
            break;
          case 'rectangular':
            lattice = wasm.create_rectangular_lattice(latticeParams.a, latticeParams.b);
            break;
          case 'centered_rectangular':
            lattice = wasm.create_centered_rectangular_lattice(latticeParams.a, latticeParams.b);
            break;
          case 'oblique':
            lattice = wasm.create_oblique_lattice(latticeParams.a, latticeParams.b, latticeParams.gamma);
            break;
          default:
            lattice = wasm.create_square_lattice(1);
        }
        
        setBaseLattice(lattice);
        
        // Clear any previous moiré error
        setMoireError(null);
        
        // Create moiré lattice with transformation
        const moireParams = {
          transformation_type: transformationType,
          angle_degrees: transformParams.angle,
          scale: transformParams.scale,
          scale_x: transformParams.scale_x,
          scale_y: transformParams.scale_y,
          shear_x: transformParams.shear_x,
          shear_y: transformParams.shear_y,
          matrix: transformationType === 'general' ? [
            transformParams.m00, transformParams.m01,
            transformParams.m10, transformParams.m11
          ] : undefined,
        };
        
        try {
          const moire = wasm.WasmMoireBuilder.build_with_params(lattice, moireParams);
          setMoireLattice(moire);
        } catch (moireErr: any) {
          // Handle specific moiré creation errors (like "lattices too similar")
          const errorMessage = moireErr.message || moireErr.toString();
          if (errorMessage.includes('too similar') || errorMessage.includes('moiré')) {
            setMoireError(errorMessage);
            setMoireLattice(null);
          } else {
            // Re-throw other errors
            throw moireErr;
          }
        }
        
      } catch (err: any) {
        setError(err.message || err.toString());
        console.error('Error creating lattices:', err);
      }
    }
    
    createLattices();
  }, [isWasmLoaded, latticeType, latticeParams, transformationType, transformParams]);
  
  // Calculate scale based on moiré lattice
  const scale = useMemo(() => {
    if (!moireLattice) {
      return Math.min(canvasWidth, height) / 10;
    }
    
    try {
      // Get moiré period ratio to adjust scale
      const periodRatio = moireLattice.moire_period_ratio();
      const baseScale = Math.min(canvasWidth, height) / 10;
      
      // Adjust scale based on period ratio, but keep it reasonable
      return baseScale / Math.max(1, Math.min(periodRatio, 5));
    } catch {
      return Math.min(canvasWidth, height) / 10;
    }
  }, [canvasWidth, height, moireLattice]);
  
  // Create scales for coordinate transformation
  const xScale = useMemo(
    () => scaleLinear({
      domain: [-(canvasWidth / scale) / 2, (canvasWidth / scale) / 2],
      range: [0, canvasWidth],
    }),
    [canvasWidth, scale]
  );
  
  const yScale = useMemo(
    () => scaleLinear({
      domain: [(height / scale) / 2, -(height / scale) / 2],
      range: [0, height],
    }),
    [height, scale]
  );
  
  // Get lattice points for visualization with clustering
  const latticePointsData = useMemo(() => {
    if (!baseLattice) return { lattice1: [], lattice2: [], moire: [] };
    
    try {
      const radius = Math.max(canvasWidth, height) / scale / 2;
      
      // Always show the base lattice (lattice 1)
      let lattice1Points = isReciprocal 
        ? baseLattice.get_reciprocal_lattice_points_in_rectangle(canvasWidth / scale, height / scale)
        : baseLattice.get_direct_lattice_points_in_rectangle(canvasWidth / scale, height / scale);
      
      // Only show lattice 2 and moiré if moireLattice exists (no error)
      let lattice2Points = [];
      let moirePoints = [];
      
      if (moireLattice) {
        lattice2Points = isReciprocal
          ? moireLattice.lattice_2().get_reciprocal_lattice_points_in_rectangle(canvasWidth / scale, height / scale)
          : moireLattice.lattice_2().get_direct_lattice_points_in_rectangle(canvasWidth / scale, height / scale);
          
        moirePoints = isReciprocal
          ? moireLattice.as_lattice2d().get_reciprocal_lattice_points_in_rectangle(canvasWidth / scale, height / scale)
          : moireLattice.as_lattice2d().get_direct_lattice_points_in_rectangle(canvasWidth / scale, height / scale);
      }
      
      // Convert to arrays if needed
      lattice1Points = Array.isArray(lattice1Points) ? lattice1Points : [];
      lattice2Points = Array.isArray(lattice2Points) ? lattice2Points : [];
      moirePoints = Array.isArray(moirePoints) ? moirePoints : [];
      
      // Calculate total point count
      const totalPoints = 
        (localShowLattice1 ? lattice1Points.length : 0) +
        (localShowLattice2 ? lattice2Points.length : 0) +
        (localShowMoireLattice ? moirePoints.length : 0);
      
      setPointCount(totalPoints);
      
      // Apply clustering if needed
      if (autoQuality && totalPoints > PERFORMANCE_THRESHOLDS.CLUSTERING) {
        const clusterSize = scale / 20; // Adaptive cluster size based on zoom
        return {
          lattice1: clusterPoints(lattice1Points, clusterSize),
          lattice2: clusterPoints(lattice2Points, clusterSize),
          moire: clusterPoints(moirePoints, clusterSize * 1.5),
        };
      }
      
      return {
        lattice1: lattice1Points,
        lattice2: lattice2Points,
        moire: moirePoints
      };
    } catch (err) {
      console.error('Error generating lattice points:', err);
      return { lattice1: [], lattice2: [], moire: [] };
    }
  }, [moireLattice, baseLattice, canvasWidth, height, scale, isReciprocal, 
      localShowLattice1, localShowLattice2, localShowMoireLattice, autoQuality]);
  
  // Determine render quality based on point count
  const renderQuality = useMemo(() => {
    if (!autoQuality || isLightweightMode) return 'lightweight';
    
    if (pointCount < PERFORMANCE_THRESHOLDS.HIGH_QUALITY) return 'high';
    if (pointCount < PERFORMANCE_THRESHOLDS.MEDIUM_QUALITY) return 'medium';
    if (pointCount < PERFORMANCE_THRESHOLDS.LOW_QUALITY) return 'low';
    return 'clustered';
  }, [pointCount, autoQuality, isLightweightMode]);
  
  // Get appropriate point radius based on quality
  const getPointRadius = useCallback((baseRadius: number) => {
    if (renderQuality === 'lightweight') return baseRadius * 0.8;
    if (renderQuality === 'medium') return baseRadius * 0.9;
    if (renderQuality === 'low') return baseRadius * 0.7;
    if (renderQuality === 'clustered') return baseRadius * 1.2;
    return baseRadius;
  }, [renderQuality]);
  
  // Render point based on quality mode
  const renderPoint = useCallback((
    key: string,
    x: number,
    y: number,
    radius: number,
    fill: string,
    opacity: number,
    stroke?: string,
    strokeWidth?: number
  ) => {
    const effectiveRadius = getPointRadius(radius);
    
    if (isLightweightMode || renderQuality === 'lightweight') {
      // Render as square for better performance
      return (
        <Bar
          key={key}
          x={x - effectiveRadius}
          y={y - effectiveRadius}
          width={effectiveRadius * 2}
          height={effectiveRadius * 2}
          fill={fill}
          opacity={opacity}
          stroke={stroke}
          strokeWidth={strokeWidth}
        />
      );
    }
    
    // Render as circle
    return (
      <Circle
        key={key}
        cx={x}
        cy={y}
        r={effectiveRadius}
        fill={fill}
        opacity={opacity}
        stroke={stroke}
        strokeWidth={strokeWidth}
      />
    );
  }, [renderQuality, isLightweightMode, getPointRadius]);
  
  // Get current parameter based on lattice type
  const getCurrentLatticeParams = () => {
    const type = LATTICE_TYPES.find(t => t.value === latticeType);
    return type?.params || [];
  };
  
  // Get current transformation parameters
  const getCurrentTransformParams = () => {
    const type = TRANSFORMATION_TYPES.find(t => t.value === transformationType);
    return type?.params || [];
  };
  
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

  return (
    <div ref={containerRef} className="moire-builder-2d w-full flex flex-col items-center">
      <svg width={canvasWidth} height={height}>
        {/* Background grid */}
        {localShowGrid && (
          <Group opacity={gridOpacity}>
            <GridRows
              scale={yScale}
              width={canvasWidth}
              strokeDasharray="2,2"
              stroke={COLORS.grid}
              strokeWidth={1}
            />
            <GridColumns
              scale={xScale}
              height={height}
              strokeDasharray="2,2"
              stroke={COLORS.grid}
              strokeWidth={1}
            />
          </Group>
        )}
        
        {/* Coordinate axes */}
        {localShowAxes && (
          <Group>
            <Arrow
              x1={20}
              y1={height - 20}
              x2={80}
              y2={height - 20}
              stroke={COLORS.axes}
              strokeWidth={2}
              fill={COLORS.axes}
            />
            <Text
              x={85}
              y={height - 15}
              fontSize={14}
              fill={COLORS.text}
            >
              x
            </Text>
            
            <Arrow
              x1={20}
              y1={height - 20}
              x2={20}
              y2={height - 80}
              stroke={COLORS.axes}
              strokeWidth={2}
              fill={COLORS.axes}
            />
            <Text
              x={25}
              y={height - 85}
              fontSize={14}
              fill={COLORS.text}
            >
              y
            </Text>
          </Group>
        )}
        
        {/* Lattice 1 points */}
        {localShowPoints && localShowLattice1 && (
          <Group>
            {latticePointsData.lattice1.map((point: any, index: number) => 
              renderPoint(
                `l1-${index}`,
                xScale(point.x || 0),
                yScale(point.y || 0),
                pointRadius,
                isReciprocal ? COLORS.reciprocalLattice1 : COLORS.lattice1,
                0.7
              )
            )}
          </Group>
        )}
        
        {/* Lattice 2 points */}
        {localShowPoints && localShowLattice2 && (
          <Group>
            {latticePointsData.lattice2.map((point: any, index: number) =>
              renderPoint(
                `l2-${index}`,
                xScale(point.x || 0),
                yScale(point.y || 0),
                pointRadius,
                isReciprocal ? COLORS.reciprocalLattice2 : COLORS.lattice2,
                0.7
              )
            )}
          </Group>
        )}
        
        {/* Moiré lattice points */}
        {localShowPoints && localShowMoireLattice && (
          <Group>
            {latticePointsData.moire.map((point: any, index: number) =>
              renderPoint(
                `moire-${index}`,
                xScale(point.x || 0),
                yScale(point.y || 0),
                pointRadius * 1.5,
                isReciprocal ? COLORS.reciprocalMoire : COLORS.moireLattice,
                renderQuality === 'clustered' ? 0.5 : 1,
                COLORS.vector,
                1
              )
            )}
          </Group>
        )}
        
        {/* Performance indicator */}
        {autoQuality && (
          <Group>
            <Bar
              x={canvasWidth - 120}
              y={10}
              width={110}
              height={25}
              fill="white"
              fillOpacity={0.8}
              stroke={renderQuality === 'high' ? '#4CAF50' : 
                     renderQuality === 'medium' ? '#FF9800' :
                     renderQuality === 'low' ? '#FF5722' : '#F44336'}
              strokeWidth={2}
              rx={4}
            />
            <Text
              x={canvasWidth - 65}
              y={26}
              fontSize={11}
              textAnchor="middle"
              fill={COLORS.text}
            >
              {renderQuality === 'high' ? 'High Quality' :
               renderQuality === 'medium' ? 'Medium' :
               renderQuality === 'low' ? 'Low Quality' :
               renderQuality === 'clustered' ? 'Clustered' :
               'Lightweight'}
            </Text>
          </Group>
        )}
        
        {/* Moiré info text */}
        {moireLattice && (
          <Group>
            <Text x={10} y={20} fontSize={14} fill={COLORS.text}>
              {`Moiré Period Ratio: ${moireLattice.moire_period_ratio().toFixed(3)}`}
            </Text>
            <Text x={10} y={40} fontSize={14} fill={COLORS.text}>
              {`Twist Angle: ${moireLattice.twist_angle_degrees().toFixed(2)}°`}
            </Text>
            {moireLattice.is_commensurate() && (
              <Text x={10} y={60} fontSize={14} fill={COLORS.moireLattice} fontWeight="bold">
                Commensurate
              </Text>
            )}
          </Group>
        )}
        
        {/* Moiré error message */}
        {moireError && (
          <Group>
            <Text
              x={canvasWidth / 2}
              y={height / 2 - 20}
              fontSize={16}
              fill="#FF6B6B"
              fontWeight="bold"
              textAnchor="middle"
            >
              ⚠️ Cannot create moiré pattern
            </Text>
            <Text
              x={canvasWidth / 2}
              y={height / 2 + 5}
              fontSize={12}
              fill="#FF6B6B"
              textAnchor="middle"
            >
              {moireError.includes('too similar') ? 
                'Lattices are too similar - try adjusting parameters' : 
                moireError}
            </Text>
            <Text
              x={canvasWidth / 2}
              y={height / 2 + 25}
              fontSize={11}
              fill="#999"
              textAnchor="middle"
            >
              Showing base lattice only
            </Text>
          </Group>
        )}
      </svg>
      
      {/* Toggle buttons */}
      <div className="flex justify-evenly w-full gap-2 mt-4">
        <div
          onClick={() => setLocalShowGrid(!localShowGrid)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Grid</span>
          {localShowGrid ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
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
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowLattice1(!localShowLattice1)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Lattice 1</span>
          {localShowLattice1 ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowLattice2(!localShowLattice2)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Lattice 2</span>
          {localShowLattice2 ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setLocalShowMoireLattice(!localShowMoireLattice)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Moiré</span>
          {localShowMoireLattice ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-purple-600 dark:text-purple-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
      
      {/* Performance controls */}
      <div className="flex justify-center w-full gap-2 mt-2">
        <button
          onClick={() => setIsReciprocal(!isReciprocal)}
          className="px-4 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 transition-all duration-200"
        >
          <span className="text-sm font-medium">
            {isReciprocal ? 'Switch to Direct Space' : 'Switch to Reciprocal Space'}
          </span>
        </button>
        
        <button
          onClick={() => setIsLightweightMode(!isLightweightMode)}
          className={`px-4 py-2 border transition-all duration-200 flex items-center gap-2 ${
            isLightweightMode 
              ? 'border-green-500 dark:border-green-400 bg-green-50 dark:bg-green-900/20' 
              : 'border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          {isLightweightMode ? (
            <Zap className="w-4 h-4 text-green-600 dark:text-green-400" />
          ) : (
            <ZapOff className="w-4 h-4 text-gray-400 dark:text-gray-500" />
          )}
          <span className="text-sm font-medium">
            {isLightweightMode ? 'Lightweight Mode' : 'Full Quality'}
          </span>
        </button>
        
        <button
          onClick={() => setAutoQuality(!autoQuality)}
          className={`px-4 py-2 border transition-all duration-200 ${
            autoQuality 
              ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20' 
              : 'border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          <span className="text-sm font-medium">
            {autoQuality ? 'Auto Quality: ON' : 'Auto Quality: OFF'}
          </span>
        </button>
      </div>
      
      {/* Control panel */}
      <div className="w-full mt-6 p-4 border border-gray-200 dark:border-gray-700 rounded-lg space-y-4">
        {/* Lattice type dropdown */}
        <div className="relative">
          <label className="block text-sm font-medium mb-2">Base Lattice Type</label>
          <div
            onClick={() => setShowLatticeDropdown(!showLatticeDropdown)}
            className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200"
          >
            <span className="text-sm">
              {LATTICE_TYPES.find(t => t.value === latticeType)?.label || 'Select Lattice'}
            </span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showLatticeDropdown ? 'rotate-180' : ''}`} />
          </div>
          {showLatticeDropdown && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded shadow-lg">
              {LATTICE_TYPES.map(type => (
                <div
                  key={type.value}
                  onClick={() => {
                    setLatticeType(type.value);
                    setShowLatticeDropdown(false);
                  }}
                  className="px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer"
                >
                  <span className="text-sm">{type.label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Lattice parameters */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Lattice Parameters</label>
          {getCurrentLatticeParams().includes('a') && (
            <div>
              <label className="text-xs text-gray-500">a: {latticeParams.a.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={latticeParams.a}
                onChange={(e) => setLatticeParams(prev => ({ ...prev, a: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentLatticeParams().includes('b') && (
            <div>
              <label className="text-xs text-gray-500">b: {latticeParams.b.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={latticeParams.b}
                onChange={(e) => setLatticeParams(prev => ({ ...prev, b: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentLatticeParams().includes('gamma') && (
            <div>
              <label className="text-xs text-gray-500">γ: {latticeParams.gamma.toFixed(0)}°</label>
              <input
                type="range"
                min="30"
                max="150"
                step="5"
                value={latticeParams.gamma}
                onChange={(e) => setLatticeParams(prev => ({ ...prev, gamma: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
        </div>
        
        {/* Transformation type dropdown */}
        <div className="relative">
          <label className="block text-sm font-medium mb-2">Transformation Type</label>
          <div
            onClick={() => setShowTransformDropdown(!showTransformDropdown)}
            className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200"
          >
            <span className="text-sm">
              {TRANSFORMATION_TYPES.find(t => t.value === transformationType)?.label || 'Select Transform'}
            </span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showTransformDropdown ? 'rotate-180' : ''}`} />
          </div>
          {showTransformDropdown && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded shadow-lg">
              {TRANSFORMATION_TYPES.map(type => (
                <div
                  key={type.value}
                  onClick={() => {
                    setTransformationType(type.value);
                    setShowTransformDropdown(false);
                  }}
                  className="px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer"
                >
                  <span className="text-sm">{type.label}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Transformation parameters */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Transformation Parameters</label>
          {getCurrentTransformParams().includes('angle') && (
            <div>
              <label className="text-xs text-gray-500">Angle: {transformParams.angle.toFixed(1)}°</label>
              <input
                type="range"
                min="1"
                max="30"
                step="0.5"
                value={transformParams.angle}
                onChange={(e) => setTransformParams(prev => ({ ...prev, angle: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentTransformParams().includes('scale') && (
            <div>
              <label className="text-xs text-gray-500">Scale: {transformParams.scale.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.05"
                value={transformParams.scale}
                onChange={(e) => setTransformParams(prev => ({ ...prev, scale: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentTransformParams().includes('scale_x') && (
            <div>
              <label className="text-xs text-gray-500">Scale X: {transformParams.scale_x.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.05"
                value={transformParams.scale_x}
                onChange={(e) => setTransformParams(prev => ({ ...prev, scale_x: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentTransformParams().includes('scale_y') && (
            <div>
              <label className="text-xs text-gray-500">Scale Y: {transformParams.scale_y.toFixed(2)}</label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.05"
                value={transformParams.scale_y}
                onChange={(e) => setTransformParams(prev => ({ ...prev, scale_y: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentTransformParams().includes('shear_x') && (
            <div>
              <label className="text-xs text-gray-500">Shear X: {transformParams.shear_x.toFixed(2)}</label>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.05"
                value={transformParams.shear_x}
                onChange={(e) => setTransformParams(prev => ({ ...prev, shear_x: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {getCurrentTransformParams().includes('shear_y') && (
            <div>
              <label className="text-xs text-gray-500">Shear Y: {transformParams.shear_y.toFixed(2)}</label>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.05"
                value={transformParams.shear_y}
                onChange={(e) => setTransformParams(prev => ({ ...prev, shear_y: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          )}
          {transformationType === 'general' && (
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-gray-500">m₀₀: {transformParams.m00.toFixed(2)}</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={transformParams.m00}
                  onChange={(e) => setTransformParams(prev => ({ ...prev, m00: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">m₀₁: {transformParams.m01.toFixed(2)}</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={transformParams.m01}
                  onChange={(e) => setTransformParams(prev => ({ ...prev, m01: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">m₁₀: {transformParams.m10.toFixed(2)}</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={transformParams.m10}
                  onChange={(e) => setTransformParams(prev => ({ ...prev, m10: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">m₁₁: {transformParams.m11.toFixed(2)}</label>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={transformParams.m11}
                  onChange={(e) => setTransformParams(prev => ({ ...prev, m11: parseFloat(e.target.value) }))}
                  className="w-full"
                />
              </div>
            </div>
          )}
        </div>
        
        {/* Special presets */}
        <div className="space-y-2">
          <label className="block text-sm font-medium">Special Configurations</label>
          <div className="flex gap-2">
            <button
              onClick={() => {
                setLatticeType('hexagonal');
                setTransformationType('rotation');
                setTransformParams(prev => ({ ...prev, angle: 1.05, scale: 1 }));
              }}
              className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            >
              Magic Angle Graphene
            </button>
            <button
              onClick={() => {
                setTransformationType('rotation');
                setTransformParams(prev => ({ ...prev, angle: 21.8, scale: 1 }));
              }}
              className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            >
              21.8° Commensurate
            </button>
            <button
              onClick={() => {
                setTransformationType('rotation');
                setTransformParams(prev => ({ ...prev, angle: 30, scale: 1 }));
              }}
              className="px-3 py-1 text-xs border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
            >
              30° Rotation
            </button>
          </div>
        </div>
      </div>
      
      {/* Debug information */}
      {is_debug && (
        <div className="w-full mt-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg text-xs font-mono">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2">Moiré Properties</h4>
              {moireLattice ? (
                <>
                  <div>Period Ratio: {moireLattice.moire_period_ratio().toFixed(4)}</div>
                  <div>Twist Angle: {moireLattice.twist_angle_degrees().toFixed(4)}°</div>
                  <div>Cell Area: {moireLattice.cell_area().toFixed(4)}</div>
                  <div>Commensurate: {moireLattice.is_commensurate() ? 'Yes' : 'No'}</div>
                  {moireLattice.coincidence_indices() && (
                    <div>Indices: {moireLattice.coincidence_indices()?.join(', ')}</div>
                  )}
                </>
              ) : (
                <div className="text-red-500">
                  {moireError ? `Error: ${moireError}` : 'No moiré lattice created'}
                </div>
              )}
            </div>
            <div>
              <h4 className="font-semibold mb-2">Performance Stats</h4>
              <div>Visible Points: {pointCount}</div>
              <div>Lattice 1: {latticePointsData.lattice1.length}</div>
              <div>Lattice 2: {latticePointsData.lattice2.length}</div>
              <div>Moiré: {latticePointsData.moire.length}</div>
              <div>Scale: {scale.toFixed(2)}</div>
              <div>Render Mode: {renderQuality}</div>
              <div>Clustering: {renderQuality === 'clustered' ? 'Active' : 'Inactive'}</div>
              {moireError && (
                <div className="text-red-500 mt-2">Status: Error state</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
