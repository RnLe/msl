'use client';

import { useState, useMemo, useRef, useEffect } from 'react';
import { Stage, Layer, Rect, Circle, Line, Text, Group } from 'react-konva';
import { Square, SquareCheck } from 'lucide-react';
import { scaleLinear } from '@visx/scale';
import { LinePath, AreaClosed } from '@visx/shape';
import { curveLinear } from '@visx/curve';

interface Moire1DExampleProps {
  width?: number;
  height?: number;
}

// Neutral vintage color palette that works in both light and dark modes
const COLORS = {
  lattice1: '#8B7355', // Tan/Brown - warm neutral
  lattice2: '#4A5568', // Cool Gray - complementary neutral
  grid: '#A0AEC0', // Light Gray
  axes: '#718096', // Medium Gray
  text: '#4A5568', // Dark Gray
  background: 'transparent',
  stripe1Light: '#A0826D', // Light Tan
  stripe1Dark: '#6B5B45', // Dark Tan
  stripe2Light: '#718096', // Light Cool Gray
  stripe2Dark: '#2D3748', // Dark Cool Gray
  black: '#1A202C', // Near Black for when colors are off
  envelope: '#805AD5', // Purple for envelope - stands out but vintage
};

export function Moire1DExample({ 
  width,
  height = 300,
}: Moire1DExampleProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(width || 800);
  const [scaleRatio, setScaleRatio] = useState(0.9); // Default to 0.75 (in range 0.1-2)
  
  // Toggle states
  const [showGrid, setShowGrid] = useState(true);
  const [showLattice1, setShowLattice1] = useState(true);
  const [showLattice2, setShowLattice2] = useState(true);
  const [coloredStripes, setColoredStripes] = useState(true);
  const [transparentStripes, setTransparentStripes] = useState(false); // Default OFF (opaque)
  const [isDarkTheme, setIsDarkTheme] = useState(true); // Dark theme default
  
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
  const gridCanvasHeight = 150; // Height for the second canvas
  
  // Lattice parameters
  const a1 = 1; // Fixed spacing for first lattice
  const a2 = scaleRatio; // Variable spacing for second lattice
  
  // Scale for converting lattice units to pixels
  const scale = 32; // pixels per lattice unit (reduced from 40)
  
  // Smart scaling for stripe widths
  const baseStripeWidth = scale * 0.25;
  const stripe1Width = baseStripeWidth * (0.6 + 0.6 * scaleRatio); // Higher upper bound: 0.6-1.2 of base (was 0.6-0.8)
  const stripe2Width = baseStripeWidth * Math.max(0.3, scaleRatio * 1.6); // Stronger upper limit: 1.6x instead of 0.8x
  
  // Dynamic scaling for lattice point size - scale down linearly from s=1
  const lattice2PointRadius = 4 * Math.min(1, scaleRatio); // Scale down from 4 when s < 1
  
  // Sine wave data for frequency plot
  const plotWidth = canvasWidth;
  const plotHeight = 200;
  const plotMargin = { top: 20, right: 20, bottom: 40, left: 40 };
  const innerPlotWidth = plotWidth - plotMargin.left - plotMargin.right;
  const innerPlotHeight = plotHeight - plotMargin.top - plotMargin.bottom;
  
  // Generate sine wave data
  const sineData = useMemo(() => {
    const numPoints = 500;
    const data = [];
    
    for (let i = 0; i < numPoints; i++) {
      const x = (i / (numPoints - 1)) * (canvasWidth / scale); // x in lattice units
      const lattice1Freq = 2 * Math.PI / a1; // frequency for lattice 1
      const lattice2Freq = 2 * Math.PI / a2; // frequency for lattice 2
      
      const sine1 = Math.sin(lattice1Freq * x);
      const sine2 = Math.sin(lattice2Freq * x);
      const sum = sine1 + sine2;
      
      // Correct envelope calculation for sum of two sine waves
      // Envelope = 2 * |cos((ω1 - ω2) * x / 2)|
      const beatFreq = (lattice1Freq - lattice2Freq) / 2;
      const envelope = 2 * Math.abs(Math.cos(beatFreq * x));
      
      data.push({
        x: x,
        sine1: sine1,
        sine2: sine2,
        sum: sum,
        envelope: envelope
      });
    }
    
    return data;
  }, [canvasWidth, scale, a1, a2]);
  
  // Scales for the plot
  const xScale = scaleLinear({
    range: [0, innerPlotWidth],
    domain: [0, canvasWidth / scale]
  });
  
  const yScale = scaleLinear({
    range: [innerPlotHeight, 0],
    domain: [-2.5, 2.5] // Range to accommodate sum of two sines
  });
  
  // Generate lattice points
  const lattice1Points = useMemo(() => {
    const points = [];
    const numPoints = Math.ceil(canvasWidth / (a1 * scale)) + 2;
    for (let i = 0; i < numPoints; i++) {
      points.push(i * a1);
    }
    return points;
  }, [canvasWidth, a1, scale]);
  
  const lattice2Points = useMemo(() => {
    const points = [];
    const numPoints = Math.ceil(canvasWidth / (a2 * scale)) + 2;
    for (let i = 0; i < numPoints; i++) {
      points.push(i * a2);
    }
    return points;
  }, [canvasWidth, a2, scale]);
  
  // Generate grid lines for the second canvas
  const gridLines = useMemo(() => {
    const lines = [];
    const numVertical = Math.ceil(canvasWidth / scale) + 1;
    
    // Vertical lines
    for (let i = 0; i < numVertical; i++) {
      lines.push({
        key: `v-${i}`,
        points: [i * scale, 0, i * scale, gridCanvasHeight],
        stroke: COLORS.grid,
        strokeWidth: 1
      });
    }
    
    // Horizontal lines - just the main axes
    lines.push({
      key: 'h-center',
      points: [0, gridCanvasHeight / 2, canvasWidth, gridCanvasHeight / 2],
      stroke: COLORS.axes,
      strokeWidth: 2
    });
    
    return lines;
  }, [canvasWidth, gridCanvasHeight, scale]);
  
  return (
    <div ref={containerRef} className="moire-1d-example w-full flex flex-col items-center">
      {/* Top canvas - Moiré pattern visualization */}
      <Stage width={canvasWidth} height={height}>
        <Layer>
          {/* First lattice stripes */}
          {showLattice1 && lattice1Points.map((x, i) => (
            <Rect
              key={`stripe1-${i}`}
              x={x * scale}
              y={0}
              width={stripe1Width}
              height={height}
              fill={coloredStripes ? COLORS.lattice1 : (isDarkTheme ? '#FFFFFF' : COLORS.black)}
              opacity={transparentStripes ? 0.6 : 1.0}
            />
          ))}
          
          {/* Second lattice stripes */}
          {showLattice2 && lattice2Points.map((x, i) => (
            <Rect
              key={`stripe2-${i}`}
              x={x * scale}
              y={0}
              width={stripe2Width}
              height={height}
              fill={coloredStripes ? COLORS.lattice2 : (isDarkTheme ? '#FFFFFF' : COLORS.black)}
              opacity={transparentStripes ? 0.5 : 1.0}
            />
          ))}
        </Layer>
      </Stage>
      
      {/* Slider */}
      <div className="w-full max-w-2xl mx-auto my-4 px-4">
        <div className="flex items-center gap-4">
          <label className="text-sm font-medium whitespace-nowrap">
            Scale ratio s = {scaleRatio.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.1"
            max="2"
            step="0.01"
            value={scaleRatio}
            onChange={(e) => setScaleRatio(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
        </div>
      </div>
      
      {/* Second canvas - Grid and lattice points */}
      <Stage width={canvasWidth} height={gridCanvasHeight}>
        <Layer>
          {/* Background grid */}
          {showGrid && (
            <Group opacity={0.3}>
              {gridLines.map((line) => {
                const { key, ...lineProps } = line;
                return <Line key={key} {...lineProps} />;
              })}
            </Group>
          )}
          
          {/* Lattice 1 points on grid line */}
          {showLattice1 && (
            <Group>
              {lattice1Points.map((x, i) => (
                <Circle
                  key={`point1-${i}`}
                  x={x * scale}
                  y={gridCanvasHeight / 2 - scale}
                  radius={4}
                  fill={COLORS.lattice1}
                />
              ))}
              <Text
                x={10}
                y={gridCanvasHeight / 2 - scale - 15}
                text="a = 1"
                fontSize={12}
                fill={COLORS.text}
              />
            </Group>
          )}
          
          {/* Lattice 2 points below */}
          {showLattice2 && (
            <Group>
              {lattice2Points.map((x, i) => (
                <Circle
                  key={`point2-${i}`}
                  x={x * scale}
                  y={gridCanvasHeight / 2 + scale}
                  radius={lattice2PointRadius}
                  fill={COLORS.lattice2}
                />
              ))}
              <Text
                x={10}
                y={gridCanvasHeight / 2 + scale - 15}
                text={`a' = ${scaleRatio.toFixed(2)}`}
                fontSize={12}
                fill={COLORS.text}
              />
            </Group>
          )}
        </Layer>
      </Stage>
      
      {/* Frequency Plot */}
      <div className="w-full mt-4">
        <svg width={plotWidth} height={plotHeight}>
          <defs>
            <linearGradient id="envelope-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={COLORS.envelope} stopOpacity={0.2} />
              <stop offset="100%" stopColor={COLORS.envelope} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          
          <g transform={`translate(${plotMargin.left}, ${plotMargin.top})`}>
            {/* Grid lines */}
            <line
              x1={0}
              y1={yScale(0)}
              x2={innerPlotWidth}
              y2={yScale(0)}
              stroke={COLORS.grid}
              strokeWidth={1}
            />
            
            {/* Envelope area */}
            <AreaClosed
              data={sineData}
              x={(d) => xScale(d.x)}
              y0={yScale(0)}
              y1={(d) => yScale(d.envelope)}
              yScale={yScale}
              fill="url(#envelope-gradient)"
              curve={curveLinear}
            />
            <AreaClosed
              data={sineData}
              x={(d) => xScale(d.x)}
              y0={yScale(0)}
              y1={(d) => yScale(-d.envelope)}
              yScale={yScale}
              fill="url(#envelope-gradient)"
              curve={curveLinear}
            />
            
            {/* Individual sine waves */}
            {showLattice1 && (
              <LinePath
                data={sineData}
                x={(d) => xScale(d.x)}
                y={(d) => yScale(d.sine1)}
                stroke={coloredStripes ? COLORS.lattice1 : (isDarkTheme ? '#FFFFFF' : COLORS.black)}
                strokeWidth={2}
                strokeOpacity={transparentStripes ? 0.6 : 1.0}
                curve={curveLinear}
              />
            )}
            
            {showLattice2 && (
              <LinePath
                data={sineData}
                x={(d) => xScale(d.x)}
                y={(d) => yScale(d.sine2)}
                stroke={coloredStripes ? COLORS.lattice2 : (isDarkTheme ? '#FFFFFF' : COLORS.black)}
                strokeWidth={2}
                strokeOpacity={transparentStripes ? 0.6 : 1.0}
                curve={curveLinear}
              />
            )}
            
            {/* Sum wave */}
            {(showLattice1 && showLattice2) && (
              <LinePath
                data={sineData}
                x={(d) => xScale(d.x)}
                y={(d) => yScale(d.sum)}
                stroke={COLORS.axes}
                strokeWidth={3}
                curve={curveLinear}
              />
            )}
            
            {/* Envelope lines */}
            {(showLattice1 && showLattice2) && (
              <>
                <LinePath
                  data={sineData}
                  x={(d) => xScale(d.x)}
                  y={(d) => yScale(d.envelope)}
                  stroke={COLORS.envelope}
                  strokeWidth={1.5}
                  strokeDasharray="5,5"
                  curve={curveLinear}
                />
                <LinePath
                  data={sineData}
                  x={(d) => xScale(d.x)}
                  y={(d) => yScale(-d.envelope)}
                  stroke={COLORS.envelope}
                  strokeWidth={1.5}
                  strokeDasharray="5,5"
                  curve={curveLinear}
                />
              </>
            )}
            
            {/* Axis labels */}
            <text
              x={innerPlotWidth / 2}
              y={innerPlotHeight + 30}
              textAnchor="middle"
              fontSize="12"
              fill={COLORS.text}
            >
              Position (lattice units)
            </text>
            <text
              x={-10}
              y={innerPlotHeight / 2}
              textAnchor="middle"
              fontSize="12"
              fill={COLORS.text}
              transform={`rotate(-90, -10, ${innerPlotHeight / 2})`}
            >
              Amplitude
            </text>
          </g>
        </svg>
        
        {/* Legend */}
        <div className="flex flex-wrap justify-center gap-4 mt-3 px-4">
          {showLattice1 && (
            <div className="flex items-center gap-2">
              <div 
                className="w-4 h-0.5" 
                style={{ 
                  backgroundColor: coloredStripes ? COLORS.lattice1 : (isDarkTheme ? '#FFFFFF' : COLORS.black),
                  opacity: transparentStripes ? 0.6 : 1.0
                }}
              />
              <span className="text-xs text-gray-600 dark:text-gray-400">
                Lattice 1 (a = 1)
              </span>
            </div>
          )}
          
          {showLattice2 && (
            <div className="flex items-center gap-2">
              <div 
                className="w-4 h-0.5" 
                style={{ 
                  backgroundColor: coloredStripes ? COLORS.lattice2 : (isDarkTheme ? '#FFFFFF' : COLORS.black),
                  opacity: transparentStripes ? 0.6 : 1.0
                }}
              />
              <span className="text-xs text-gray-600 dark:text-gray-400">
                Lattice 2 (a' = {scaleRatio.toFixed(2)})
              </span>
            </div>
          )}
          
          <div className="flex items-center gap-2">
            <div 
              className="w-4 h-0.5" 
              style={{ backgroundColor: COLORS.axes }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400">
              Sum
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <div 
              className="w-4 h-0.5 border-dashed border-t-2" 
              style={{ borderColor: COLORS.envelope }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400">
              Envelope
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <div 
              className="w-4 h-2 opacity-30" 
              style={{ backgroundColor: COLORS.envelope }}
            />
            <span className="text-xs text-gray-600 dark:text-gray-400">
              Envelope Area
            </span>
          </div>
        </div>
      </div>
      
      {/* Toggle buttons */}
      <div className="flex justify-evenly w-full gap-2 mt-4">
        <div
          onClick={() => setShowGrid(!showGrid)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Grid</span>
          {showGrid ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => {
            if (showLattice2) {  // Only allow turning off if the other is on
              setShowLattice1(!showLattice1);
            }
          }}
          className={`flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1 ${
            !showLattice2 ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-sm font-medium mr-2">Lattice 1</span>
          {showLattice1 ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => {
            if (showLattice1) {  // Only allow turning off if the other is on
              setShowLattice2(!showLattice2);
            }
          }}
          className={`flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1 ${
            !showLattice1 ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          <span className="text-sm font-medium mr-2">Lattice 2</span>
          {showLattice2 ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-blue-600 dark:text-blue-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setColoredStripes(!coloredStripes)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Colors</span>
          {coloredStripes ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-yellow-600 dark:text-yellow-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => setTransparentStripes(!transparentStripes)}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Transparent</span>
          {transparentStripes ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-yellow-600 dark:text-yellow-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
      
      {/* Second row - Color selection */}
      <div className="flex justify-evenly w-full gap-2 mt-2">
        <div
          onClick={() => {
            setIsDarkTheme(false);
            setColoredStripes(false);
          }}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">Black Stripes</span>
          {!isDarkTheme && !coloredStripes ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-gray-600 dark:text-gray-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
        
        <div
          onClick={() => {
            setIsDarkTheme(true);
            setColoredStripes(false);
          }}
          className="flex items-center justify-between px-3 py-2 border border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer transition-all duration-200 flex-1"
        >
          <span className="text-sm font-medium mr-2">White Stripes</span>
          {isDarkTheme && !coloredStripes ? (
            <SquareCheck className="w-4 h-4 flex-shrink-0 text-gray-600 dark:text-gray-400" />
          ) : (
            <Square className="w-4 h-4 flex-shrink-0 text-gray-400 dark:text-gray-500" />
          )}
        </div>
      </div>
    </div>
  );
}
