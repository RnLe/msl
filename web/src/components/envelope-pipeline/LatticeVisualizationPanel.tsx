'use client'

import React, { useMemo, useRef, useState, useEffect } from 'react'
import MoireLatticeCanvas from './MoireLatticeCanvas'
import BilayerZoomCanvas from './BilayerZoomCanvas'
import RegistryShiftCanvas from './RegistryShiftCanvas'
import TwoAtomicBasisCanvas from './TwoAtomicBasisCanvas'
import { useEnvelopePipelineStore } from './store'
import { normalizeLatticeType, computeMoireLength, LATTICE_COLORS, type LatticeType, type Vec2 } from './lattice-utils'

interface LatticeVisualizationPanelProps {
  /** Lattice type ('hex', 'triangular', 'square', etc.) */
  latticeType: string
  /** Twist angle in degrees */
  thetaDeg: number
  /** Lattice constant (default 1.0) */
  a?: number
  /** Hole radius / a */
  rOverA?: number
  /** Moiré length scale (if provided, used for R calculation) */
  moireLength?: number
  /** Grid dimensions for R calculation from pixel coords */
  gridDims?: { rows: number; cols: number }
  /** Optional eta override (default: computed from geometry) */
  eta?: number
}

/**
 * Combined visualization panel for lattice geometry.
 * 
 * Shows four key views in a 2x2 grid:
 * 1. Moiré Unit Cell - The full moiré pattern with both layers
 * 2. Bilayer Zoom - Zoomed view of ~3×3 unit cells at selected R
 * 3. 2-Atomic Basis - Physical view of shifted atoms at selected R
 * 4. Registry Shift - The δ(R) calculation for BLAZE
 * 
 * All components are linked to the Zustand store for pixel selection.
 */
export function LatticeVisualizationPanel({
  latticeType,
  thetaDeg,
  a = 1.0,
  rOverA = 0.3,
  moireLength: moireLengthProp,
  gridDims,
  eta,
}: LatticeVisualizationPanelProps) {
  const normalizedType = normalizeLatticeType(latticeType)
  const thetaRad = (thetaDeg * Math.PI) / 180
  
  // Get store state
  const selectedPixel = useEnvelopePipelineStore((s) => s.selectedPixel)
  const hoveredPixel = useEnvelopePipelineStore((s) => s.hoveredPixel)
  const cursorPosition = useEnvelopePipelineStore((s) => s.cursorPosition)
  const selectedPosition = useEnvelopePipelineStore((s) => s.selectedPosition)
  
  // Compute moiré length if not provided
  const moireLength = moireLengthProp ?? computeMoireLength(a, thetaRad)
  
  // The moiré lattice shows 2×L_m extent, so it needs 2× the pixel resolution
  // This way each data plot pixel corresponds to exactly one moiré unit cell
  const moireGridDims = useMemo(() => {
    if (!gridDims) return undefined
    return {
      rows: gridDims.rows * 2,
      cols: gridDims.cols * 2,
    }
  }, [gridDims])
  
  // The position used by the three detail components:
  // - Uses continuous cursorPosition when hovering in moiré lattice
  // - Falls back to selectedPosition (exact R coords from click)
  // - Defaults to origin
  const detailPosition = useMemo<Vec2>(() => {
    if (cursorPosition) {
      return { x: cursorPosition.x, y: cursorPosition.y }
    }
    if (selectedPosition) {
      return { x: selectedPosition.x, y: selectedPosition.y }
    }
    return { x: 0, y: 0 }
  }, [cursorPosition, selectedPosition])
  
  const hasPixelSelected = selectedPixel !== null || hoveredPixel !== null
  const activePixel = selectedPixel ?? hoveredPixel
  
  // Responsive canvas sizing
  const gridRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState(340)
  
  useEffect(() => {
    const updateSize = () => {
      if (gridRef.current) {
        // Get available width, subtract gap (16px), divide by 2 columns
        const availableWidth = gridRef.current.clientWidth
        const gap = 16
        const cellWidth = Math.floor((availableWidth - gap) / 2)
        // Clamp between reasonable bounds
        const size = Math.max(200, Math.min(600, cellWidth))
        setCanvasSize(size)
      }
    }
    
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])
  
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      padding: '16px',
      background: LATTICE_COLORS.panelBg,
      borderRadius: '12px',
      border: `1px solid ${LATTICE_COLORS.border}`,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: `1px solid ${LATTICE_COLORS.border}`,
        paddingBottom: '12px',
      }}>
        <h3 style={{
          margin: 0,
          fontSize: '18px',
          fontWeight: 600,
          color: LATTICE_COLORS.text,
        }}>
          Lattice Geometry
        </h3>
        <div style={{
          display: 'flex',
          gap: '16px',
          fontSize: '13px',
          color: LATTICE_COLORS.textMuted,
        }}>
          <span><strong style={{ color: LATTICE_COLORS.text }}>Type:</strong> {formatLatticeType(normalizedType)}</span>
          <span><strong style={{ color: LATTICE_COLORS.text }}>θ:</strong> {thetaDeg.toFixed(2)}°</span>
          <span><strong style={{ color: LATTICE_COLORS.text }}>r/a:</strong> {rOverA.toFixed(2)}</span>
          <span><strong style={{ color: LATTICE_COLORS.text }}>L<sub>m</sub>:</strong> {moireLength.toFixed(1)}a</span>
        </div>
      </div>
      
      {/* Canvas grid - 2x2 layout with responsive sizing */}
      <div 
        ref={gridRef}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '16px',
        }}
      >
        <MoireLatticeCanvas
          latticeType={normalizedType}
          thetaDeg={thetaDeg}
          a={a}
          rOverA={rOverA}
          moireLength={moireLength}
          gridDims={moireGridDims}
          dataGridDims={gridDims}
          width={canvasSize}
          height={canvasSize}
        />
        
        <BilayerZoomCanvas
          latticeType={normalizedType}
          thetaDeg={thetaDeg}
          a={a}
          rOverA={rOverA}
          viewRadius={1.5}
          zoomPosition={detailPosition}
          width={canvasSize}
          height={canvasSize}
        />
        
        <RegistryShiftCanvas
          latticeType={normalizedType}
          thetaDeg={thetaDeg}
          a={a}
          rOverA={rOverA}
          moirePosition={detailPosition}
          eta={eta}
          width={canvasSize}
          height={canvasSize}
        />
        
        <TwoAtomicBasisCanvas
          latticeType={normalizedType}
          thetaDeg={thetaDeg}
          a={a}
          rOverA={rOverA}
          moirePosition={detailPosition}
          eta={eta}
          width={canvasSize}
          height={canvasSize}
        />
      </div>
    </div>
  )
}

function formatLatticeType(type: LatticeType): string {
  switch (type) {
    case 'triangular':
    case 'hex':
      return 'Triangular'
    case 'square':
      return 'Square'
    case 'rect':
      return 'Rectangular'
    default:
      return type
  }
}

export default LatticeVisualizationPanel
