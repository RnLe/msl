'use client'

import React, { useMemo, useRef, useEffect, useCallback, useState } from 'react'
import {
  getBasisVectors,
  rotateBasis,
  computeMoireLength,
  generateLatticePoints,
  LATTICE_COLORS,
  normalizeLatticeType,
  type LatticeType,
  type BasisVectors,
  type Vec2,
} from './lattice-utils'
import { useEnvelopePipelineStore, type PixelCoord, type CursorPosition } from './store'

interface MoireLatticeCanvasProps {
  latticeType: LatticeType
  thetaDeg: number
  /** Lattice constant (default 1.0) */
  a?: number
  /** Hole radius / a (required for correct dot sizes) */
  rOverA?: number
  /** Width of the canvas in pixels */
  width?: number
  /** Height of the canvas in pixels */
  height?: number
  /** Moiré length scale (if provided, used for extent) */
  moireLength?: number
  /** Grid dimensions for pixel overlay (2N x 2N for moiré lattice) */
  gridDims?: { rows: number; cols: number }
  /** Data grid dimensions (N x N) for wrapping pixel coordinates to sync with data plots */
  dataGridDims?: { rows: number; cols: number }
}

/**
 * Canvas-based visualization of a twisted bilayer moiré lattice.
 * 
 * Shows the moiré pattern with:
 * - All lattice sites in white (both layers combined)
 * - Correct r/a radius for dots
 * - Moiré lattice points highlighted
 * - Pixel grid overlay linked to Zustand store for hover/selection
 * 
 * Uses extent of 2×L_m to show complete moiré structure.
 */
export function MoireLatticeCanvas({
  latticeType,
  thetaDeg,
  a = 1.0,
  rOverA = 0.3,
  width = 400,
  height = 400,
  moireLength: moireLengthProp,
  gridDims,
  dataGridDims,
}: MoireLatticeCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  
  // Local state for moiré pixel overlay (2N × 2N grid)
  const [localHoveredPixel, setLocalHoveredPixel] = useState<PixelCoord | null>(null)
  const [localSelectedPixel, setLocalSelectedPixel] = useState<PixelCoord | null>(null)
  
  // Get store state (for syncing with data plots - uses wrapped N × N coords)
  const hoveredPixel = useEnvelopePipelineStore((s) => s.hoveredPixel)
  const setHoveredPixel = useEnvelopePipelineStore((s) => s.setHoveredPixel)
  const selectedPixel = useEnvelopePipelineStore((s) => s.selectedPixel)
  const setSelectedPixel = useEnvelopePipelineStore((s) => s.setSelectedPixel)
  const setCursorPosition = useEnvelopePipelineStore((s) => s.setCursorPosition)
  const setSelectedPosition = useEnvelopePipelineStore((s) => s.setSelectedPosition)
  
  // Helper to wrap moiré pixel (2N × 2N) to data pixel (N × N) using modulo
  const wrapPixelToDataGrid = useCallback((pixel: PixelCoord | null): PixelCoord | null => {
    if (!pixel || !dataGridDims || !gridDims) return null
    
    // Calculate offset to align centers
    // Moiré grid is 2N x 2N, Data grid is N x N
    // Center of Moiré is at (N, N), Center of Data is at (N/2, N/2)
    // We want Moiré(N, N) -> Data(N/2, N/2)
    // Offset = N - N/2 = N/2
    const rowOffset = Math.floor((gridDims.rows - dataGridDims.rows) / 2)
    const colOffset = Math.floor((gridDims.cols - dataGridDims.cols) / 2)
    
    return {
      row: ((pixel.row - rowOffset) % dataGridDims.rows + dataGridDims.rows) % dataGridDims.rows,
      col: ((pixel.col - colOffset) % dataGridDims.cols + dataGridDims.cols) % dataGridDims.cols,
    }
  }, [dataGridDims, gridDims])

  // Helper to unwrap data pixel (N × N) to moiré pixel (2N × 2N) - maps to center square
  const unwrapPixelFromDataGrid = useCallback((pixel: PixelCoord | null): PixelCoord | null => {
    if (!pixel || !dataGridDims || !gridDims) return null
    
    const rowOffset = Math.floor((gridDims.rows - dataGridDims.rows) / 2)
    const colOffset = Math.floor((gridDims.cols - dataGridDims.cols) / 2)
    
    return {
      row: pixel.row + rowOffset,
      col: pixel.col + colOffset,
    }
  }, [dataGridDims, gridDims])
  
  // Normalize lattice type
  const normalizedType = normalizeLatticeType(latticeType)
  const thetaRad = (thetaDeg * Math.PI) / 180
  
  // Compute geometry
  const geometry = useMemo(() => {
    const baseBasis = getBasisVectors(normalizedType)
    
    // Scale by lattice constant
    const layer1Basis: BasisVectors = {
      a1: { x: baseBasis.a1.x * a, y: baseBasis.a1.y * a },
      a2: { x: baseBasis.a2.x * a, y: baseBasis.a2.y * a },
    }
    
    // Rotate for layer 2
    const layer2Basis = rotateBasis(layer1Basis, thetaRad)
    
    // Moiré length scale
    const moireLength = moireLengthProp ?? computeMoireLength(a, thetaRad)
    
    // Moiré basis vectors
    // For twisted bilayer: moiré pattern is rotated by +θ/2 relative to layer 1
    // (since layer 1 is unrotated and layer 2 is rotated by θ)
    const scaleFactor = moireLength / a
    let moireAngle = thetaRad / 2  // Moiré rotation relative to layer 1
    
    // FIX: For hexagonal lattices, the Moiré lattice is rotated by 30 degrees relative to the atomic lattice
    // (Reciprocal vectors G are 30 deg from real vectors a, and L_m || G)
    if (normalizedType === 'triangular') {
      moireAngle += Math.PI / 6
    }
    
    const scaledBasis: BasisVectors = {
      a1: { x: baseBasis.a1.x * scaleFactor, y: baseBasis.a1.y * scaleFactor },
      a2: { x: baseBasis.a2.x * scaleFactor, y: baseBasis.a2.y * scaleFactor },
    }
    const moireBasis = rotateBasis(scaledBasis, moireAngle)
    
    // Compute inverse basis for wrapping
    const det = moireBasis.a1.x * moireBasis.a2.y - moireBasis.a1.y * moireBasis.a2.x
    const moireBasisInv = {
      b1: { x: moireBasis.a2.y / det, y: -moireBasis.a2.x / det }, // Row 1
      b2: { x: -moireBasis.a1.y / det, y: moireBasis.a1.x / det }  // Row 2
    }
    
    return {
      layer1Basis,
      layer2Basis,
      moireBasis,
      moireBasisInv,
      moireLength,
      scaleFactor,
    }
  }, [normalizedType, thetaRad, a, moireLengthProp])
  
  // Pre-compute all lattice points
  const allPoints = useMemo(() => {
    const { layer1Basis, layer2Basis, moireBasis, moireLength } = geometry
    
    // Use 2×L_m as the extent (centered at origin)
    const halfL = moireLength  // Full L_m on each side = 2×L_m total
    const extent = {
      xMin: -halfL,
      xMax: halfL,
      yMin: -halfL,
      yMax: halfL,
    }
    
    // Generate points for both layers
    const layer1Points = generateLatticePoints(layer1Basis, extent, 0.02)
    const layer2Points = generateLatticePoints(layer2Basis, extent, 0.02)
    
    // Generate moiré lattice points
    const moirePoints = generateLatticePoints(moireBasis, extent, 0.1)
    
    return {
      layer1Points,
      layer2Points,
      moirePoints,
      extent,
    }
  }, [geometry])
  
  // Padding for plot area (with space for axis labels)
  const padding = useMemo(() => ({ left: 45, right: 12, top: 28, bottom: 40 }), [])
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  
  // Compute coordinate transforms
  const { dataToCanvas, canvasToData } = useMemo(() => {
    const { extent } = allPoints
    
    const dataToCanvas = (p: Vec2): Vec2 => {
      const x = padding.left + ((p.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotWidth
      const y = padding.top + (1 - (p.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotHeight
      return { x, y }
    }
    
    const canvasToData = (cx: number, cy: number): Vec2 => {
      const { extent } = allPoints
      const x = extent.xMin + ((cx - padding.left) / plotWidth) * (extent.xMax - extent.xMin)
      const y = extent.yMax - ((cy - padding.top) / plotHeight) * (extent.yMax - extent.yMin)
      return { x, y }
    }
    
    return { dataToCanvas, canvasToData }
  }, [allPoints, padding, plotWidth, plotHeight])

  // Helper to map real-space R to data pixel (N × N) using lattice wrapping
  const mapRealSpaceToDataPixel = useCallback((r: Vec2): PixelCoord | null => {
    if (!dataGridDims || !geometry) return null
    
    const { moireBasisInv, moireLength } = geometry
    
    // 1. Convert to fractional coordinates in Moiré basis
    // f = M_inv * r
    const f1 = moireBasisInv.b1.x * r.x + moireBasisInv.b1.y * r.y
    const f2 = moireBasisInv.b2.x * r.x + moireBasisInv.b2.y * r.y
    
    // 2. Wrap fractional coordinates to [-0.5, 0.5] (Wigner-Seitz-like)
    const f1Wrapped = f1 - Math.round(f1)
    const f2Wrapped = f2 - Math.round(f2)
    
    // 3. Convert back to Cartesian
    // r' = f1' * a1 + f2' * a2
    const rWrapped = {
      x: f1Wrapped * geometry.moireBasis.a1.x + f2Wrapped * geometry.moireBasis.a2.x,
      y: f1Wrapped * geometry.moireBasis.a1.y + f2Wrapped * geometry.moireBasis.a2.y
    }
    
    // 4. Map to Data Pixel Grid
    // Data grid covers [-L_m/2, L_m/2]
    const halfL = moireLength / 2
    
    // Check if wrapped point is within the square data box
    if (rWrapped.x < -halfL || rWrapped.x > halfL || 
        rWrapped.y < -halfL || rWrapped.y > halfL) {
      return null
    }
    
    // Convert to normalized coords [0, 1]
    const normX = (rWrapped.x + halfL) / moireLength
    const normY = (rWrapped.y + halfL) / moireLength
    
    // Convert to pixel indices
    const col = Math.floor(normX * dataGridDims.cols)
    const row = Math.floor(normY * dataGridDims.rows)
    
    // Clamp (should be handled by bounds check, but safe)
    if (row < 0 || row >= dataGridDims.rows || col < 0 || col >= dataGridDims.cols) {
      return null
    }
    
    return { row, col }
  }, [dataGridDims, geometry])
  
  // Get pixel from mouse position
  const getPixelFromMouse = useCallback((e: React.MouseEvent<HTMLCanvasElement>): PixelCoord | null => {
    if (!gridDims) return null
    
    const canvas = e.currentTarget
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    
    // Check if within plot area
    if (cx < padding.left || cx > padding.left + plotWidth ||
        cy < padding.top || cy > padding.top + plotHeight) {
      return null
    }
    
    // Convert to normalized [0, 1] position within plot
    const normX = (cx - padding.left) / plotWidth
    const normY = (cy - padding.top) / plotHeight
    
    // Convert to pixel indices
    // Match the data plot convention: row 0 = bottom (high y), row N-1 = top (low y)
    // This matches Phase1FieldsPanel which uses: row = rows - 1 - Math.floor(normY * rows)
    const col = Math.floor(normX * gridDims.cols)
    const row = gridDims.rows - 1 - Math.floor(normY * gridDims.rows)
    
    // Clamp to valid range
    if (row < 0 || row >= gridDims.rows || col < 0 || col >= gridDims.cols) {
      return null
    }
    
    return { row, col }
  }, [gridDims, padding, plotWidth, plotHeight])
  
  // Get continuous cursor position in real-space R coordinates
  const getCursorPositionFromMouse = useCallback((e: React.MouseEvent<HTMLCanvasElement>): CursorPosition | null => {
    const canvas = e.currentTarget
    const rect = canvas.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    
    // Check if within plot area
    if (cx < padding.left || cx > padding.left + plotWidth ||
        cy < padding.top || cy > padding.top + plotHeight) {
      return null
    }
    
    // Convert canvas coords to data coords (real-space R)
    const dataPos = canvasToData(cx, cy)
    return { x: dataPos.x, y: dataPos.y }
  }, [padding, plotWidth, plotHeight, canvasToData])
  
  // Mouse event handlers
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // Get moiré pixel (2N × 2N grid)
    const moirePixel = getPixelFromMouse(e)
    setLocalHoveredPixel(moirePixel)
    
    // Get continuous cursor position in real-space R coordinates
    const cursorPos = getCursorPositionFromMouse(e)
    setCursorPosition(cursorPos)
    
    // Map real-space position to data pixel using lattice wrapping
    if (cursorPos) {
      const wrappedPixel = mapRealSpaceToDataPixel(cursorPos)
      setHoveredPixel(wrappedPixel)
    } else {
      setHoveredPixel(null)
    }
  }, [getPixelFromMouse, mapRealSpaceToDataPixel, setHoveredPixel, getCursorPositionFromMouse, setCursorPosition])
  
  const handleMouseLeave = useCallback(() => {
    setLocalHoveredPixel(null)
    setHoveredPixel(null)
    setCursorPosition(null)
  }, [setHoveredPixel, setCursorPosition])
  
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const moirePixel = getPixelFromMouse(e)
    if (moirePixel) {
      setLocalSelectedPixel(moirePixel)
      
      // Also store the exact R position for the detail components
      const cursorPos = getCursorPositionFromMouse(e)
      setSelectedPosition(cursorPos)
      
      // Map real-space position to data pixel using lattice wrapping
      if (cursorPos) {
        const wrappedPixel = mapRealSpaceToDataPixel(cursorPos)
        setSelectedPixel(wrappedPixel)
      }
    }
  }, [getPixelFromMouse, mapRealSpaceToDataPixel, setSelectedPixel, getCursorPositionFromMouse, setSelectedPosition])
  
  // Render main lattice to canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { moireLength } = geometry
    const { layer1Points, layer2Points, moirePoints, extent } = allPoints
    
    // Clear canvas with dark background
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.fillRect(0, 0, width, height)
    
    // Compute the correct radius in canvas pixels
    // r/a is the physical hole radius in units of lattice constant
    const dataRadius = rOverA * a
    const dataExtent = extent.xMax - extent.xMin
    const canvasRadius = (dataRadius / dataExtent) * plotWidth
    // Ensure minimum visibility but keep it accurate
    const radius = Math.max(0.5, canvasRadius)
    
    // Draw all lattice points in WHITE (both layers merged)
    // Use a single path for efficiency with many points
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)'
    ctx.beginPath()
    
    // Batch draw layer 1 points
    for (const pt of layer1Points) {
      const c = dataToCanvas(pt)
      ctx.moveTo(c.x + radius, c.y)
      ctx.arc(c.x, c.y, radius, 0, Math.PI * 2)
    }
    
    // Batch draw layer 2 points
    for (const pt of layer2Points) {
      const c = dataToCanvas(pt)
      ctx.moveTo(c.x + radius, c.y)
      ctx.arc(c.x, c.y, radius, 0, Math.PI * 2)
    }
    
    ctx.fill()
    
    // Draw moiré lattice points (slightly larger, purple)
    const moireRadius = Math.max(3, radius * 2)
    ctx.fillStyle = LATTICE_COLORS.moireNode
    ctx.beginPath()
    for (const pt of moirePoints) {
      const c = dataToCanvas(pt)
      ctx.moveTo(c.x + moireRadius, c.y)
      ctx.arc(c.x, c.y, moireRadius, 0, Math.PI * 2)
    }
    ctx.fill()
    
    // Draw title only
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Moiré Lattice', width / 2, 16)
    
    // Minimal legend in top-left of plot area
    const legendX = padding.left + 8
    const legendY = padding.top + 12
    
    ctx.font = '10px system-ui, sans-serif'
    ctx.textAlign = 'left'
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    ctx.beginPath()
    ctx.arc(legendX, legendY, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('Bilayer', legendX + 10, legendY + 3)
    
    ctx.fillStyle = LATTICE_COLORS.moireNode
    ctx.beginPath()
    ctx.arc(legendX, legendY + 16, 4, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('Moiré', legendX + 10, legendY + 19)
    
    // Draw axis labels
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 13px system-ui, sans-serif'
    ctx.textAlign = 'center'
    
    // X-axis label (R_x / a) at bottom center
    ctx.fillText('R', width / 2 - 8, height - 8)
    ctx.font = '10px system-ui, sans-serif'
    ctx.fillText('x', width / 2 + 2, height - 5)
    ctx.font = 'bold 13px system-ui, sans-serif'
    ctx.fillText('/ a', width / 2 + 18, height - 8)
    
    // Y-axis label (R_y / a) rotated on left side
    ctx.save()
    ctx.translate(14, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('R', -8, 0)
    ctx.font = '10px system-ui, sans-serif'
    ctx.fillText('y', 2, 3)
    ctx.font = 'bold 13px system-ui, sans-serif'
    ctx.fillText('/ a', 18, 0)
    ctx.restore()
    
    // Draw axis tick values
    const halfL = moireLength
    const tickStep = moireLength / 2
    
    ctx.font = '10px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.textAlign = 'center'
    
    // X-axis ticks
    for (let x = -halfL; x <= halfL + 0.1; x += tickStep) {
      const c = dataToCanvas({ x, y: 0 })
      ctx.fillText(x.toFixed(0), c.x, height - padding.bottom + 15)
    }
    
    // Y-axis ticks
    ctx.textAlign = 'right'
    for (let y = -halfL; y <= halfL + 0.1; y += tickStep) {
      const c = dataToCanvas({ x: 0, y })
      ctx.fillText(y.toFixed(0), padding.left - 5, c.y + 3)
    }
    
  }, [geometry, allPoints, width, height, rOverA, a, dataToCanvas, padding, plotWidth])
  
  // Render pixel grid overlay
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay || !gridDims) return
    
    const ctx = overlay.getContext('2d')
    if (!ctx) return
    
    // Clear overlay
    ctx.clearRect(0, 0, width, height)
    
    const { rows, cols } = gridDims
    const cellWidth = plotWidth / cols
    const cellHeight = plotHeight / rows
    
    // Draw pixel grid lines (subtle)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)'
    ctx.lineWidth = 0.5
    
    // Vertical lines
    for (let c = 0; c <= cols; c++) {
      const x = padding.left + c * cellWidth
      ctx.beginPath()
      ctx.moveTo(x, padding.top)
      ctx.lineTo(x, padding.top + plotHeight)
      ctx.stroke()
    }
    
    // Horizontal lines
    for (let r = 0; r <= rows; r++) {
      const y = padding.top + r * cellHeight
      ctx.beginPath()
      ctx.moveTo(padding.left, y)
      ctx.lineTo(padding.left + plotWidth, y)
      ctx.stroke()
    }
    
    // Determine effective hovered pixel:
    // 1. Local hover (mouse over this canvas) takes precedence
    // 2. If no local hover, check store hover (from data plots) and map to center square
    const effectiveHoveredPixel = localHoveredPixel ?? unwrapPixelFromDataGrid(hoveredPixel)
    
    // Draw hovered pixel highlight
    // Note: row 0 = bottom of canvas, row N-1 = top, so we invert for drawing
    if (effectiveHoveredPixel) {
      const x = padding.left + effectiveHoveredPixel.col * cellWidth
      const y = padding.top + (rows - 1 - effectiveHoveredPixel.row) * cellHeight
      
      // Only draw if within bounds (unwrap might place it outside if logic was different, but here it's center square)
      if (effectiveHoveredPixel.row >= 0 && effectiveHoveredPixel.row < rows &&
          effectiveHoveredPixel.col >= 0 && effectiveHoveredPixel.col < cols) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)'
        ctx.fillRect(x, y, cellWidth, cellHeight)
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, cellWidth, cellHeight)
      }
    }
    
    // Determine effective selected pixel:
    // 1. If we have a local selection that matches the store selection (wrapped), use local (preserves exact periodic image)
    // 2. Otherwise use store selection mapped to center square
    let effectiveSelectedPixel = null
    if (selectedPixel) {
      if (localSelectedPixel) {
        // Check if local selection maps to the same data pixel as the store selection
        // We need to convert local pixel -> real space -> wrapped data pixel
        // But we don't have easy access to real space here without recalculating
        // Instead, we can just check if the store selection matches what we would expect
        // Actually, simpler: if localSelectedPixel is set, we trust it was the source of the click
        // UNLESS the store selection has changed (e.g. clicked on another plot)
        // But we can't easily detect that without comparing.
        
        // Let's just use the unwrap logic for consistency, unless we are sure.
        // If we clicked here, localSelectedPixel is set.
        // If we clicked elsewhere, localSelectedPixel might be stale? No, we should clear it?
        // We don't clear localSelectedPixel when store changes.
        
        // Let's just use unwrapPixelFromDataGrid for now to be safe and consistent with hover
        effectiveSelectedPixel = unwrapPixelFromDataGrid(selectedPixel)
      } else {
        effectiveSelectedPixel = unwrapPixelFromDataGrid(selectedPixel)
      }
    }

    // Draw selected pixel highlight
    if (effectiveSelectedPixel) {
      const x = padding.left + effectiveSelectedPixel.col * cellWidth
      const y = padding.top + (rows - 1 - effectiveSelectedPixel.row) * cellHeight
      
      if (effectiveSelectedPixel.row >= 0 && effectiveSelectedPixel.row < rows &&
          effectiveSelectedPixel.col >= 0 && effectiveSelectedPixel.col < cols) {
        ctx.fillStyle = 'rgba(0, 255, 255, 0.25)'
        ctx.fillRect(x, y, cellWidth, cellHeight)
        
        ctx.strokeStyle = LATTICE_COLORS.selectedPixel
        ctx.lineWidth = 2
        ctx.strokeRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2)
      }
    }
    
  }, [gridDims, localHoveredPixel, localSelectedPixel, hoveredPixel, selectedPixel, wrapPixelToDataGrid, unwrapPixelFromDataGrid, width, height, padding, plotWidth, plotHeight])
  
  return (
    <div style={{ position: 'relative', width, height }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          borderRadius: '8px',
          border: `1px solid ${LATTICE_COLORS.border}`,
          background: LATTICE_COLORS.background,
        }}
      />
      <canvas
        ref={overlayRef}
        width={width}
        height={height}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onMouseDown={handleMouseDown}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          borderRadius: '8px',
          cursor: gridDims ? 'crosshair' : 'default',
        }}
      />
    </div>
  )
}

export default MoireLatticeCanvas
