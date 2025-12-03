'use client'

import React, { useRef, useEffect, useMemo } from 'react'
import {
  getBasisVectors,
  rotateBasis,
  generateLatticePoints,
  LATTICE_COLORS,
  normalizeLatticeType,
  type LatticeType,
  type BasisVectors,
  type Vec2,
} from './lattice-utils'

interface BilayerZoomCanvasProps {
  latticeType: LatticeType
  thetaDeg: number
  /** Lattice constant (default 1.0) */
  a?: number
  /** Hole radius / a (optional) */
  rOverA?: number
  /** How many lattice constants to show in each direction from origin */
  viewRadius?: number
  /** Width of the canvas in pixels */
  width?: number
  /** Height of the canvas in pixels */
  height?: number
  /** Position in moiré coordinates to zoom in on (default: origin) */
  zoomPosition?: Vec2
}

/**
 * Canvas-based zoomed view of a bilayer photonic crystal.
 * 
 * Shows roughly 3x3 lattice constants around a specified position,
 * displaying the 2-atomic basis (one atom from each layer).
 * 
 * This helps visualize what the solver "sees" at each registry point.
 */
export function BilayerZoomCanvas({
  latticeType,
  thetaDeg,
  a = 1.0,
  rOverA = 0.3,
  viewRadius = 1.5,
  width = 400,
  height = 400,
  zoomPosition = { x: 0, y: 0 },
}: BilayerZoomCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
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
    
    return {
      layer1Basis,
      layer2Basis,
      baseBasis,
    }
  }, [normalizedType, thetaRad, a])
  
  // Render to canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { layer1Basis, layer2Basis, baseBasis } = geometry
    
    // Clear canvas with dark background
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.fillRect(0, 0, width, height)
    
    // Extent around zoom position
    const halfL = viewRadius * a
    const extent = {
      xMin: zoomPosition.x - halfL,
      xMax: zoomPosition.x + halfL,
      yMin: zoomPosition.y - halfL,
      yMax: zoomPosition.y + halfL,
    }
    
    // Padding
    const padding = { left: 45, right: 20, top: 40, bottom: 40 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom
    
    // Transform from data coords to canvas coords
    const dataToCanvas = (p: Vec2): Vec2 => {
      const x = padding.left + ((p.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotWidth
      const y = padding.top + (1 - (p.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotHeight
      return { x, y }
    }
    
    // Draw grid lines (at lattice constant intervals)
    ctx.strokeStyle = LATTICE_COLORS.gridLine
    ctx.lineWidth = 0.5
    const gridStep = a
    for (let x = Math.floor(extent.xMin / gridStep) * gridStep; x <= extent.xMax; x += gridStep) {
      const p1 = dataToCanvas({ x, y: extent.yMin })
      const p2 = dataToCanvas({ x, y: extent.yMax })
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }
    for (let y = Math.floor(extent.yMin / gridStep) * gridStep; y <= extent.yMax; y += gridStep) {
      const p1 = dataToCanvas({ x: extent.xMin, y })
      const p2 = dataToCanvas({ x: extent.xMax, y })
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }
    
    // Draw axes through origin
    ctx.strokeStyle = LATTICE_COLORS.axis
    ctx.lineWidth = 1
    const originCanvas = dataToCanvas({ x: 0, y: 0 })
    if (originCanvas.x >= padding.left && originCanvas.x <= padding.left + plotWidth) {
      ctx.beginPath()
      ctx.moveTo(originCanvas.x, padding.top)
      ctx.lineTo(originCanvas.x, padding.top + plotHeight)
      ctx.stroke()
    }
    if (originCanvas.y >= padding.top && originCanvas.y <= padding.top + plotHeight) {
      ctx.beginPath()
      ctx.moveTo(padding.left, originCanvas.y)
      ctx.lineTo(padding.left + plotWidth, originCanvas.y)
      ctx.stroke()
    }
    
    // Draw unit cell outline for layer 1 (centered at origin)
    ctx.strokeStyle = LATTICE_COLORS.layer1
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.5
    const cellCorners = [
      { x: 0, y: 0 },
      layer1Basis.a1,
      { x: layer1Basis.a1.x + layer1Basis.a2.x, y: layer1Basis.a1.y + layer1Basis.a2.y },
      layer1Basis.a2,
    ]
    ctx.beginPath()
    const cc0 = dataToCanvas(cellCorners[0])
    ctx.moveTo(cc0.x, cc0.y)
    for (let i = 1; i < cellCorners.length; i++) {
      const c = dataToCanvas(cellCorners[i])
      ctx.lineTo(c.x, c.y)
    }
    ctx.closePath()
    ctx.stroke()
    ctx.globalAlpha = 1.0
    
    // Generate lattice points for both layers
    const layer1Points = generateLatticePoints(layer1Basis, extent, 0.1)
    const layer2Points = generateLatticePoints(layer2Basis, extent, 0.1)
    
    // Draw holes/atoms with proper radius
    const dataRadius = rOverA * a
    const canvasRadius = (dataRadius / (extent.xMax - extent.xMin)) * plotWidth
    const radius = Math.max(3, Math.min(40, canvasRadius))
    
    // Draw layer 1 holes (blue, as circles with dark fill and colored border)
    ctx.lineWidth = 2
    for (const pt of layer1Points) {
      const c = dataToCanvas(pt)
      ctx.fillStyle = LATTICE_COLORS.background  // Dark fill (like air holes)
      ctx.strokeStyle = LATTICE_COLORS.layer1
      ctx.beginPath()
      ctx.arc(c.x, c.y, radius, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }
    
    // Draw layer 2 holes (orange, slightly transparent overlay)
    ctx.globalAlpha = 0.8
    for (const pt of layer2Points) {
      const c = dataToCanvas(pt)
      ctx.fillStyle = 'rgba(251, 146, 60, 0.15)'  // Transparent orange
      ctx.strokeStyle = LATTICE_COLORS.layer2
      ctx.beginPath()
      ctx.arc(c.x, c.y, radius, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }
    ctx.globalAlpha = 1.0
    
    // Highlight the 2-atomic basis at origin
    // Atom 0: Layer 1 at origin
    // Atom 1: Layer 2 at (approx) origin + small shift from twist
    const atom0 = dataToCanvas({ x: 0, y: 0 })
    const atom1 = dataToCanvas({ x: 0, y: 0 })  // At origin, shift is ~0
    
    // Draw markers for the basis atoms
    ctx.fillStyle = LATTICE_COLORS.layer1
    ctx.beginPath()
    ctx.arc(atom0.x, atom0.y, 4, 0, Math.PI * 2)
    ctx.fill()
    
    // Draw basis vectors from origin
    const arrowLength = 0.8 * a
    const a1End = dataToCanvas({ x: arrowLength * baseBasis.a1.x, y: arrowLength * baseBasis.a1.y })
    const a2End = dataToCanvas({ x: arrowLength * baseBasis.a2.x, y: arrowLength * baseBasis.a2.y })
    
    // Draw a1 arrow (red)
    ctx.strokeStyle = '#dc2626'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(originCanvas.x, originCanvas.y)
    ctx.lineTo(a1End.x, a1End.y)
    ctx.stroke()
    drawArrowhead(ctx, originCanvas, a1End, '#dc2626')
    
    // Draw a2 arrow (orange)
    ctx.strokeStyle = '#f97316'
    ctx.beginPath()
    ctx.moveTo(originCanvas.x, originCanvas.y)
    ctx.lineTo(a2End.x, a2End.y)
    ctx.stroke()
    drawArrowhead(ctx, originCanvas, a2End, '#f97316')
    
    // Labels for basis vectors
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = '#dc2626'
    ctx.fillText('a₁', a1End.x + 4, a1End.y)
    ctx.fillStyle = '#f97316'
    ctx.fillText('a₂', a2End.x + 4, a2End.y - 4)
    
    // Draw title
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('Bilayer at R (Zoomed)', width / 2, 18)
    
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText(`R = (${zoomPosition.x.toFixed(2)}a, ${zoomPosition.y.toFixed(2)}a)`, width / 2, 32)
    
    // Draw axis labels
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.font = '11px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('x / a', width / 2, height - 8)
    
    ctx.save()
    ctx.translate(12, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('y / a', 0, 0)
    ctx.restore()
    
    // Legend
    const legendX = padding.left + 8
    const legendY = padding.top + 12
    const legendSpacing = 16
    
    ctx.font = '10px system-ui, sans-serif'
    ctx.textAlign = 'left'
    
    ctx.strokeStyle = LATTICE_COLORS.layer1
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(legendX, legendY, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('L1 holes', legendX + 12, legendY + 3)
    
    ctx.strokeStyle = LATTICE_COLORS.layer2
    ctx.fillStyle = 'rgba(251, 146, 60, 0.15)'
    ctx.beginPath()
    ctx.arc(legendX, legendY + legendSpacing, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('L2 holes', legendX + 12, legendY + legendSpacing + 3)
    
  }, [geometry, width, height, thetaDeg, rOverA, a, viewRadius, zoomPosition])
  
  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        borderRadius: '8px',
        border: `1px solid ${LATTICE_COLORS.border}`,
        background: LATTICE_COLORS.background,
      }}
    />
  )
}

// Helper to draw arrowhead
function drawArrowhead(
  ctx: CanvasRenderingContext2D,
  from: Vec2,
  to: Vec2,
  color: string,
  size: number = 8
) {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const angle = Math.atan2(dy, dx)
  
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(to.x, to.y)
  ctx.lineTo(
    to.x - size * Math.cos(angle - Math.PI / 7),
    to.y - size * Math.sin(angle - Math.PI / 7)
  )
  ctx.lineTo(
    to.x - size * Math.cos(angle + Math.PI / 7),
    to.y - size * Math.sin(angle + Math.PI / 7)
  )
  ctx.closePath()
  ctx.fill()
}

export default BilayerZoomCanvas
