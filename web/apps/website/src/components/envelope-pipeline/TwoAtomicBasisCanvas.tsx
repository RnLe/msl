'use client'

import React, { useRef, useEffect, useMemo } from 'react'
import {
  getBasisVectors,
  rotateBasis,
  computeRegistryShift,
  cartesianToFractional,
  computeMoireLength,
  LATTICE_COLORS,
  normalizeLatticeType,
  type LatticeType,
  type BasisVectors,
  type Vec2,
} from './lattice-utils'

interface TwoAtomicBasisCanvasProps {
  latticeType: LatticeType
  thetaDeg: number
  /** Lattice constant (default 1.0) */
  a?: number
  /** Position in moiré coordinates (default: origin) */
  moirePosition?: Vec2
  /** Hole radius / a for visualization */
  rOverA?: number
  /** How many lattice constants to show in each direction (default: 1.5 to match BilayerZoomCanvas) */
  viewRadius?: number
  /** Width of the canvas in pixels */
  width?: number
  /** Height of the canvas in pixels */
  height?: number
  /** Custom eta parameter (default: a/L_m) */
  eta?: number
  /** Stacking gauge τ (default: [0, 0]) */
  tau?: Vec2
}

/**
 * Canvas-based visualization of the 2-atomic basis at a given moiré position.
 * 
 * Shows the local bilayer structure with:
 * - Layer 1 atom at origin (blue)
 * - Layer 2 atom shifted by δ(R) (orange)
 * - Unit cell parallelogram for context
 * 
 * This is the "physical" view complementing the RegistryShiftCanvas's
 * fractional coordinate view.
 */
export function TwoAtomicBasisCanvas({
  latticeType,
  thetaDeg,
  a = 1.0,
  moirePosition = { x: 0, y: 0 },
  rOverA = 0.3,
  viewRadius = 1.5,
  width = 320,
  height = 320,
  eta: etaProp,
  tau = { x: 0, y: 0 },
}: TwoAtomicBasisCanvasProps) {
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
    
    // Moiré length scale
    const moireLength = computeMoireLength(a, thetaRad)
    const eta = etaProp ?? (a / moireLength)
    
    // Compute registry shift at this position
    const deltaCartesian = computeRegistryShift(moirePosition, thetaRad, eta, tau)
    const deltaFrac = cartesianToFractional(deltaCartesian, layer1Basis)
    
    return {
      layer1Basis,
      layer2Basis,
      baseBasis,
      moireLength,
      eta,
      deltaCartesian,
      deltaFrac,
    }
  }, [normalizedType, thetaRad, a, moirePosition, etaProp, tau])
  
  // Render to canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { layer1Basis, layer2Basis, baseBasis, deltaCartesian, deltaFrac } = geometry
    
    // Clear canvas with dark background
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.fillRect(0, 0, width, height)
    
    // View extent: same as BilayerZoomCanvas (viewRadius * a centered at origin)
    const halfL = viewRadius * a
    const extent = {
      xMin: -halfL,
      xMax: halfL,
      yMin: -halfL,
      yMax: halfL,
    }
    
    // Padding for labels
    const padding = { left: 45, right: 20, top: 40, bottom: 40 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom
    
    // Transform from data coords to canvas coords
    const dataToCanvas = (p: Vec2): Vec2 => {
      const x = padding.left + ((p.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotWidth
      const y = padding.top + (1 - (p.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotHeight
      return { x, y }
    }
    
    // Draw grid (at lattice constant intervals, same as BilayerZoomCanvas)
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
    ctx.beginPath()
    ctx.moveTo(padding.left, originCanvas.y)
    ctx.lineTo(padding.left + plotWidth, originCanvas.y)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(originCanvas.x, padding.top)
    ctx.lineTo(originCanvas.x, padding.top + plotHeight)
    ctx.stroke()
    
    // Draw Layer 1 unit cell parallelogram (dashed)
    ctx.strokeStyle = LATTICE_COLORS.layer1
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 3])
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
    ctx.setLineDash([])
    ctx.globalAlpha = 1.0
    
    // Draw atom radius
    const dataRadius = rOverA * a
    const canvasRadius = (dataRadius / (extent.xMax - extent.xMin)) * plotWidth
    const radius = Math.max(8, Math.min(50, canvasRadius))
    
    // Draw Layer 1 atom at origin (blue, with hole effect)
    const atom0 = dataToCanvas({ x: 0, y: 0 })
    ctx.fillStyle = '#18181b'  // Dark fill (hole in dielectric)
    ctx.strokeStyle = LATTICE_COLORS.layer1
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(atom0.x, atom0.y, radius, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    
    // Draw Layer 2 atom at δ (orange, with hole effect)
    const atom1 = dataToCanvas(deltaCartesian)
    ctx.fillStyle = 'rgba(251, 146, 60, 0.15)'  // Transparent orange
    ctx.strokeStyle = LATTICE_COLORS.layer2
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(atom1.x, atom1.y, radius, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    
    // Draw shift arrow from atom0 to atom1
    const arrowLen = Math.sqrt(
      (atom1.x - atom0.x) ** 2 + (atom1.y - atom0.y) ** 2
    )
    if (arrowLen > 5) {
      ctx.strokeStyle = LATTICE_COLORS.moireNode
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(atom0.x, atom0.y)
      ctx.lineTo(atom1.x, atom1.y)
      ctx.stroke()
      drawArrowhead(ctx, atom0, atom1, LATTICE_COLORS.moireNode, 10)
    }
    
    // Draw basis vector arrows from origin
    const arrowScale = 0.7
    const a1End = dataToCanvas({ 
      x: arrowScale * baseBasis.a1.x * a, 
      y: arrowScale * baseBasis.a1.y * a 
    })
    const a2End = dataToCanvas({ 
      x: arrowScale * baseBasis.a2.x * a, 
      y: arrowScale * baseBasis.a2.y * a 
    })
    
    // a1 arrow (red-ish)
    ctx.strokeStyle = '#ef4444'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(originCanvas.x, originCanvas.y)
    ctx.lineTo(a1End.x, a1End.y)
    ctx.stroke()
    drawArrowhead(ctx, originCanvas, a1End, '#ef4444', 8)
    
    // a2 arrow (yellow-ish)
    ctx.strokeStyle = '#eab308'
    ctx.beginPath()
    ctx.moveTo(originCanvas.x, originCanvas.y)
    ctx.lineTo(a2End.x, a2End.y)
    ctx.stroke()
    drawArrowhead(ctx, originCanvas, a2End, '#eab308', 8)
    
    // Labels for basis vectors
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = '#ef4444'
    ctx.fillText('a₁', a1End.x + 6, a1End.y + 4)
    ctx.fillStyle = '#eab308'
    ctx.fillText('a₂', a2End.x + 4, a2End.y - 6)
    
    // Draw title
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('2-Atomic Basis at R', width / 2, 18)
    
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText(`δ = (${deltaFrac.x.toFixed(3)}, ${deltaFrac.y.toFixed(3)}) frac`, width / 2, 34)
    
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
    
    // Layer 1 marker
    ctx.strokeStyle = LATTICE_COLORS.layer1
    ctx.lineWidth = 2
    ctx.fillStyle = '#18181b'
    ctx.beginPath()
    ctx.arc(legendX, legendY, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('L1 (origin)', legendX + 12, legendY + 3)
    
    // Layer 2 marker
    ctx.strokeStyle = LATTICE_COLORS.layer2
    ctx.fillStyle = 'rgba(251, 146, 60, 0.15)'
    ctx.beginPath()
    ctx.arc(legendX, legendY + legendSpacing, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('L2 (shifted)', legendX + 12, legendY + legendSpacing + 3)
    
  }, [geometry, width, height, thetaDeg, rOverA, a, moirePosition, viewRadius])
  
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
  const len = Math.sqrt(dx * dx + dy * dy)
  if (len < 1) return
  
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

export default TwoAtomicBasisCanvas
