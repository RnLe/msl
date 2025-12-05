'use client'

import React, { useRef, useEffect, useMemo } from 'react'
import {
  getBasisVectors,
  computeRegistryShift,
  cartesianToFractional,
  registryToAtomPosition,
  computeMoireLength,
  LATTICE_COLORS,
  normalizeLatticeType,
  type LatticeType,
  type Vec2,
} from './lattice-utils'

interface RegistryShiftCanvasProps {
  latticeType: LatticeType
  thetaDeg: number
  /** Lattice constant (default 1.0) */
  a?: number
  /** Position in moiré coordinates (default: origin) */
  moirePosition?: Vec2
  /** Hole radius / a for visualization */
  rOverA?: number
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
 * Canvas-based visualization of the 2-atomic basis in fractional coordinates.
 * 
 * Shows the atom positions in the unit cell [0,1)² with proper ellipse
 * transformation for non-orthogonal lattice bases.
 */
export function RegistryShiftCanvas({
  latticeType,
  thetaDeg,
  a = 1.0,
  moirePosition = { x: 0, y: 0 },
  rOverA = 0.3,
  width = 320,
  height = 320,
  eta: etaProp,
  tau = { x: 0, y: 0 },
}: RegistryShiftCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  const normalizedType = normalizeLatticeType(latticeType)
  const thetaRad = (thetaDeg * Math.PI) / 180
  
  // Compute all the shift-related quantities
  const shiftData = useMemo(() => {
    const baseBasis = getBasisVectors(normalizedType)
    
    // Scale by lattice constant
    const basis = {
      a1: { x: baseBasis.a1.x * a, y: baseBasis.a1.y * a },
      a2: { x: baseBasis.a2.x * a, y: baseBasis.a2.y * a },
    }
    
    const moireLength = computeMoireLength(a, thetaRad)
    const eta = etaProp ?? (a / moireLength)
    
    // Compute theoretical shift in Cartesian
    const deltaCartesian = computeRegistryShift(moirePosition, thetaRad, eta, tau)
    
    // Convert to fractional coordinates
    const deltaFrac = cartesianToFractional(deltaCartesian, basis)
    
    // What we actually send to BLAZE: wrap to [0, 1) for atom positions
    const atom0Pos = { x: 0, y: 0 }  // Fixed atom at origin
    const atom1Pos = registryToAtomPosition(deltaFrac)  // Swept atom
    
    return {
      basis,
      baseBasis,
      moireLength,
      eta,
      deltaCartesian,
      deltaFrac,
      atom0Pos,
      atom1Pos,
    }
  }, [normalizedType, thetaRad, a, moirePosition, etaProp, tau])
  
  // Render to canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { baseBasis, atom0Pos, atom1Pos } = shiftData
    
    // Clear canvas with dark background
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.fillRect(0, 0, width, height)
    
    // Make the plot area square and centered
    const padding = { left: 50, right: 20, top: 35, bottom: 45 }
    const availableWidth = width - padding.left - padding.right
    const availableHeight = height - padding.top - padding.bottom
    const plotSize = Math.min(availableWidth, availableHeight)
    
    // Center the square plot
    const offsetX = padding.left + (availableWidth - plotSize) / 2
    const offsetY = padding.top + (availableHeight - plotSize) / 2
    
    // Extent for unit cell view [0, 1] x [0, 1] with small margin
    const margin = 0.1
    const extent = { xMin: -margin, xMax: 1 + margin, yMin: -margin, yMax: 1 + margin }
    
    const dataToCanvas = (p: Vec2): Vec2 => {
      const x = offsetX + ((p.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotSize
      const y = offsetY + (1 - (p.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotSize
      return { x, y }
    }
    
    // Scale factor: canvas pixels per fractional unit
    const scale = plotSize / (extent.xMax - extent.xMin)
    
    // Draw unit cell box (dashed)
    ctx.strokeStyle = LATTICE_COLORS.axis
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 3])
    const corners = [
      dataToCanvas({ x: 0, y: 0 }),
      dataToCanvas({ x: 1, y: 0 }),
      dataToCanvas({ x: 1, y: 1 }),
      dataToCanvas({ x: 0, y: 1 }),
    ]
    ctx.beginPath()
    ctx.moveTo(corners[0].x, corners[0].y)
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(corners[i].x, corners[i].y)
    }
    ctx.closePath()
    ctx.stroke()
    ctx.setLineDash([])
    
    // ============ Ellipse transformation ============
    // A circle in Cartesian coords with radius r becomes an ellipse in fractional coords.
    // The transformation from Cartesian to fractional is: frac = B^{-1} · cart
    // where B = [a1 | a2] is the lattice matrix.
    //
    // For a circle parameterized as cart(t) = r·[cos(t), sin(t)],
    // in fractional coords: frac(t) = B^{-1} · r·[cos(t), sin(t)]
    //
    // This is an ellipse. We compute B^{-1} and use canvas transform to draw it.
    
    const B = {
      a: baseBasis.a1.x, b: baseBasis.a2.x,
      c: baseBasis.a1.y, d: baseBasis.a2.y,
    }
    const det = B.a * B.d - B.b * B.c
    
    // B^{-1} transforms Cartesian -> fractional
    const Binv = {
      a: B.d / det, b: -B.b / det,
      c: -B.c / det, d: B.a / det,
    }
    
    // The circle radius in Cartesian is rOverA * a, but since baseBasis is in units of a=1,
    // the radius is just rOverA in these units.
    const cartesianRadius = rOverA
    
    // Helper to draw an ellipse using canvas transform
    // We draw a unit circle and apply the B^{-1} transformation scaled by radius
    const drawTransformedCircle = (centerFrac: Vec2) => {
      const centerCanvas = dataToCanvas(centerFrac)
      
      ctx.save()
      ctx.translate(centerCanvas.x, centerCanvas.y)
      
      // Apply B^{-1} transformation (but in canvas coords where y is flipped)
      // The transform matrix for canvas is [a, c, b, d, 0, 0] but we need to account for y-flip
      // Since canvas y increases downward, we negate the y components
      ctx.transform(
        Binv.a * scale * cartesianRadius,  // a: scale x by Binv row 1
        -Binv.c * scale * cartesianRadius, // b: (note: canvas transform uses column-major-ish order)
        Binv.b * scale * cartesianRadius,  // c
        -Binv.d * scale * cartesianRadius, // d: scale y by Binv row 2 (negated for y-flip)
        0, 0
      )
      
      // Draw unit circle (which becomes the transformed ellipse)
      ctx.beginPath()
      ctx.arc(0, 0, 1, 0, Math.PI * 2)
      
      ctx.restore()
    }
    
    // Atom 0 (fixed, blue) at origin
    ctx.fillStyle = LATTICE_COLORS.layer1
    drawTransformedCircle(atom0Pos)
    ctx.fill()
    ctx.strokeStyle = '#60a5fa'
    ctx.lineWidth = 2
    drawTransformedCircle(atom0Pos)
    ctx.stroke()
    
    // Atom 1 (swept, orange)
    ctx.fillStyle = LATTICE_COLORS.layer2
    drawTransformedCircle(atom1Pos)
    ctx.fill()
    ctx.strokeStyle = '#fdba74'
    ctx.lineWidth = 2
    drawTransformedCircle(atom1Pos)
    ctx.stroke()
    
    // Draw arrow from atom0 to atom1 showing the shift
    const a0 = dataToCanvas(atom0Pos)
    const a1 = dataToCanvas(atom1Pos)
    const arrowLen = Math.sqrt((a1.x - a0.x) ** 2 + (a1.y - a0.y) ** 2)
    if (arrowLen > 5) {
      ctx.strokeStyle = LATTICE_COLORS.text
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(a0.x, a0.y)
      ctx.lineTo(a1.x, a1.y)
      ctx.stroke()
      drawArrowhead(ctx, a0, a1, LATTICE_COLORS.text, 10)
    }
    
    // Labels on atoms
    ctx.font = 'bold 12px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillStyle = 'white'
    ctx.fillText('0', a0.x, a0.y + 4)
    ctx.fillText('1', a1.x, a1.y + 4)
    
    // Title - bold and larger
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 16px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('2-Atomic Basis (fractional)', width / 2, 22)
    
    // Axis labels - bold with subscript notation
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.textAlign = 'center'
    ctx.fillText('δ₁', width / 2, height - 8)
    
    ctx.save()
    ctx.translate(16, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('δ₂', 0, 0)
    ctx.restore()
    
    // Tick marks and labels
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.strokeStyle = LATTICE_COLORS.axis
    ctx.lineWidth = 1
    
    // X-axis ticks (0, 0.5, 1)
    for (const val of [0, 0.5, 1]) {
      const p = dataToCanvas({ x: val, y: 0 })
      ctx.beginPath()
      ctx.moveTo(p.x, offsetY + plotSize)
      ctx.lineTo(p.x, offsetY + plotSize + 5)
      ctx.stroke()
      ctx.textAlign = 'center'
      ctx.fillText(val.toString(), p.x, offsetY + plotSize + 16)
    }
    
    // Y-axis ticks (0, 0.5, 1)
    for (const val of [0, 0.5, 1]) {
      const p = dataToCanvas({ x: 0, y: val })
      ctx.beginPath()
      ctx.moveTo(offsetX - 5, p.y)
      ctx.lineTo(offsetX, p.y)
      ctx.stroke()
      ctx.textAlign = 'right'
      ctx.fillText(val.toString(), offsetX - 8, p.y + 4)
    }
    
  }, [shiftData, width, height, rOverA])
  
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

export default RegistryShiftCanvas
