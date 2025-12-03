'use client'

import React, { useRef, useEffect, useMemo } from 'react'
import {
  getBasisVectors,
  computeRegistryShift,
  cartesianToFractional,
  fractionalToCartesian,
  registryToAtomPosition,
  computeMoireLength,
  LATTICE_COLORS,
  normalizeLatticeType,
  type LatticeType,
  type BasisVectors,
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
 * Canvas-based visualization of the registry shift δ(R).
 * 
 * Shows:
 * 1. The theoretical shift δ(R) = (R(θ) - I) · R / η + τ
 * 2. The fractional coordinates δ_frac sent to BLAZE
 * 3. The resulting atom positions in the 2-atom basis
 * 
 * This is the key diagnostic for debugging lattice conventions!
 */
export function RegistryShiftCanvas({
  latticeType,
  thetaDeg,
  a = 1.0,
  moirePosition = { x: 0, y: 0 },
  rOverA = 0.3,
  width = 400,
  height = 450,
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
    const basis: BasisVectors = {
      a1: { x: baseBasis.a1.x * a, y: baseBasis.a1.y * a },
      a2: { x: baseBasis.a2.x * a, y: baseBasis.a2.y * a },
    }
    
    const moireLength = computeMoireLength(a, thetaRad)
    const eta = etaProp ?? (a / moireLength)  // Physical eta
    
    // Compute theoretical shift in Cartesian
    const deltaCartesian = computeRegistryShift(moirePosition, thetaRad, eta, tau)
    
    // Convert to fractional coordinates
    const deltaFrac = cartesianToFractional(deltaCartesian, basis)
    
    // What we actually send to BLAZE: wrap to [0, 1) and add 0.5 for atom positions
    const atom0Pos = { x: 0.5, y: 0.5 }  // Fixed atom
    const atom1Pos = registryToAtomPosition(deltaFrac)  // Swept atom
    
    // The effective relative shift (what the solver sees)
    const effectiveShiftFrac = {
      x: atom1Pos.x - atom0Pos.x,
      y: atom1Pos.y - atom0Pos.y,
    }
    // Wrap to minimal representation
    const wrapToMinimal = (v: number) => {
      if (v > 0.5) return v - 1
      if (v < -0.5) return v + 1
      return v
    }
    const minimalShift = {
      x: wrapToMinimal(effectiveShiftFrac.x),
      y: wrapToMinimal(effectiveShiftFrac.y),
    }
    
    // Convert back to Cartesian for display
    const minimalShiftCartesian = fractionalToCartesian(minimalShift, basis)
    
    return {
      basis,
      baseBasis,
      moireLength,
      eta,
      deltaCartesian,
      deltaFrac,
      atom0Pos,
      atom1Pos,
      effectiveShiftFrac,
      minimalShift,
      minimalShiftCartesian,
    }
  }, [normalizedType, thetaRad, a, moirePosition, etaProp, tau])
  
  // Render to canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    const { basis, baseBasis, atom0Pos, atom1Pos, minimalShift, minimalShiftCartesian } = shiftData
    
    // Clear canvas with dark background
    ctx.fillStyle = LATTICE_COLORS.background
    ctx.fillRect(0, 0, width, height)
    
    // ============ TOP SECTION: 2-atomic basis in BLAZE fractional coords ============
    const topHeight = height * 0.55
    const padding = { left: 45, right: 20, top: 45, bottom: 35 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = topHeight - padding.top - padding.bottom
    
    // Extent for unit cell view [0, 1] x [0, 1]
    const extent = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }
    
    const dataToCanvas = (p: Vec2): Vec2 => {
      const x = padding.left + ((p.x - extent.xMin) / (extent.xMax - extent.xMin)) * plotWidth
      const y = padding.top + (1 - (p.y - extent.yMin) / (extent.yMax - extent.yMin)) * plotHeight
      return { x, y }
    }
    
    // Draw unit cell box
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
    
    // Draw grid lines at 0.5
    ctx.strokeStyle = LATTICE_COLORS.gridLine
    ctx.lineWidth = 0.5
    const mid = dataToCanvas({ x: 0.5, y: 0.5 })
    ctx.beginPath()
    ctx.moveTo(corners[0].x, mid.y)
    ctx.lineTo(corners[1].x, mid.y)
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(mid.x, corners[0].y)
    ctx.lineTo(mid.x, corners[3].y)
    ctx.stroke()
    
    // Draw atoms
    const radius = 16
    
    // Atom 0 (fixed, blue)
    const a0 = dataToCanvas(atom0Pos)
    ctx.fillStyle = LATTICE_COLORS.layer1
    ctx.beginPath()
    ctx.arc(a0.x, a0.y, radius, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#60a5fa'  // Lighter blue border
    ctx.lineWidth = 2
    ctx.stroke()
    
    // Atom 1 (swept, orange)
    const a1 = dataToCanvas(atom1Pos)
    ctx.fillStyle = LATTICE_COLORS.layer2
    ctx.beginPath()
    ctx.arc(a1.x, a1.y, radius, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#fdba74'  // Lighter orange border
    ctx.lineWidth = 2
    ctx.stroke()
    
    // Draw arrow from atom0 to atom1 showing the shift
    ctx.strokeStyle = LATTICE_COLORS.text
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(a0.x, a0.y)
    ctx.lineTo(a1.x, a1.y)
    ctx.stroke()
    drawArrowhead(ctx, a0, a1, LATTICE_COLORS.text, 10)
    
    // Labels on atoms
    ctx.font = 'bold 11px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillStyle = 'white'
    ctx.fillText('0', a0.x, a0.y + 4)
    ctx.fillText('1', a1.x, a1.y + 4)
    
    // Title
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('BLAZE 2-Atomic Basis', width / 2, 18)
    
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText('Unit cell fractional coords [0,1)²', width / 2, 34)
    
    // Axis labels
    ctx.font = '11px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.textAlign = 'center'
    ctx.fillText('frac_x', width / 2, topHeight - 5)
    
    ctx.save()
    ctx.translate(12, topHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('frac_y', 0, 0)
    ctx.restore()
    
    // Coordinate annotations
    ctx.font = '9px monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = LATTICE_COLORS.layer1
    ctx.fillText(`Atom 0: (${atom0Pos.x.toFixed(2)}, ${atom0Pos.y.toFixed(2)})`, padding.left, topHeight - 22)
    ctx.fillStyle = LATTICE_COLORS.layer2
    ctx.fillText(`Atom 1: (${atom1Pos.x.toFixed(2)}, ${atom1Pos.y.toFixed(2)})`, padding.left, topHeight - 10)
    
    // ============ BOTTOM SECTION: Info panel ============
    const infoTop = topHeight + 10
    const infoHeight = height - infoTop - 10
    
    ctx.fillStyle = LATTICE_COLORS.panelBg
    ctx.fillRect(10, infoTop, width - 20, infoHeight)
    ctx.strokeStyle = LATTICE_COLORS.border
    ctx.lineWidth = 1
    ctx.strokeRect(10, infoTop, width - 20, infoHeight)
    
    // Info text
    ctx.font = '12px system-ui, sans-serif'
    ctx.fillStyle = LATTICE_COLORS.text
    ctx.textAlign = 'left'
    
    const lineHeight = 16
    let y = infoTop + 18
    
    ctx.font = 'bold 11px system-ui, sans-serif'
    ctx.fillText('Registry Shift δ(R)', 20, y)
    y += lineHeight + 2
    
    ctx.font = '10px monospace'
    ctx.fillStyle = LATTICE_COLORS.textMuted
    
    // Theoretical values
    ctx.fillText(`θ = ${thetaDeg.toFixed(3)}°`, 20, y)
    ctx.fillText(`η = ${shiftData.eta.toExponential(2)}`, 135, y)
    y += lineHeight
    
    ctx.fillText(`R = (${moirePosition.x.toFixed(2)}, ${moirePosition.y.toFixed(2)}) a`, 20, y)
    y += lineHeight + 4
    
    // Shift values
    ctx.fillStyle = LATTICE_COLORS.layer1
    ctx.fillText('δ_cart:', 20, y)
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText(`(${shiftData.deltaCartesian.x.toExponential(2)}, ${shiftData.deltaCartesian.y.toExponential(2)}) a`, 65, y)
    y += lineHeight
    
    ctx.fillStyle = LATTICE_COLORS.layer2
    ctx.fillText('δ_frac:', 20, y)
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText(`(${shiftData.deltaFrac.x.toFixed(4)}, ${shiftData.deltaFrac.y.toFixed(4)})`, 65, y)
    y += lineHeight
    
    ctx.fillStyle = LATTICE_COLORS.moireNode
    ctx.fillText('δ_eff:', 20, y)
    ctx.fillStyle = LATTICE_COLORS.textMuted
    ctx.fillText(`(${minimalShift.x.toFixed(4)}, ${minimalShift.y.toFixed(4)})`, 65, y)
    
  }, [shiftData, width, height, thetaDeg, moirePosition])
  
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
