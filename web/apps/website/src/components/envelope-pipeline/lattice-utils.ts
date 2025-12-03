/**
 * Lattice geometry utilities for visualization.
 * 
 * CONVENTION NOTE (as of 2024-12):
 * - Triangular/hex lattice uses 60° basis: a1 = [1, 0], a2 = [0.5, √3/2]
 * - This matches MPB convention and the updated BLAZE convention
 * - Square lattice: a1 = [1, 0], a2 = [0, 1]
 */

export type LatticeType = 'hex' | 'triangular' | 'square' | 'rect'

export interface Vec2 {
  x: number
  y: number
}

export interface BasisVectors {
  a1: Vec2
  a2: Vec2
}

export interface LatticeParams {
  latticeType: LatticeType
  thetaDeg: number
  a: number  // lattice constant
  rOverA?: number  // hole radius / a
}

/**
 * Get basis vectors for a given lattice type (unrotated, in units of a=1)
 * 
 * 60° convention for triangular: a1 = [1, 0], a2 = [0.5, √3/2]
 */
export function getBasisVectors(latticeType: LatticeType): BasisVectors {
  switch (latticeType) {
    case 'hex':
    case 'triangular':
      // 60° convention (MPB/BLAZE unified)
      return {
        a1: { x: 1, y: 0 },
        a2: { x: 0.5, y: Math.sqrt(3) / 2 },
      }
    case 'square':
      return {
        a1: { x: 1, y: 0 },
        a2: { x: 0, y: 1 },
      }
    case 'rect':
      // Default rectangular with aspect ratio 1.5
      return {
        a1: { x: 1, y: 0 },
        a2: { x: 0, y: 1.5 },
      }
    default:
      return {
        a1: { x: 1, y: 0 },
        a2: { x: 0, y: 1 },
      }
  }
}

/**
 * Rotate a 2D vector by angle (radians)
 */
export function rotateVec(v: Vec2, theta: number): Vec2 {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return {
    x: c * v.x - s * v.y,
    y: s * v.x + c * v.y,
  }
}

/**
 * Rotate basis vectors by angle (radians)
 */
export function rotateBasis(basis: BasisVectors, theta: number): BasisVectors {
  return {
    a1: rotateVec(basis.a1, theta),
    a2: rotateVec(basis.a2, theta),
  }
}

/**
 * Compute moiré length scale: L_m = a / (2 * sin(θ/2))
 */
export function computeMoireLength(a: number, thetaRad: number): number {
  if (Math.abs(thetaRad) < 1e-9) return Infinity
  return a / (2 * Math.sin(Math.abs(thetaRad) / 2))
}

/**
 * Generate lattice points within a bounding box.
 * Returns fractional coordinates (i, j) and Cartesian positions.
 */
export function generateLatticePoints(
  basis: BasisVectors,
  extent: { xMin: number; xMax: number; yMin: number; yMax: number },
  padding: number = 0.1
): Array<{ i: number; j: number; x: number; y: number }> {
  const points: Array<{ i: number; j: number; x: number; y: number }> = []
  
  const { xMin, xMax, yMin, yMax } = extent
  const width = xMax - xMin
  const height = yMax - yMin
  const padX = width * padding
  const padY = height * padding
  
  // Compute inverse basis matrix for fractional coords
  const det = basis.a1.x * basis.a2.y - basis.a1.y * basis.a2.x
  if (Math.abs(det) < 1e-12) return points
  
  const invDet = 1 / det
  const B_inv = {
    a: basis.a2.y * invDet,
    b: -basis.a2.x * invDet,
    c: -basis.a1.y * invDet,
    d: basis.a1.x * invDet,
  }
  
  // Find range of i, j indices to cover the extent
  const corners = [
    { x: xMin - padX, y: yMin - padY },
    { x: xMax + padX, y: yMin - padY },
    { x: xMin - padX, y: yMax + padY },
    { x: xMax + padX, y: yMax + padY },
  ]
  
  let iMin = Infinity, iMax = -Infinity
  let jMin = Infinity, jMax = -Infinity
  
  for (const c of corners) {
    const i = B_inv.a * c.x + B_inv.b * c.y
    const j = B_inv.c * c.x + B_inv.d * c.y
    iMin = Math.min(iMin, i)
    iMax = Math.max(iMax, i)
    jMin = Math.min(jMin, j)
    jMax = Math.max(jMax, j)
  }
  
  iMin = Math.floor(iMin) - 1
  iMax = Math.ceil(iMax) + 1
  jMin = Math.floor(jMin) - 1
  jMax = Math.ceil(jMax) + 1
  
  for (let i = iMin; i <= iMax; i++) {
    for (let j = jMin; j <= jMax; j++) {
      const x = i * basis.a1.x + j * basis.a2.x
      const y = i * basis.a1.y + j * basis.a2.y
      
      if (x >= xMin - padX && x <= xMax + padX &&
          y >= yMin - padY && y <= yMax + padY) {
        points.push({ i, j, x, y })
      }
    }
  }
  
  return points
}

/**
 * Compute the registry shift δ(R) at a moiré position R.
 * 
 * δ(R) = (R(θ) - I) @ R / η + τ
 * 
 * This is in Cartesian coordinates. To get fractional coordinates,
 * multiply by the inverse lattice matrix.
 * 
 * @param R - Position in moiré coordinates (Cartesian)
 * @param thetaRad - Twist angle in radians
 * @param eta - Small parameter (typically a/L_m)
 * @param tau - Stacking gauge vector (default: [0, 0])
 */
export function computeRegistryShift(
  R: Vec2,
  thetaRad: number,
  eta: number = 1.0,
  tau: Vec2 = { x: 0, y: 0 }
): Vec2 {
  // Rotation matrix R(θ) - I
  const c = Math.cos(thetaRad)
  const s = Math.sin(thetaRad)
  const R_minus_I = {
    a: c - 1, b: -s,
    c: s,     d: c - 1,
  }
  
  // δ = (R(θ) - I) @ R_vec / η + τ
  const deltaX = (R_minus_I.a * R.x + R_minus_I.b * R.y) / eta + tau.x
  const deltaY = (R_minus_I.c * R.x + R_minus_I.d * R.y) / eta + tau.y
  
  return { x: deltaX, y: deltaY }
}

/**
 * Convert Cartesian shift to fractional coordinates.
 */
export function cartesianToFractional(delta: Vec2, basis: BasisVectors): Vec2 {
  const det = basis.a1.x * basis.a2.y - basis.a1.y * basis.a2.x
  if (Math.abs(det) < 1e-12) return { x: 0, y: 0 }
  
  const invDet = 1 / det
  return {
    x: (basis.a2.y * delta.x - basis.a2.x * delta.y) * invDet,
    y: (-basis.a1.y * delta.x + basis.a1.x * delta.y) * invDet,
  }
}

/**
 * Convert fractional coordinates to Cartesian.
 */
export function fractionalToCartesian(frac: Vec2, basis: BasisVectors): Vec2 {
  return {
    x: frac.x * basis.a1.x + frac.y * basis.a2.x,
    y: frac.x * basis.a1.y + frac.y * basis.a2.y,
  }
}

/**
 * Wrap fractional coordinates to [0, 1)
 */
export function wrapToUnitCell(frac: Vec2): Vec2 {
  const wrap = (v: number) => v - Math.floor(v)
  return { x: wrap(frac.x), y: wrap(frac.y) }
}

/**
 * Convert registry shift to BLAZE atom position.
 * 
 * In BLAZE, we have:
 * - Atom 0 (fixed): pos = [0.5, 0.5]
 * - Atom 1 (swept): pos = δ_frac + 0.5 (wrapped to [0, 1))
 * 
 * The relative shift is then atom1.pos - atom0.pos = δ_frac
 */
export function registryToAtomPosition(deltaFrac: Vec2): Vec2 {
  const pos = {
    x: deltaFrac.x + 0.5,
    y: deltaFrac.y + 0.5,
  }
  return wrapToUnitCell(pos)
}

/**
 * Compute the relative shift from two atom positions.
 * This reverses what we put into BLAZE.
 */
export function atomPositionsToShift(atom0: Vec2, atom1: Vec2): Vec2 {
  let dx = atom1.x - atom0.x
  let dy = atom1.y - atom0.y
  
  // Wrap to [-0.5, 0.5) to get the minimal shift
  if (dx > 0.5) dx -= 1
  if (dx < -0.5) dx += 1
  if (dy > 0.5) dy -= 1
  if (dy < -0.5) dy += 1
  
  return { x: dx, y: dy }
}

/**
 * Colors for dark mode visualization
 */
export const LATTICE_COLORS = {
  layer1: '#3b82f6',      // Blue-500
  layer2: '#fb923c',      // Orange-400
  moireNode: '#a855f7',   // Purple-500
  moireCell: '#e2e8f0',   // Slate-200 (light for dark bg)
  axis: '#64748b',        // Slate-500
  background: '#111111',  // Dark gray (requested)
  gridLine: '#27272a',    // Zinc-800
  text: '#f4f4f5',        // Zinc-100
  textMuted: '#a1a1aa',   // Zinc-400
  border: '#3f3f46',      // Zinc-700
  panelBg: '#18181b',     // Zinc-900
  selectedPixel: '#22d3ee', // Cyan-400 for selected pixel marker
} as const

/**
 * Normalize lattice type string
 */
export function normalizeLatticeType(type: string): LatticeType {
  const lower = type.toLowerCase()
  if (lower === 'hex' || lower === 'hexagonal' || lower === 'triangular') {
    return 'triangular'
  }
  if (lower === 'sq' || lower === 'square') {
    return 'square'
  }
  if (lower === 'rect' || lower === 'rectangular') {
    return 'rect'
  }
  return 'square'
}
