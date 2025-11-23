import type { LatticeType } from './types'

export type SymmetryStop = {
  label: string
  position: number
  fractional: [number, number]
}

const PATH_DEFINITIONS: Record<string, SymmetryStop[]> = {
  square: [
    { label: 'Γ', position: 0, fractional: [0, 0] },
    { label: 'X', position: 1 / 3, fractional: [0.5, 0] },
    { label: 'M', position: 2 / 3, fractional: [0.5, 0.5] },
    { label: 'Γ', position: 1, fractional: [0, 0] },
  ],
  hex: [
    { label: 'Γ', position: 0, fractional: [0, 0] },
    { label: 'M', position: 1 / 3, fractional: [0.5, 0] },
    { label: 'K', position: 2 / 3, fractional: [1 / 3, 1 / 3] },
    { label: 'Γ', position: 1, fractional: [0, 0] },
  ],
}

export function getPathDefinition(lattice: LatticeType) {
  return PATH_DEFINITIONS[lattice] ?? PATH_DEFINITIONS.square
}

export function getSymmetryStops(lattice: LatticeType) {
  return getPathDefinition(lattice)
}

export function findNearestStop(stops: { label: string; position: number }[], ratio: number) {
  if (!stops.length) return { label: 'Γ', position: 0 }
  return stops.reduce((nearest, stop) => {
    return Math.abs(stop.position - ratio) < Math.abs(nearest.position - ratio) ? stop : nearest
  })
}

export function interpolateFractionalCoordinate(lattice: LatticeType, ratio: number) {
  const stops = getPathDefinition(lattice)
  if (!stops.length) return null
  const clamped = Math.min(1, Math.max(0, ratio))
  for (let idx = 0; idx < stops.length - 1; idx += 1) {
    const current = stops[idx]
    const next = stops[idx + 1]
    if (clamped >= current.position && clamped <= next.position) {
      const span = next.position - current.position || 1
      const t = (clamped - current.position) / span
      return {
        fractional: [
          current.fractional[0] + t * (next.fractional[0] - current.fractional[0]),
          current.fractional[1] + t * (next.fractional[1] - current.fractional[1]),
        ] as [number, number],
      }
    }
  }
  const last = stops[stops.length - 1]
  return { fractional: [...last.fractional] as [number, number] }
}

export function getUniqueHighSymmetryPoints(lattice: LatticeType) {
  const stops = getPathDefinition(lattice)
  const unique = new Map<string, [number, number]>()
  for (const stop of stops) {
    if (!unique.has(stop.label)) {
      unique.set(stop.label, stop.fractional)
    }
  }
  return Array.from(unique, ([label, fractional]) => ({ label, fractional }))
}

export function getPathLabels(lattice: LatticeType) {
  return getPathDefinition(lattice).map((stop) => stop.label)
}

export function getFallbackBrillouinZoneFractionalVertices(lattice: LatticeType) {
  if (lattice === 'hex' || lattice === 'hexagonal') {
    return [
      [0, 2 / 3],
      [1 / 3, 1 / 3],
      [2 / 3, -1 / 3],
      [0, -2 / 3],
      [-1 / 3, -1 / 3],
      [-2 / 3, 1 / 3],
    ] as [number, number][]
  }
  return [
    [0.5, 0.5],
    [0.5, -0.5],
    [-0.5, -0.5],
    [-0.5, 0.5],
  ] as [number, number][]
}
