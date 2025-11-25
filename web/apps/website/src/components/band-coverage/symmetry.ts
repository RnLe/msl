import type { LatticeType } from './types'

export type SymmetryStop = {
  label: string
  position: number
  fractional: [number, number]
}

type SymmetryStopDefinition = {
  label: string
  fractional: [number, number]
}

const PATH_DEFINITIONS: Record<string, SymmetryStopDefinition[]> = {
  square: [
    { label: 'Γ', fractional: [0, 0] },
    { label: 'X', fractional: [0.5, 0] },
    { label: 'M', fractional: [0.5, 0.5] },
    { label: 'Γ', fractional: [0, 0] },
  ],
  hex: [
    { label: 'Γ', fractional: [0, 0] },
    { label: 'K', fractional: [1 / 3, 1 / 3] },
    { label: 'M', fractional: [0.5, 0] },
    { label: 'Γ', fractional: [0, 0] },
  ],
}

const PATH_CACHE = new Map<string, SymmetryStop[]>()

function normalizeLatticeKey(lattice: LatticeType): string {
  const key = typeof lattice === 'string' ? lattice.toLowerCase() : ''
  if (key === 'hex' || key === 'hexagonal' || key === 'triangular') return 'hex'
  return 'square'
}

function segmentLength(a: [number, number], b: [number, number]) {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

function buildPathDefinition(lattice: LatticeType): SymmetryStop[] {
  const key = normalizeLatticeKey(lattice)
  const cached = PATH_CACHE.get(key)
  if (cached) return cached

  const definition = PATH_DEFINITIONS[key] ?? PATH_DEFINITIONS.square

  if (!definition.length) {
    const fallback: SymmetryStop[] = [{ label: 'Γ', position: 0, fractional: [0, 0] }]
    PATH_CACHE.set(key, fallback)
    return fallback
  }

  let totalLength = 0
  const segments: number[] = []
  for (let idx = 0; idx < definition.length - 1; idx += 1) {
    const current = definition[idx].fractional
    const next = definition[idx + 1].fractional
    const length = segmentLength(current, next)
    segments.push(length)
    totalLength += length
  }

  let cumulative = 0
  const stepDenominator = definition.length > 1 ? definition.length - 1 : 1

  const stops: SymmetryStop[] = definition.map((stop, idx) => {
    if (idx === 0) {
      return { label: stop.label, position: 0, fractional: stop.fractional }
    }
    if (totalLength <= 0) {
      const stepRatio = idx / stepDenominator
      return { label: stop.label, position: Math.min(1, stepRatio), fractional: stop.fractional }
    }
    cumulative += segments[idx - 1] ?? 0
    const ratio = cumulative / totalLength
    return { label: stop.label, position: Math.min(1, ratio), fractional: stop.fractional }
  })

  if (stops.length) {
    stops[0] = { ...stops[0], position: 0 }
    stops[stops.length - 1] = { ...stops[stops.length - 1], position: 1 }
  }

  PATH_CACHE.set(key, stops)
  return stops
}

export function getPathDefinition(lattice: LatticeType) {
  return buildPathDefinition(lattice)
}

export function getSymmetryStops(lattice: LatticeType) {
  return getPathDefinition(lattice)
}

export function formatSymmetryLabel(lattice: LatticeType, label: string) {
  const key = normalizeLatticeKey(lattice)
  if (key === 'hex') {
    if (label === 'K') return 'M'
    if (label === 'M') return 'K'
  }
  return label
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
