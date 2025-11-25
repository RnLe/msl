import type { LatticeConstantPrefix } from './store'
import { LATTICE_CONSTANT_PREFIXES } from './store'

export const SPEED_OF_LIGHT = 299_792_458 // m/s

const LENGTH_PREFIX_FACTORS: Record<LatticeConstantPrefix, number> = {
  fm: 1e-15,
  pm: 1e-12,
  nm: 1e-9,
  μm: 1e-6,
  mm: 1e-3,
  cm: 1e-2,
  dm: 1e-1,
  m: 1,
}

const LENGTH_PREFIX_ENTRIES = [
  { prefix: 'fm', factor: 1e-15 },
  { prefix: 'pm', factor: 1e-12 },
  { prefix: 'nm', factor: 1e-9 },
  { prefix: 'μm', factor: 1e-6 },
  { prefix: 'mm', factor: 1e-3 },
  { prefix: 'cm', factor: 1e-2 },
  { prefix: 'dm', factor: 1e-1 },
  { prefix: 'm', factor: 1 },
  { prefix: 'km', factor: 1e3 },
]

const FREQUENCY_PREFIXES = [
  { unit: 'pHz', factor: 1e-12 },
  { unit: 'nHz', factor: 1e-9 },
  { unit: 'µHz', factor: 1e-6 },
  { unit: 'mHz', factor: 1e-3 },
  { unit: 'Hz', factor: 1 },
  { unit: 'kHz', factor: 1e3 },
  { unit: 'MHz', factor: 1e6 },
  { unit: 'GHz', factor: 1e9 },
  { unit: 'THz', factor: 1e12 },
  { unit: 'PHz', factor: 1e15 },
]

export function resolveLatticeConstantMeters(value: number | null, prefix: LatticeConstantPrefix | null): number | null {
  if (value === null || !prefix || !Number.isFinite(value) || value <= 0) {
    return null
  }
  const factor = LENGTH_PREFIX_FACTORS[prefix]
  return factor ? value * factor : null
}

export function formatLengthWithPrefix(value: number, prefix: LatticeConstantPrefix): string {
  return `${formatNumberForDisplay(value)} ${prefix}`
}

export function formatLengthWithBestPrefix(meters: number | null): string | null {
  if (!meters || !Number.isFinite(meters) || meters <= 0) return null
  const absValue = Math.abs(meters)
  let chosen = LENGTH_PREFIX_ENTRIES[0]
  for (const entry of LENGTH_PREFIX_ENTRIES) {
    if (absValue >= entry.factor) {
      chosen = entry
    } else {
      break
    }
  }
  const scaled = absValue / chosen.factor
  return `${formatNumberForDisplay(scaled)} ${chosen.prefix}`
}

export function formatPhysicalRadius(rOverA: number | null | undefined, latticeValue: number | null, prefix: LatticeConstantPrefix | null): string | null {
  if (!Number.isFinite(rOverA) || latticeValue === null || !prefix) {
    return null
  }
  return formatLengthWithPrefix((rOverA as number) * latticeValue, prefix)
}

export function normalizedValueToFrequency(normalizedValue: number | null | undefined, latticeConstantMeters: number | null): number | null {
  if (!Number.isFinite(normalizedValue) || !latticeConstantMeters || latticeConstantMeters <= 0) {
    return null
  }
  return (normalizedValue as number) * (SPEED_OF_LIGHT / latticeConstantMeters)
}

export function describeFrequency(valueHz: number | null) {
  if (!valueHz || !Number.isFinite(valueHz) || valueHz <= 0) {
    return null
  }
  const absValue = Math.abs(valueHz)
  let chosen = FREQUENCY_PREFIXES[0]
  for (const entry of FREQUENCY_PREFIXES) {
    if (absValue >= entry.factor) {
      chosen = entry
    } else {
      break
    }
  }
  const scaled = absValue / chosen.factor
  return {
    value: formatNumberForDisplay(scaled),
    unit: chosen.unit,
  }
}

function formatNumberForDisplay(value: number): string {
  const absValue = Math.abs(value)
  if (absValue >= 100) return value.toFixed(0)
  if (absValue >= 10) return value.toFixed(1)
  if (absValue >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

export function metersToPrefixedLattice(meters: number): { value: number; prefix: LatticeConstantPrefix } | null {
  if (!Number.isFinite(meters) || meters <= 0) return null
  const orderedPrefixes = [...LATTICE_CONSTANT_PREFIXES].reverse()
  for (const prefix of orderedPrefixes) {
    const factor = LENGTH_PREFIX_FACTORS[prefix]
    if (!factor) continue
    const rawValue = meters / factor
    if (rawValue >= 1 && rawValue <= 999) {
      return { value: Math.max(1, Math.min(999, Math.round(rawValue))), prefix }
    }
  }
  const smallest = LATTICE_CONSTANT_PREFIXES[0]
  const largest = LATTICE_CONSTANT_PREFIXES[LATTICE_CONSTANT_PREFIXES.length - 1]
  if (meters < LENGTH_PREFIX_FACTORS[smallest]) {
    return { value: 1, prefix: smallest }
  }
  return { value: 999, prefix: largest }
}
