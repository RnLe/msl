import { SPEED_OF_LIGHT, describeFrequency } from './units'

export type MaterialSpectralWindow = {
  wavelengthMicron: [number, number]
  wavelengthMeters: [number, number]
  frequencyHz: [number, number]
}

export type MaterialSpectralProfile = {
  id: string
  windows: MaterialSpectralWindow[]
}

const MICRON_TO_METER = 1e-6

type RawWindowMap = Record<string, Array<[number, number]>>

const RAW_WINDOWS: RawWindowMap = {
  air: [[0.2, 20]],
  water: [[0.4, 0.7], [0.8, 1.3]],
  silica: [[0.18, 3.5]],
  fluoride: [[0.2, 8]],
  polymer: [[0.3, 2.8]],
  ptfe: [[150, 600]],
  su8: [[0.36, 1.5]],
  alumina: [[0.2, 5.5]],
  si3n4: [[0.4, 4]],
  aln: [[0.2, 5]],
  gan: [[0.36, 4]],
  tio2: [[0.4, 3]],
  chalcogenide: [[0.7, 6], [1, 10]],
  si: [[1.2, 8]],
  gaas: [[0.9, 16]],
  inp: [[1.0, 9]],
  gap: [[0.55, 11]],
  ge: [[2, 14]],
}

function clampRange([min, max]: [number, number]): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 0]
  }
  const sorted: [number, number] = min <= max ? [min, max] : [max, min]
  return [Math.max(sorted[0], 0), Math.max(sorted[1], 0)]
}

function computeWindow(range: [number, number]): MaterialSpectralWindow {
  const [minMicronRaw, maxMicronRaw] = clampRange(range)
  const minMicron = Math.max(minMicronRaw, 1e-4)
  const maxMicron = Math.max(maxMicronRaw, minMicron + 1e-6)
  const lambdaMinMeters = minMicron * MICRON_TO_METER
  const lambdaMaxMeters = maxMicron * MICRON_TO_METER
  const freqHigh = SPEED_OF_LIGHT / lambdaMinMeters
  const freqLow = SPEED_OF_LIGHT / lambdaMaxMeters
  return {
    wavelengthMicron: [minMicron, maxMicron],
    wavelengthMeters: [lambdaMinMeters, lambdaMaxMeters],
    frequencyHz: [freqLow, freqHigh],
  }
}

export const MATERIAL_SPECTRAL_PROFILES: Record<string, MaterialSpectralProfile> = Object.fromEntries(
  Object.entries(RAW_WINDOWS).map(([id, ranges]) => [
    id,
    {
      id,
      windows: ranges.map((range) => computeWindow(range)),
    },
  ])
)

export function getMaterialSpectralProfile(id: string): MaterialSpectralProfile | null {
  return MATERIAL_SPECTRAL_PROFILES[id] ?? null
}

function formatRangeNumber(value: number) {
  const absValue = Math.abs(value)
  if (absValue >= 100) return value.toFixed(0)
  if (absValue >= 10) return value.toFixed(1)
  if (absValue >= 1) return value.toFixed(2)
  return value.toFixed(3)
}

export function formatWavelengthRangeMicron([start, end]: [number, number]) {
  return `${formatRangeNumber(start)}–${formatRangeNumber(end)} µm`
}

export function formatFrequencyRangeHz([start, end]: [number, number]) {
  const low = Math.min(start, end)
  const high = Math.max(start, end)
  const lowDesc = describeFrequency(low)
  const highDesc = describeFrequency(high)
  if (lowDesc && highDesc && lowDesc.unit === highDesc.unit) {
    return `${lowDesc.value}–${highDesc.value} ${highDesc.unit}`
  }
  if (highDesc && !lowDesc) {
    return `≤ ${highDesc.value} ${highDesc.unit}`
  }
  if (lowDesc && !highDesc) {
    return `${lowDesc.value} ${lowDesc.unit}+`
  }
  if (lowDesc && highDesc) {
    return `${lowDesc.value} ${lowDesc.unit} – ${highDesc.value} ${highDesc.unit}`
  }
  return null
}

export function mapMaterialWindows<T>(
  profile: MaterialSpectralProfile | null,
  projector: (window: MaterialSpectralWindow, index: number) => T
): T[] {
  if (!profile?.windows?.length) return []
  return profile.windows.map(projector)
}
