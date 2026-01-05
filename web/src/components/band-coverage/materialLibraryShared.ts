import type { MaterialCategory } from './library'
import { metersToPrefixedLattice, SPEED_OF_LIGHT } from './units'

type MaterialAccent = { background: string; text: string }

export const MATERIAL_CATEGORY_ORDER: MaterialCategory[] = ['reference', 'polymer', 'intermediate', 'semiconductor', 'matrix']

export const MATERIAL_CATEGORY_LABELS: Record<MaterialCategory, string> = {
  reference: 'Reference / Low Index',
  polymer: 'Polymers',
  intermediate: 'Intermediate Dielectrics',
  semiconductor: 'High-Index Semiconductors',
  matrix: 'Matrix Pairs',
}

export const MATERIAL_STICKER_SIZE = 88
export const MATERIAL_CARD_PADDING_X = 16
export const MATERIAL_CARD_PADDING_Y = 12

const MATERIAL_ACCENTS: Record<string, MaterialAccent> = {
  air: { background: '#F6F3EC', text: '#201c13' },
  water: { background: '#6FAFDB', text: '#062238' },
  silica: { background: '#C7D1DB', text: '#101a24' },
  fluoride: { background: '#E3D9FF', text: '#2a1a4a' },
  polymer: { background: '#F6B27A', text: '#3a1d07' },
  ptfe: { background: '#E4E1C4', text: '#2c2711' },
  su8: { background: '#D76B73', text: '#ffffff' },
  alumina: { background: '#B79AC8', text: '#240f33' },
  si3n4: { background: '#A8C686', text: '#1e2610' },
  aln: { background: '#8EB7B5', text: '#082626' },
  gan: { background: '#4B90A6', text: '#ffffff' },
  tio2: { background: '#E0B45A', text: '#2d1a05' },
  chalcogenide: { background: '#7E4F7B', text: '#ffffff' },
  si: { background: '#335C4C', text: '#ffffff' },
  gaas: { background: '#75485E', text: '#ffffff' },
  inp: { background: '#C47D3E', text: '#2c1400' },
  gap: { background: '#CC9F3C', text: '#2f1600' },
  ge: { background: '#494F5C', text: '#ffffff' },
}

const DEFAULT_MATERIAL_ACCENT: MaterialAccent = { background: '#dedede', text: '#111111' }

export function getMaterialAccent(id: string): MaterialAccent {
  return MATERIAL_ACCENTS[id] ?? DEFAULT_MATERIAL_ACCENT
}

export function lightenHexColor(hex: string, factor: number) {
  if (!hex || typeof hex !== 'string' || !hex.startsWith('#')) return hex
  const normalized = hex.length === 4
    ? hex
        .slice(1)
        .split('')
        .map((char) => char + char)
        .join('')
    : hex.slice(1)

  if (normalized.length !== 6) return hex

  const num = parseInt(normalized, 16)
  if (Number.isNaN(num)) return hex

  const clamp = (value: number) => Math.min(255, Math.max(0, value))
  const adjust = (value: number) => clamp(Math.round(value + (255 - value) * factor))

  const r = adjust((num >> 16) & 0xff)
  const g = adjust((num >> 8) & 0xff)
  const b = adjust(num & 0xff)

  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`
}

export function computeLatticeForFrequency(axisMaxNormalized: number, targetFrequencyHz?: number | null) {
  if (!Number.isFinite(targetFrequencyHz) || (targetFrequencyHz as number) <= 0) return null
  const maxNormalized = axisMaxNormalized > 0 ? axisMaxNormalized : 1
  const latticeMeters = (maxNormalized * SPEED_OF_LIGHT) / (targetFrequencyHz as number)
  return metersToPrefixedLattice(latticeMeters)
}
