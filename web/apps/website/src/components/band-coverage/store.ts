import { create } from 'zustand'
import type { BandDataPoint, CoveragePoint } from './types'

export const LATTICE_CONSTANT_PREFIXES = ['fm', 'pm', 'nm', 'μm', 'mm', 'cm', 'dm', 'm'] as const
export type LatticeConstantPrefix = (typeof LATTICE_CONSTANT_PREFIXES)[number]

function selectionsEqual(a: CoveragePoint | null, b: CoveragePoint | null) {
  if (a === b) return true
  if (!a || !b) return false
  return (
    a.lattice === b.lattice &&
    a.epsIndex === b.epsIndex &&
    a.rIndex === b.rIndex &&
    a.epsBg === b.epsBg &&
    a.rOverA === b.rOverA &&
    a.statusCode === b.statusCode
  )
}

export type BandCoverageStore = {
  hovered: CoveragePoint | null
  selected: CoveragePoint | null
  bandHovered: BandDataPoint | null
  bandSelected: BandDataPoint | null
  activeAxis: 'row' | 'column' | null
  latticeConstantValue: number | null
  latticeConstantPrefix: LatticeConstantPrefix | null
  setHovered: (point: CoveragePoint | null) => void
  setSelected: (point: CoveragePoint | null) => void
  setBandHovered: (point: BandDataPoint | null) => void
  setBandSelected: (point: BandDataPoint | null) => void
  setActiveAxis: (axis: 'row' | 'column' | null) => void
  setLatticeConstantValue: (value: number | null) => void
  setLatticeConstantPrefix: (prefix: LatticeConstantPrefix | null) => void
  resetLatticeConstant: () => void
}

export const useBandCoverageStore = create<BandCoverageStore>((set) => ({
  hovered: null,
  selected: null,
  bandHovered: null,
  bandSelected: null,
  activeAxis: null,
  latticeConstantValue: null,
  latticeConstantPrefix: null,
  setHovered: (point) => set({ hovered: point }),
  setSelected: (point) =>
    set((state) => (selectionsEqual(state.selected, point) ? state : { selected: point })),
  setBandHovered: (point) => set({ bandHovered: point }),
  setBandSelected: (point) => set({ bandSelected: point }),
  setActiveAxis: (axis) => set({ activeAxis: axis }),
  setLatticeConstantValue: (value) => set({ latticeConstantValue: value }),
  setLatticeConstantPrefix: (prefix) => set({ latticeConstantPrefix: prefix }),
  resetLatticeConstant: () => set({ latticeConstantValue: null, latticeConstantPrefix: null }),
}))
