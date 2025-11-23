export type MaterialCategory = 'reference' | 'polymer' | 'intermediate' | 'semiconductor' | 'matrix'

export type MaterialLibraryEntry = {
  id: string
  label: string // abbreviation / sticker text
  fullName: string
  epsilon: number
  refractiveIndex: number
  category: MaterialCategory
  summary: string
  designWindow?: string
  notes?: string
  aliases?: string[]
  sources?: Array<{ label: string; url: string }>
}

export const MATERIAL_LIBRARY: MaterialLibraryEntry[] = [
  {
    id: 'air',
    label: 'Air',
    fullName: 'Air / Vacuum',
    epsilon: 1.0,
    refractiveIndex: 1.0,
    category: 'reference',
    summary: 'Baseline low-index background used for most air-hole photonic crystals.',
    designWindow: '≈0.2–20 µm (UV–mid-IR)',
    notes: 'Modeled as ε = 1 with no dispersion.',
  },
  {
    id: 'water',
    label: 'H2O',
    fullName: 'Water',
    epsilon: 1.75,
    refractiveIndex: 1.32,
    category: 'reference',
    summary: 'Liquid background for infiltrated PCs and sensing platforms.',
    designWindow: '0.40–0.70 µm + narrow 0.8–1.3 µm windows',
  },
  {
    id: 'silica',
    label: 'SiO2',
    fullName: 'Fused Silica',
    epsilon: 2.1,
    refractiveIndex: 1.45,
    category: 'reference',
    summary: 'Workhorse low-index solid, ubiquitous in silica-air fibers and slabs.',
    designWindow: '≈0.18–3.5 µm',
  },
  {
    id: 'fluoride',
    label: 'CaF2',
    fullName: 'Fluoride Glass',
    epsilon: 1.95,
    refractiveIndex: 1.4,
    category: 'reference',
    summary: 'Ultra-low-index fluorides (MgF2, CaF2) for wide-band PCs and fibers.',
    designWindow: '≈0.20–8 µm',
  },
  {
    id: 'polymer',
    label: 'PMMA',
    fullName: 'Optical Polymer',
    epsilon: 2.35,
    refractiveIndex: 1.53,
    category: 'polymer',
    summary: 'Generic PMMA / polystyrene range used in soft or THz photonic crystals.',
    designWindow: '≈0.30–2.8 µm',
  },
  {
    id: 'ptfe',
    label: 'PTFE',
    fullName: 'PTFE / Teflon',
    epsilon: 2.05,
    refractiveIndex: 1.43,
    category: 'polymer',
    summary: 'Low-index matrix for polymer-air PCs and flexible devices.',
    designWindow: '≈150–600 µm (0.5–2 THz)',
  },
  {
    id: 'su8',
    label: 'SU8',
    fullName: 'SU-8 Photoresist',
    epsilon: 2.5,
    refractiveIndex: 1.58,
    category: 'polymer',
    summary: 'Lithography resist often used as the high-index region in micro-fabricated PCs.',
    designWindow: '≥0.36–~1.5 µm (practical)',
  },
  {
    id: 'alumina',
    label: 'Al2O3',
    fullName: 'Alumina',
    epsilon: 3.1,
    refractiveIndex: 1.76,
    category: 'intermediate',
    summary: 'Moderate-index dielectric for 2D/3D PCs and multilayers.',
    designWindow: '≈0.20–5.5 µm',
  },
  {
    id: 'si3n4',
    label: 'Si3N4',
    fullName: 'Silicon Nitride',
    epsilon: 4.0,
    refractiveIndex: 2.0,
    category: 'intermediate',
    summary: 'Integrated photonics workhorse; pairs well with silica backgrounds.',
    designWindow: '≈0.40–4 µm',
  },
  {
    id: 'aln',
    label: 'AlN',
    fullName: 'Aluminum Nitride',
    epsilon: 4.2,
    refractiveIndex: 2.05,
    category: 'intermediate',
    summary: 'Piezo + photonic material, typically treated as scalar ε.',
    designWindow: '≈0.20–5 µm',
  },
  {
    id: 'gan',
    label: 'GaN',
    fullName: 'Gallium Nitride',
    epsilon: 5.2,
    refractiveIndex: 2.28,
    category: 'intermediate',
    summary: 'Blue/UV photonic crystal slabs and LEDs.',
    designWindow: '≈0.36–4 µm (PC work ≤0.8 µm)',
  },
  {
    id: 'tio2',
    label: 'TiO2',
    fullName: 'Titanium Dioxide',
    epsilon: 5.8,
    refractiveIndex: 2.4,
    category: 'intermediate',
    summary: 'Visible-range dielectric for inverse opals and structural color PCs.',
    designWindow: '≈0.40–3 µm',
  },
  {
    id: 'chalcogenide',
    label: 'ChG',
    fullName: 'Chalcogenide Glass',
    epsilon: 6.0,
    refractiveIndex: 2.45,
    category: 'intermediate',
    summary: 'IR / nonlinear photonic crystal fibers (As2S3, As2Se3).',
    designWindow: 'As2S3 ≈0.7–6 µm, As2Se3 ≈1–10 µm',
  },
  {
    id: 'si',
    label: 'Si',
    fullName: 'Silicon',
    epsilon: 12.0,
    refractiveIndex: 3.46,
    category: 'semiconductor',
    summary: 'Canonical high-index platform for strong-gap 2D slabs.',
    designWindow: '≈1.2–8 µm',
  },
  {
    id: 'gaas',
    label: 'GaAs',
    fullName: 'Gallium Arsenide',
    epsilon: 10.9,
    refractiveIndex: 3.3,
    category: 'semiconductor',
    summary: 'III-V photonic crystal slabs and cavities.',
    designWindow: '≈0.9–16 µm',
  },
  {
    id: 'inp',
    label: 'InP',
    fullName: 'Indium Phosphide',
    epsilon: 10.1,
    refractiveIndex: 3.18,
    category: 'semiconductor',
    summary: 'Telecom photonic crystals in InP/air membranes.',
    designWindow: '≈1.0–9 µm (practical 1.2–5 µm)',
  },
  {
    id: 'gap',
    label: 'GaP',
    fullName: 'Gallium Phosphide',
    epsilon: 9.6,
    refractiveIndex: 3.1,
    category: 'semiconductor',
    summary: 'High-index visible photonic crystals and nanobeam devices.',
    designWindow: '≈0.55–11 µm',
  },
  {
    id: 'ge',
    label: 'Ge',
    fullName: 'Germanium',
    epsilon: 16.0,
    refractiveIndex: 4.0,
    category: 'semiconductor',
    summary: 'Mid-IR photonic crystals with extreme index contrast.',
    designWindow: '≈2–14 µm',
  },
]

const MATERIAL_SHORTCUT_IDS = ['air', 'water', 'silica', 'polymer', 'si3n4', 'tio2', 'gaas', 'si', 'inp', 'gan', 'aln', 'gap', 'ge']

export const MATERIAL_SHORTCUTS: MaterialLibraryEntry[] = MATERIAL_SHORTCUT_IDS.map((id) => MATERIAL_LIBRARY.find((entry) => entry.id === id)).filter((entry): entry is MaterialLibraryEntry => Boolean(entry))

export function getMaterialById(id: string) {
  return MATERIAL_LIBRARY.find((entry) => entry.id === id) ?? null
}

export function formatMaterialDisplay(entry: MaterialLibraryEntry) {
  const epsilonLabel = entry.epsilon.toFixed(2)
  return `${entry.fullName} · ε≈${epsilonLabel}`
}
