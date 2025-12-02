import { useEffect, useMemo } from 'react'
import { X } from 'lucide-react'
import type { MaterialLibraryEntry } from './library'
import { MaterialSpectraSidebar } from './MaterialSpectraSidebar'
import { MaterialLibraryCard } from './MaterialLibraryCard'
import { MATERIAL_CATEGORY_LABELS, MATERIAL_CATEGORY_ORDER } from './materialLibraryShared'

type MaterialLibraryModalProps = {
  open: boolean
  onClose: () => void
  entries: MaterialLibraryEntry[]
  activeId: string | null
  onSelect: (entry: MaterialLibraryEntry) => void
  onAlignMaterial?: (entry: MaterialLibraryEntry, frequencyHz: number) => void
  axisMaxNormalized: number
}

export function MaterialLibraryModal({ open, onClose, entries, activeId, onSelect, onAlignMaterial, axisMaxNormalized }: MaterialLibraryModalProps) {
  useEffect(() => {
    if (!open) return undefined
    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = originalOverflow
    }
  }, [open])

  const grouped = useMemo(() => {
    const bucket = new Map<string, MaterialLibraryEntry[]>()
    entries.forEach((entry) => {
      const list = bucket.get(entry.category) ?? []
      list.push(entry)
      bucket.set(entry.category, list)
    })
    return bucket
  }, [entries])

  if (!open) return null

  const orderedGroups = MATERIAL_CATEGORY_ORDER.map((category) => ({
    category,
    items: grouped.get(category) ?? [],
  })).filter((group) => group.items.length)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 flex w-full max-w-[min(calc(100vw-2rem),1600px)] flex-col items-stretch gap-4 text-white lg:flex-row lg:gap-0">
        <div className="flex w-full flex-col lg:w-auto lg:shrink-0">
          <MaterialSpectraSidebar entries={entries} activeId={activeId} onSelect={onSelect} onAlign={onAlignMaterial} />
        </div>
        <div className="relative w-full shrink-0 overflow-hidden border border-[#1c1c1c] bg-[#111111] text-white shadow-[0_40px_120px_rgba(0,0,0,0.65)] lg:w-[92vw] lg:max-w-5xl">
          <div className="flex items-center justify-between border-b border-[#1f1f1f] px-6 py-4" style={{ backgroundColor: '#131313' }}>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.4em]" style={{ color: '#8a8a8a' }}>
                Material Library
              </div>
              <div className="text-lg font-semibold" style={{ color: '#ffffff' }}>
                Photonic Crystal Reference Set
              </div>
            </div>
            <button type="button" onClick={onClose} className="p-2 text-white transition hover:text-white/70" aria-label="Close material library">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="max-h-[70vh] overflow-y-auto px-6 py-5" style={{ backgroundColor: '#111111' }}>
            <div className="space-y-6">
              <p className="text-sm leading-relaxed" style={{ color: '#d3d3d3' }}>
                This library lists <strong>isotropic, non-metallic dielectrics</strong> for photonic crystals, all modeled with a simple real scalar (<em>ε<sub>r</sub></em>).
                Each card calls out a <strong>typical low-loss λ-window (µm)</strong> (sources below).
                The scalar ε<sub>r</sub> model stays valid anywhere the material is transparent (visible, telecom, or mid-IR alike).
                <span className="block mt-3">Click a material to apply its ε<sub>r</sub> to the heatmap.</span>
              </p>
              {orderedGroups.map((group) => (
                <div key={group.category} className="space-y-3">
                  <div className="text-base font-semibold uppercase tracking-[0.35em]" style={{ color: '#e6e6e6' }}>
                    {MATERIAL_CATEGORY_LABELS[group.category]}
                  </div>
                  <div className="space-y-2">
                    {group.items.map((entry) => (
                      <MaterialLibraryCard
                        key={entry.id}
                        entry={entry}
                        active={activeId === entry.id}
                        onSelect={onSelect}
                        onAlign={onAlignMaterial}
                        axisMaxNormalized={axisMaxNormalized}
                      />
                    ))}
                  </div>
                </div>
              ))}
              <div className="border-t border-[#1f1f1f] pt-4">
                <div className="text-xs font-semibold uppercase tracking-[0.35em]" style={{ color: '#8a8a8a' }}>
                  Sources
                </div>
                <ul className="mt-3 space-y-2 text-sm" style={{ color: '#c7c7c7' }}>
                  <li>
                    <div className="flex flex-wrap items-baseline gap-4">
                      <a href="https://refractiveindex.info" className="underline" style={{ color: '#1b82c8' }} target="_blank" rel="noreferrer">
                        RefractiveIndex.INFO
                      </a>
                      <span>Consolidated n, k datasets across UV–mid-IR for bulk materials.</span>
                    </div>
                  </li>
                  <li>
                    <div className="flex flex-wrap items-baseline gap-4">
                      <a href="https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6973" className="underline" style={{ color: '#1b82c8' }} target="_blank" rel="noreferrer">
                        Thorlabs optical polymer substrates
                      </a>
                      <span>PMMA/plexiglass transmission data and THz PTFE references.</span>
                    </div>
                  </li>
                  <li>
                    <div className="flex flex-wrap items-baseline gap-4">
                      <a href="https://www.crystran.com/optical-materials" className="underline" style={{ color: '#1b82c8' }} target="_blank" rel="noreferrer">
                        Crystran optical materials database
                      </a>
                      <span>Vendor transmission ranges for silica, sapphire, fluorides, etc.</span>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
