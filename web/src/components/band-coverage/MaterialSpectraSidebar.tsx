import { useCallback, useMemo, useState } from 'react'
import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import type { MaterialLibraryEntry } from './library'
import { useElementSize } from './use-element-size'
import { getMaterialSpectralProfile } from './materialSpectra'
import { getMaterialAccent, lightenHexColor } from './materialLibraryShared'

type SpectralRow = {
  entry: MaterialLibraryEntry
  accent: { background: string; text: string }
  windows: Array<{ start: number; end: number }>
  alignFrequencyHz: number | null
}

type MaterialSpectraSidebarProps = {
  entries: MaterialLibraryEntry[]
  activeId: string | null
  onSelect: (entry: MaterialLibraryEntry) => void
  onAlign?: (entry: MaterialLibraryEntry, frequencyHz: number) => void
}

export function MaterialSpectraSidebar({ entries, activeId, onSelect, onAlign }: MaterialSpectraSidebarProps) {
  const rows = useMemo<SpectralRow[]>(() => {
    const collected: SpectralRow[] = []
    entries.forEach((entry) => {
      const profile = getMaterialSpectralProfile(entry.id)
      if (!profile?.windows?.length) return
      const windows = profile.windows
        .map((window) => {
          const start = clampWavelengthToDomain(window.wavelengthMicron[0])
          const end = clampWavelengthToDomain(window.wavelengthMicron[1])
          if (!(end > start)) return null
          return { start, end }
        })
        .filter((window): window is { start: number; end: number } => Boolean(window))
      if (!windows.length) return
      const accentBase = getMaterialAccent(entry.id)
      let alignFrequencyHz: number | null = null
      profile.windows.forEach((window) => {
        if (!window.frequencyHz?.length) return
        const upper = Math.max(window.frequencyHz[0], window.frequencyHz[1])
        if (upper > (alignFrequencyHz ?? 0)) alignFrequencyHz = upper
      })
      collected.push({
        entry,
        accent: entry.id === 'water' ? { ...accentBase, text: '#ffffff' } : accentBase,
        windows,
        alignFrequencyHz,
      })
    })
    return collected
  }, [entries])

  if (!rows.length) return null

  return (
    <div
      className="flex h-full w-full flex-col overflow-hidden border border-[#1c1c1c] bg-[#111111] text-white lg:max-w-[360px]"
      style={{ minHeight: `${TRANSMISSION_SIDEBAR_MIN_HEIGHT}px` }}
    >
      <div className="flex flex-1 flex-col px-5 py-5" style={{ backgroundColor: '#111111' }}>
        <div className="flex h-full w-full">
          <TransmissionRangePlot rows={rows} activeId={activeId} onSelect={onSelect} onAlign={onAlign} />
        </div>
      </div>
    </div>
  )
}

type TransmissionRangePlotProps = {
  rows: SpectralRow[]
  activeId: string | null
  onSelect: (entry: MaterialLibraryEntry) => void
  onAlign?: (entry: MaterialLibraryEntry, frequencyHz: number) => void
}

function TransmissionRangePlot({ rows, activeId, onSelect, onAlign }: TransmissionRangePlotProps) {
  const [plotRef, plotSize] = useElementSize<HTMLDivElement>()
  const width = Math.max(plotSize.width || TRANSMISSION_SIDEBAR_WIDTH, 320)
  const margins = { top: 90, right: 36, bottom: 90, left: 32 }
  const innerWidth = width - margins.left - margins.right
  const barHeight = 18
  const barGap = 8
  const rawContentHeight = rows.length ? rows.length * (barHeight + barGap) - barGap : 0
  const availableHeight = Math.max(plotSize.height || TRANSMISSION_SIDEBAR_MIN_HEIGHT - 40, 360)
  const targetInnerHeight = Math.max(availableHeight - (margins.top + margins.bottom), 200)
  const innerHeight = Math.max(rawContentHeight, targetInnerHeight)
  const totalHeight = innerHeight + margins.top + margins.bottom
  const axisY = margins.top + innerHeight
  const rowStartY = margins.top + Math.max(0, (innerHeight - rawContentHeight) / 2)
  const logMin = Math.log10(TRANSMISSION_DOMAIN[0])
  const logSpan = Math.log10(TRANSMISSION_DOMAIN[1]) - logMin
  const projectX = (value: number) => {
    const clamped = clampWavelengthToDomain(value)
    const ratio = (Math.log10(clamped) - logMin) / logSpan
    return margins.left + ratio * innerWidth
  }

  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const handleSelectWithAlign = useCallback(
    (entry: MaterialLibraryEntry, alignFrequencyHz?: number | null) => {
      onSelect(entry)
      if (onAlign && Number.isFinite(alignFrequencyHz) && (alignFrequencyHz as number) > 0) {
        onAlign(entry, alignFrequencyHz as number)
      }
    },
    [onAlign, onSelect]
  )

  const handleRowKey = useCallback(
    (event: ReactKeyboardEvent<SVGGElement>, entry: MaterialLibraryEntry, alignFrequencyHz?: number | null) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        handleSelectWithAlign(entry, alignFrequencyHz)
      }
    },
    [handleSelectWithAlign]
  )

  return (
    <div ref={plotRef} className="h-full w-full">
      <svg role="presentation" width="100%" height="100%" viewBox={`0 0 ${width} ${totalHeight}`} preserveAspectRatio="none" style={{ display: 'block' }}>
        {TRANSMISSION_MINOR_TICKS.map((tick) => {
          const x = projectX(tick)
          return <line key={`minor-${tick}`} x1={x} y1={margins.top} x2={x} y2={axisY} stroke="#1b1b1b" strokeWidth={1} />
        })}
        {TRANSMISSION_MAJOR_TICKS.map((tick) => {
          const x = projectX(tick)
          return (
            <g key={`major-${tick}`}>
              <line x1={x} y1={margins.top} x2={x} y2={axisY} stroke="#2d2d2d" strokeWidth={1.5} />
              <text x={x} y={axisY + 24} fill="#f5f5f5" fontSize={12} textAnchor="middle" fontFamily="inherit">
                {tick}
              </text>
            </g>
          )
        })}
        <line x1={margins.left} y1={axisY} x2={width - margins.right} y2={axisY} stroke="#c9c9c9" strokeWidth={1.5} />
        <text
          x={margins.left + innerWidth / 2}
          y={axisY + 48}
          textAnchor="middle"
          fontSize={12}
          letterSpacing="0.08em"
          fill="#f5f5f5"
        >
          Wavelength (µm)
        </text>
        {TRANSMISSION_REGIONS.map((region) => {
          const midpoint = (region.start + region.end) / 2
          const x = projectX(midpoint)
          return (
            <text key={region.label} x={x} y={margins.top - 24} textAnchor="middle" fontSize={12} fontWeight={600} fill="#d2d2d2">
              {region.label}
            </text>
          )
        })}
        {rows.map((row, index) => {
          const y = rowStartY + index * (barHeight + barGap)
          const overallStart = Math.min(...row.windows.map((window) => window.start))
          const overallEnd = Math.max(...row.windows.map((window) => window.end))
          const labelX = (projectX(overallStart) + projectX(overallEnd)) / 2
          const isActive = activeId === row.entry.id
          const isHovered = hoveredId === row.entry.id
          return (
            <g
              key={row.entry.id}
              role="button"
              tabIndex={0}
              aria-label={`View ${row.entry.fullName}`}
              onClick={() => handleSelectWithAlign(row.entry, row.alignFrequencyHz)}
              onKeyDown={(event) => handleRowKey(event, row.entry, row.alignFrequencyHz)}
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHoveredId(row.entry.id)}
              onMouseLeave={() => setHoveredId((prev) => (prev === row.entry.id ? null : prev))}
            >
              {row.windows.map((window, windowIndex) => {
                const xStart = projectX(window.start)
                const xEnd = projectX(window.end)
                const widthSegment = Math.max(4, xEnd - xStart)
                const baseColor = row.accent.background
                const fillColor = isHovered ? lightenHexColor(baseColor, 0.18) : baseColor
                return (
                  <rect
                    key={`${row.entry.id}-${windowIndex}`}
                    x={xStart}
                    y={y}
                    width={widthSegment}
                    height={barHeight}
                    rx={barHeight / 2}
                    fill={fillColor}
                    fillOpacity={0.95}
                    stroke={isActive ? '#ffffff' : 'transparent'}
                    strokeWidth={isActive ? 2 : 0}
                  />
                )
              })}
              <text x={labelX} y={y + barHeight / 2} textAnchor="middle" dominantBaseline="middle" fontSize={12} fontWeight={600} fill={row.accent.text}>
                {row.entry.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

const TRANSMISSION_DOMAIN: [number, number] = [0.1, 1000]
const TRANSMISSION_MAJOR_TICKS = [0.1, 1, 10, 100, 1000]
const TRANSMISSION_MINOR_TICKS = buildLogMinorTicks(TRANSMISSION_DOMAIN, TRANSMISSION_MAJOR_TICKS)
const TRANSMISSION_REGIONS = [
  { label: 'UV', start: 0.1, end: 0.4 },
  { label: 'VIS', start: 0.4, end: 0.7 },
  { label: 'NIR', start: 0.7, end: 3 },
  { label: 'MIR', start: 3, end: 30 },
]
const TRANSMISSION_SIDEBAR_MIN_HEIGHT = 520
const TRANSMISSION_SIDEBAR_WIDTH = 360

function clampWavelengthToDomain(value: number) {
  if (!Number.isFinite(value)) return TRANSMISSION_DOMAIN[0]
  return Math.min(TRANSMISSION_DOMAIN[1], Math.max(TRANSMISSION_DOMAIN[0], value))
}

function buildLogMinorTicks(domain: [number, number], majors: number[]) {
  const [min, max] = domain
  const ticks: number[] = []
  const startExp = Math.floor(Math.log10(min))
  const endExp = Math.ceil(Math.log10(max))
  for (let exp = startExp; exp <= endExp; exp += 1) {
    for (let multiplier = 2; multiplier < 10; multiplier += 1) {
      const value = Math.pow(10, exp) * multiplier
      if (value <= min || value >= max) continue
      if (majors.includes(value)) continue
      ticks.push(Number(value.toPrecision(6)))
    }
  }
  return ticks
}
