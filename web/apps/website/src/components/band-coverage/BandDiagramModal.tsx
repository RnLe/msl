'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronRight, Download, X } from 'lucide-react'
import type { LatticeType } from './types'
import { BandDiagramPlot, type BandSeries, type MaterialHighlight } from './BandDiagramPlot'

const PRIMARY_BG = '#111111'

export type BandPreviewPayload = {
  lattice: LatticeType
  epsBg: number
  rOverA: number
  kPath: number[]
  series: BandSeries[]
  note?: string
}

type BandDiagramModalProps = {
  open: boolean
  onClose: () => void
  lattice: LatticeType
  kPath?: number[]
  series?: BandSeries[]
  loading?: boolean
  radius?: number
  epsBg?: number
  materialHighlight?: MaterialHighlight | null
  preview?: BandPreviewPayload | null
}

export function BandDiagramModal({
  open,
  onClose,
  lattice,
  kPath,
  series,
  loading = false,
  radius,
  epsBg,
  materialHighlight,
  preview,
}: BandDiagramModalProps) {
  const [diagramMode, setDiagramMode] = useState<'lines' | 'points'>('lines')
  const [isDownloadMenuOpen, setDownloadMenuOpen] = useState(false)
  const [pngBranchOpen, setPngBranchOpen] = useState(false)
  const downloadButtonRef = useRef<HTMLButtonElement | null>(null)
  const downloadMenuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!open) {
      setDownloadMenuOpen(false)
      setPngBranchOpen(false)
    }
  }, [open])

  useEffect(() => {
    if (!isDownloadMenuOpen) return
    const handleClick = (event: MouseEvent) => {
      const target = event.target as Node
      if (
        downloadMenuRef.current &&
        !downloadMenuRef.current.contains(target) &&
        downloadButtonRef.current &&
        !downloadButtonRef.current.contains(target)
      ) {
        setDownloadMenuOpen(false)
        setPngBranchOpen(false)
      }
    }
    document.addEventListener('pointerdown', handleClick)
    return () => document.removeEventListener('pointerdown', handleClick)
  }, [isDownloadMenuOpen])

  const handleCsvDownload = useCallback(() => {
    if (!preview || !preview.series?.length || !preview.kPath?.length) {
      console.warn('[BandDiagramModal] No preview data available for CSV export.')
      return
    }

    const categorizeSeries = () => {
      const te: Array<{ number: number; values: number[] }> = []
      const tm: Array<{ number: number; values: number[] }> = []
      preview.series?.forEach((band) => {
        const polarization = extractPolarization(band.label)
        const values = band.values ?? []
        const extracted = extractBandNumber(band.label)
        const meaningful = values.some((value) => Number.isFinite(value) && Math.abs(value) > 1e-9)
        if (!meaningful) return
        const number = Number.isFinite(extracted) ? Math.max(0, extracted - 1) : undefined
        if (polarization === 'TE') {
          te.push({ number: number ?? te.length, values })
        } else {
          tm.push({ number: number ?? tm.length, values })
        }
      })
      const sorter = (a: { number: number }, b: { number: number }) => a.number - b.number
      te.sort(sorter)
      tm.sort(sorter)
      return { te, tm }
    }

    const { te, tm } = categorizeSeries()

    const header = [
      'kPath',
      ...te.map((band) => `te${band.number}`),
      ...tm.map((band) => `tm${band.number}`),
    ]

    const rows: string[] = preview.kPath.map((kValue, index) => {
      const teValues = te.map((band) => formatCsvNumber(band.values[index]))
      const tmValues = tm.map((band) => formatCsvNumber(band.values[index]))
      return [formatCsvNumber(kValue), ...teValues, ...tmValues].join(',')
    })

    const csvContent = [header.join(','), ...rows].join('\n')
    triggerDownload(csvContent, buildFilename(preview, 'csv'), 'text/csv')
    setDownloadMenuOpen(false)
  }, [preview])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="relative w-full max-w-5xl rounded-2xl border border-white/10 p-6 shadow-2xl" style={{ backgroundColor: PRIMARY_BG }}>
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 rounded-full bg-transparent p-2.5 text-white transition hover:bg-white/10"
          aria-label="Close expanded band diagram"
        >
          <X className="h-5 w-5" />
        </button>
        <div className="absolute right-16 top-4" style={{ position: 'absolute' }}>
          <div className="relative">
            <button
              ref={downloadButtonRef}
              type="button"
              className="rounded-full bg-transparent p-2.5 text-white transition hover:bg-white/10"
              aria-label="Open band diagram downloads"
              onClick={() => setDownloadMenuOpen((prev) => !prev)}
            >
              <Download className="h-5 w-5" />
            </button>
            {isDownloadMenuOpen ? (
              <div
                ref={downloadMenuRef}
                className="absolute right-0 mt-2 w-52 border border-white/10 bg-[#181818] text-sm shadow-2xl"
                style={{ borderRadius: 0, zIndex: 40 }}
              >
                {/* Temporarily disable non-CSV downloads while preserving the code for future use */}
                {false && (
                  <>
                    <div className="relative" onMouseEnter={() => setPngBranchOpen(true)}>
                      <button
                        type="button"
                        className="flex h-10 w-full items-center justify-between bg-[#181818] text-left text-white transition hover:bg-[#242424]"
                        style={{ padding: 0, margin: 0 }}
                        onMouseEnter={() => setPngBranchOpen(true)}
                      >
                        <span className="flex-1 pl-3">PNG</span>
                        <span className="flex items-center pr-3">
                          <ChevronRight className="h-4 w-4" />
                        </span>
                      </button>
                      {pngBranchOpen ? (
                        <div
                          className="absolute left-full top-0 w-36 border border-white/10 bg-[#1c1c1c]"
                          style={{ borderRadius: 0, zIndex: 50 }}
                          onMouseEnter={() => setPngBranchOpen(true)}
                          onMouseLeave={() => setPngBranchOpen(false)}
                        >
                          <button
                            type="button"
                            className="flex h-10 w-full items-center bg-[#1c1c1c] text-left text-white transition hover:bg-[#2a2a2a]"
                            style={{ padding: 0, margin: 0 }}
                          >
                            <span className="pl-3">Dark</span>
                          </button>
                          <button
                            type="button"
                            className="flex h-10 w-full items-center bg-[#1c1c1c] text-left text-white transition hover:bg-[#2a2a2a]"
                            style={{ padding: 0, margin: 0 }}
                          >
                            <span className="pl-3">Light</span>
                          </button>
                        </div>
                      ) : null}
                    </div>
                    <button
                      type="button"
                      className="flex h-10 w-full items-center bg-[#181818] text-left text-white transition hover:bg-[#242424]"
                      style={{ padding: 0, margin: 0 }}
                    >
                      <span className="pl-3">SVG</span>
                    </button>
                    <button
                      type="button"
                      className="flex h-10 w-full items-center bg-[#181818] text-left text-white transition hover:bg-[#242424]"
                      style={{ padding: 0, margin: 0 }}
                    >
                      <span className="pl-3">PDF</span>
                    </button>
                  </>
                )}
                <button
                  type="button"
                  onClick={handleCsvDownload}
                  className="flex h-10 w-full items-center bg-[#181818] text-left text-white transition hover:bg-[#242424]"
                  style={{ padding: 0, margin: 0 }}
                >
                  <span className="pl-3">CSV</span>
                </button>
              </div>
            ) : null}
          </div>
        </div>
        <BandDiagramPlot
          lattice={lattice}
          kPath={kPath}
          series={series}
          loading={loading}
          size="large"
          mode={diagramMode}
          onToggleMode={() => setDiagramMode((mode) => (mode === 'lines' ? 'points' : 'lines'))}
          radius={radius}
          epsBg={epsBg}
          showParams
          materialHighlight={materialHighlight ?? null}
        />
      </div>
    </div>
  )
}

function extractPolarization(label: string) {
  const match = label.match(/TE|TM/i)
  return match ? match[0].toUpperCase() : 'TM'
}

function extractBandNumber(label: string) {
  const match = label.match(/(\d+)/)
  return match ? Number(match[1]) : 1
}

function triggerDownload(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function formatCsvNumber(value: number | undefined) {
  if (value === undefined || Number.isNaN(value)) return ''
  if (!Number.isFinite(value)) return ''
  return value.toFixed(6)
}

function buildFilename(preview: BandPreviewPayload, extension: string) {
  const { lattice, epsBg, rOverA } = preview
  const slug = `${lattice ?? 'band'}_${(epsBg ?? '').toString().replace(/[^0-9a-z.]+/gi, '-')}_${(rOverA ?? '').toString().replace(/[^0-9a-z.]+/gi, '-')}`
  return `band-diagram-${slug}.${extension}`
}
