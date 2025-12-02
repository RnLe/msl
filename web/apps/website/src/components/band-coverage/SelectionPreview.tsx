'use client'

import { useEffect, useState } from 'react'
import type { CoverageMode, CoveragePoint, LatticeType } from './types'
import { COLORS } from './palette'
import { HighSymmetryVisualization2D } from '../HighSymmetryVisualization2D'
import { BandDiagramPlot, type BandSeries, type MaterialHighlight } from './BandDiagramPlot'
import { BandDiagramModal, type BandPreviewPayload } from './BandDiagramModal'
import { GeometryPreview } from './GeometryPreview'

export type { BandSeries, BandPreviewPayload }

type SelectionPreviewProps = {
  point: CoveragePoint | null
  mode: CoverageMode
  bandState: { status: 'idle' | 'loading' | 'ready' | 'error' | 'prefetching'; data?: BandPreviewPayload; message?: string }
  demo: BandPreviewPayload
  materialHighlight?: MaterialHighlight | null
}

const DEMO_PREVIEW_DELAY_MS = 50

export function SelectionPreview({ point, mode, bandState, demo, materialHighlight }: SelectionPreviewProps) {
  const isDemo = mode === 'offline'
  const [demoLoading, setDemoLoading] = useState(false)
  const [isDiagramModalOpen, setDiagramModalOpen] = useState(false)
  const [diagramMode, setDiagramMode] = useState<'lines' | 'points'>('lines')

  useEffect(() => {
    if (!isDemo) {
      setDemoLoading(false)
      return
    }
    setDemoLoading(true)
    const timer = setTimeout(() => setDemoLoading(false), DEMO_PREVIEW_DELAY_MS)
    return () => clearTimeout(timer)
  }, [isDemo, point?.lattice, point?.epsIndex, point?.rIndex])

  const preview = isDemo ? demo : bandState.data
  const status = isDemo ? (demoLoading ? 'loading' : 'ready') : bandState.status ?? 'idle'
  const normalizedStatus = status === 'prefetching' ? 'loading' : status
  const loading = normalizedStatus === 'loading'

  const message = isDemo
    ? 'Demo preview (static pair)'
    : bandState.message || (status === 'idle' ? 'Select a geometry to fetch its band diagram.' : undefined)

  const geometrySource = point ?? (preview ? { lattice: preview.lattice, epsBg: preview.epsBg, rOverA: preview.rOverA } : null)

  if (!preview && !loading) {
    return (
      <div
        className="rounded-2xl border border-dashed px-4 py-6 text-sm"
        style={{ borderColor: COLORS.border, color: COLORS.textMuted }}
      >
        {message}
      </div>
    )
  }

  if (normalizedStatus === 'error' && !isDemo) {
    return (
      <div className="rounded-2xl border px-4 py-6 text-sm" style={{ borderColor: '#ff9f7a', color: '#ff9f7a' }}>
        {bandState.message ?? 'Could not load the band diagram yet.'}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <GeometryPreview
          lattice={geometrySource?.lattice ?? preview?.lattice ?? 'square'}
          rOverA={geometrySource?.rOverA ?? preview?.rOverA ?? 0}
        />
        <BandDiagramPlot
          lattice={geometrySource?.lattice ?? preview?.lattice ?? 'square'}
          kPath={preview?.kPath}
          series={preview?.series}
          loading={loading}
          onExpand={() => setDiagramModalOpen(true)}
          onToggleMode={() => setDiagramMode((mode) => (mode === 'lines' ? 'points' : 'lines'))}
          mode={diagramMode}
          radius={geometrySource?.rOverA ?? preview?.rOverA}
          epsBg={geometrySource?.epsBg ?? preview?.epsBg}
          showParams={false}
          materialHighlight={materialHighlight ?? null}
        />
      </div>

      <div>
        <div className="mb-2 text-center text-base font-semibold" style={{ color: COLORS.textPrimary }}>
          High-symmetry k-path
        </div>
        <HighSymmetryVisualization2D latticeType={toVizLattice(geometrySource?.lattice ?? preview?.lattice ?? 'square')} shells={1} height={260} syncBandStore />
      </div>

      <BandDiagramModal
        open={isDiagramModalOpen}
        onClose={() => setDiagramModalOpen(false)}
        lattice={geometrySource?.lattice ?? preview?.lattice ?? 'square'}
        kPath={preview?.kPath}
        series={preview?.series}
        loading={loading}
        radius={geometrySource?.rOverA ?? preview?.rOverA}
        epsBg={geometrySource?.epsBg ?? preview?.epsBg}
        materialHighlight={materialHighlight ?? null}
        preview={preview ?? null}
      />
    </div>
  )
}

function toVizLattice(lattice: LatticeType): 'square' | 'hexagonal' {
  if (lattice === 'hex' || lattice === 'hexagonal') return 'hexagonal'
  return 'square'
}

export function buildDemoPreview(): BandPreviewPayload {
  const lattice: LatticeType = 'hex'
  const epsBg = 8.4
  const rOverA = 0.26
  const kPath = Array.from({ length: 80 }, (_, idx) => idx / 79)
  const te = kPath.map((k) => 0.22 + 0.08 * Math.sin(2.5 * Math.PI * k) + 0.015 * Math.sin(6 * Math.PI * k))
  const tm = kPath.map((k) => 0.35 + 0.06 * Math.cos(2.0 * Math.PI * k + 0.5) - 0.01 * Math.sin(8 * k))

  return {
    lattice,
    epsBg,
    rOverA,
    kPath,
    series: [
      { id: 'TE-1', label: 'TE band (demo)', color: '#c97a7a', values: te },
      { id: 'TM-1', label: 'TM band (demo)', color: '#7096b7', values: tm },
    ],
    note: 'Demo pair • hex lattice, ε_bg = 8.40, r/a = 0.260',
  }
}