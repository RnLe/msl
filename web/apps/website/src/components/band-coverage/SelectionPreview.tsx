'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import clsx from 'clsx'
import { ChevronRight, Download, LineChart, Maximize2, ScatterChart, X } from 'lucide-react'
import type { BandDataPoint, CoverageMode, CoveragePoint, LatticeType } from './types'
import { COLORS } from './palette'
import { useBandCoverageStore } from './store'
import {
  SPEED_OF_LIGHT,
  describeFrequency,
  formatLengthWithBestPrefix,
  formatLengthWithPrefix,
  formatPhysicalRadius,
  normalizedValueToFrequency,
  resolveLatticeConstantMeters,
} from './units'
import { findNearestStop, getSymmetryStops } from './symmetry'
import { HighSymmetryVisualization2D } from '../HighSymmetryVisualization2D'

export type BandSeries = {
  id: string
  label: string
  color: string
  values: number[]
}

type MaterialHighlight = {
  color: string
  frequencyWindows: Array<[number, number]>
}

type ChartPointMeta = {
  key: string
  bandId: string
  pointIndex: number
  x: number
  y: number
  color: string
  omega: number
  kLabel: string
  polarization: string
  bandNumber: number
  kRatio: number
  kValue: number
  lattice: LatticeType
}

export type BandPreviewPayload = {
  lattice: LatticeType
  epsBg: number
  rOverA: number
  kPath: number[]
  series: BandSeries[]
  note?: string
}

type SelectionPreviewProps = {
  point: CoveragePoint | null
  mode: CoverageMode
  bandState: { status: 'idle' | 'loading' | 'ready' | 'error' | 'prefetching'; data?: BandPreviewPayload; message?: string }
  demo: BandPreviewPayload
  materialHighlight?: MaterialHighlight | null
}

const DEMO_PREVIEW_DELAY_MS = 50
const PRIMARY_BG = '#111111'
const SECONDARY_BG = '#1b1b1b'

export function SelectionPreview({ point, mode, bandState, demo, materialHighlight }: SelectionPreviewProps) {
  const isDemo = mode === 'offline'
  const [demoLoading, setDemoLoading] = useState(false)
  const [isDiagramModalOpen, setDiagramModalOpen] = useState(false)
  const [diagramMode, setDiagramMode] = useState<'lines' | 'points'>('lines')
  const [isDownloadMenuOpen, setDownloadMenuOpen] = useState(false)
  const [pngBranchOpen, setPngBranchOpen] = useState(false)
  const downloadButtonRef = useRef<HTMLButtonElement | null>(null)
  const downloadMenuRef = useRef<HTMLDivElement | null>(null)

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

  useEffect(() => {
    if (!isDiagramModalOpen) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setDiagramModalOpen(false)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isDiagramModalOpen])

  useEffect(() => {
    if (isDiagramModalOpen) return
    setDownloadMenuOpen(false)
    setPngBranchOpen(false)
  }, [isDiagramModalOpen])

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
      console.warn('[SelectionPreview] No preview data available for CSV export.')
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

      {isDiagramModalOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          role="dialog"
          aria-modal="true"
          onClick={(event) => {
            if (event.target === event.currentTarget) setDiagramModalOpen(false)
          }}
        >
          <div className="relative w-full max-w-5xl rounded-2xl border border-white/10 p-6 shadow-2xl" style={{ backgroundColor: PRIMARY_BG }}>
            <button
              type="button"
              onClick={() => setDiagramModalOpen(false)}
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
                      <div
                        className="relative"
                        onMouseEnter={() => setPngBranchOpen(true)}
                      >
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
              lattice={geometrySource?.lattice ?? preview?.lattice ?? 'square'}
              kPath={preview?.kPath}
              series={preview?.series}
              loading={loading}
              size="large"
              mode={diagramMode}
              onToggleMode={() => setDiagramMode((mode) => (mode === 'lines' ? 'points' : 'lines'))}
              radius={geometrySource?.rOverA ?? preview?.rOverA}
              epsBg={geometrySource?.epsBg ?? preview?.epsBg}
              showParams
              materialHighlight={materialHighlight ?? null}
            />
          </div>
        </div>
      ) : null}
    </div>
  )
}

const GEOMETRY_WIDTH = 260
const GEOMETRY_HEIGHT = 220

type GeometryCircle = { key: string; x: number; y: number; row: number; col: number }
type GeometryLineMetadata = {
  centerPoint: GeometryCircle | null
  rightNeighbor: GeometryCircle | null
  neighborDistance: number | null
  bottomLeft: GeometryCircle | null
  bottomRight: GeometryCircle | null
  bottomRatio: number | null
  bottomBaseline: number | null
}

const buildBottomSpan = (
  left: GeometryCircle | null,
  right: GeometryCircle | null
): [GeometryCircle, GeometryCircle] | null => {
  if (!left || !right) return null
  return [left, right]
}

const EMPTY_LINE_GEOMETRY: GeometryLineMetadata = {
  centerPoint: null,
  rightNeighbor: null,
  neighborDistance: null,
  bottomLeft: null,
  bottomRight: null,
  bottomRatio: null,
  bottomBaseline: null,
}

function GeometryPreview({ lattice, rOverA }: { lattice: LatticeType; rOverA: number }) {
  const { circles, radius } = useMemo<{ circles: GeometryCircle[]; radius: number }>(
    () => buildGeometry(lattice, rOverA),
    [lattice, rOverA]
  )
  const latticeConstantValue = useBandCoverageStore((store) => store.latticeConstantValue)
  const latticeConstantPrefix = useBandCoverageStore((store) => store.latticeConstantPrefix)
  const latticeConstantMeters = useMemo(
    () => resolveLatticeConstantMeters(latticeConstantValue, latticeConstantPrefix),
    [latticeConstantValue, latticeConstantPrefix]
  )
  const latticeLabel = useMemo(() => {
    if (latticeConstantValue === null || !latticeConstantPrefix) return null
    return formatLengthWithPrefix(latticeConstantValue, latticeConstantPrefix)
  }, [latticeConstantValue, latticeConstantPrefix])

  const circlesForDisplay = useMemo<GeometryCircle[]>(() => {
    if (!latticeLabel || !circles.length) return circles
    const maxY = circles.reduce((acc, circle) => Math.max(acc, circle.y), -Infinity)
    const threshold = 0.6
    const trimmed = circles.filter((circle) => maxY - circle.y > threshold)
    return trimmed.length ? trimmed : circles
  }, [circles, latticeLabel])

  const lineGeometry = useMemo<GeometryLineMetadata>(() => {
    if (!circlesForDisplay.length) return EMPTY_LINE_GEOMETRY
    const centerRef = { x: GEOMETRY_WIDTH / 2, y: GEOMETRY_HEIGHT / 2 }
    let centerPoint: GeometryCircle | null = null
    let minCenterDist = Infinity
    circlesForDisplay.forEach((circle: GeometryCircle) => {
      const dist = Math.hypot(circle.x - centerRef.x, circle.y - centerRef.y)
      if (dist < minCenterDist) {
        minCenterDist = dist
        centerPoint = circle
      }
    })
    if (!centerPoint) return EMPTY_LINE_GEOMETRY
    // NOTE: When we introduce oblique bases we should rotate the neighbor search
    // vectors so that these guides remain aligned to the primitive lattice.
    let rightNeighbor: GeometryCircle | null = null
    let neighborDistance = Infinity
    circlesForDisplay.forEach((circle: GeometryCircle) => {
      if (circle.x <= centerPoint!.x) return
      const dist = Math.hypot(circle.x - centerPoint!.x, circle.y - centerPoint!.y)
      if (dist < neighborDistance) {
        neighborDistance = dist
        rightNeighbor = circle
      }
    })
    if (!rightNeighbor || !Number.isFinite(neighborDistance) || neighborDistance <= 0) {
      rightNeighbor = null
      neighborDistance = Infinity
    }
    const maxY = circlesForDisplay.reduce((acc, circle: GeometryCircle) => Math.max(acc, circle.y), -Infinity)
    const bottomPoints = circlesForDisplay.filter((circle: GeometryCircle) => Math.abs(circle.y - maxY) < 0.6)
    let bottomLeft: GeometryCircle | null = null
    let bottomRight: GeometryCircle | null = null
    bottomPoints.forEach((circle: GeometryCircle) => {
      if (!bottomLeft || circle.x < bottomLeft.x) bottomLeft = circle
      if (!bottomRight || circle.x > bottomRight.x) bottomRight = circle
    })
    const bottomSpan = buildBottomSpan(bottomLeft, bottomRight)
    const bottomDistance = bottomSpan ? Math.hypot(bottomSpan[1].x - bottomSpan[0].x, bottomSpan[1].y - bottomSpan[0].y) : null
    const bottomRatio = bottomDistance && Number.isFinite(bottomDistance) && neighborDistance !== Infinity ? bottomDistance / neighborDistance : null
    const bottomBaseline = bottomSpan ? Math.max(bottomSpan[0].y, bottomSpan[1].y) + 12 : null
    return {
      centerPoint,
      rightNeighbor,
      neighborDistance: neighborDistance === Infinity ? null : neighborDistance,
      bottomLeft,
      bottomRight,
      bottomRatio,
      bottomBaseline,
    }
  }, [circlesForDisplay])

  const showLatticeLines = Boolean(latticeConstantMeters && latticeLabel && lineGeometry.centerPoint && lineGeometry.rightNeighbor)
  const bottomLengthLabel = useMemo(() => {
    if (!showLatticeLines || !latticeConstantMeters || !lineGeometry.bottomRatio) return null
    return formatLengthWithBestPrefix(latticeConstantMeters * lineGeometry.bottomRatio)
  }, [showLatticeLines, latticeConstantMeters, lineGeometry.bottomRatio])

  return (
    <div className="space-y-3">
      <div className="text-center text-base font-semibold" style={{ color: COLORS.textPrimary }}>
        Geometry preview
      </div>
      <div className="w-full" style={{ aspectRatio: `${GEOMETRY_WIDTH} / ${GEOMETRY_HEIGHT}` }}>
        <svg viewBox={`0 0 ${GEOMETRY_WIDTH} ${GEOMETRY_HEIGHT}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
          <rect width={GEOMETRY_WIDTH} height={GEOMETRY_HEIGHT} rx="16" fill={PRIMARY_BG} />
          {circlesForDisplay.map((circle) => (
            <circle key={circle.key} cx={circle.x} cy={circle.y} r={radius} fill="#ffffff" fillOpacity={0.9} />
          ))}
          {showLatticeLines && lineGeometry.centerPoint && lineGeometry.rightNeighbor ? (
            <g>
              <line
                x1={lineGeometry.centerPoint.x}
                y1={lineGeometry.centerPoint.y}
                x2={lineGeometry.rightNeighbor.x}
                y2={lineGeometry.rightNeighbor.y}
                stroke={COLORS.accent}
                strokeWidth={2}
                strokeLinecap="round"
              />
              {latticeLabel ? (
                <text
                  x={(lineGeometry.centerPoint.x + lineGeometry.rightNeighbor.x) / 2}
                  y={Math.min(lineGeometry.centerPoint.y, lineGeometry.rightNeighbor.y) - 8}
                  textAnchor="middle"
                  fontSize={12}
                  fill="#8d93a4"
                >
                  {latticeLabel}
                </text>
              ) : null}
            </g>
          ) : null}
          {showLatticeLines && lineGeometry.bottomLeft && lineGeometry.bottomRight && lineGeometry.bottomBaseline ? (
            <g>
              <line
                x1={lineGeometry.bottomLeft.x}
                y1={lineGeometry.bottomBaseline}
                x2={lineGeometry.bottomRight.x}
                y2={lineGeometry.bottomBaseline}
                stroke={COLORS.accent}
                strokeWidth={2}
                strokeLinecap="round"
              />
              {bottomLengthLabel ? (
                <text
                  x={(lineGeometry.bottomLeft.x + lineGeometry.bottomRight.x) / 2}
                  y={lineGeometry.bottomBaseline + 14}
                  textAnchor="middle"
                  fontSize={12}
                  fill="#ffffff"
                >
                  {bottomLengthLabel}
                </text>
              ) : null}
            </g>
          ) : null}
        </svg>
      </div>
    </div>
  )
}

type BandDiagramPlotProps = {
  lattice: LatticeType
  kPath?: number[]
  series?: BandSeries[]
  loading?: boolean
  onExpand?: () => void
  size?: 'default' | 'large'
  mode?: 'lines' | 'points'
  onToggleMode?: () => void
  radius?: number
  epsBg?: number
  showParams?: boolean
  materialHighlight?: MaterialHighlight | null
}

function BandDiagramPlot({
  lattice,
  kPath = [],
  series = [],
  loading = false,
  onExpand,
  size = 'default',
  mode = 'lines',
  onToggleMode,
  radius,
  epsBg,
  showParams = true,
  materialHighlight,
}: BandDiagramPlotProps) {
  const latticeConstantValue = useBandCoverageStore((store) => store.latticeConstantValue)
  const latticeConstantPrefix = useBandCoverageStore((store) => store.latticeConstantPrefix)
  const dimensions = size === 'large' ? { width: 900, height: 520 } : { width: 320, height: 220 }
  const padding = size === 'large' ? { top: 36, right: 32, bottom: 44, left: 80 } : { top: 22, right: 14, bottom: 36, left: 52 }
  const width = dimensions.width
  const height = dimensions.height
  const innerWidth = width - padding.left - padding.right
  const innerHeight = height - padding.top - padding.bottom
  const isLarge = size === 'large'
  const plotBackground = isLarge ? PRIMARY_BG : SECONDARY_BG
  const expandAvailable = typeof onExpand === 'function'
  const canToggleMode = typeof onToggleMode === 'function'
  const formattedRadius = Number.isFinite(radius) ? (radius as number).toFixed(3) : null
  const formattedEps = Number.isFinite(epsBg) ? (epsBg as number).toFixed(3) : null
  const formattedLattice = lattice ? lattice.charAt(0).toUpperCase() + lattice.slice(1) : null
  const showParameterTuple = showParams && Boolean(formattedLattice || formattedRadius || formattedEps)
  const latticeConstantMeters = useMemo(
    () => resolveLatticeConstantMeters(latticeConstantValue, latticeConstantPrefix),
    [latticeConstantValue, latticeConstantPrefix]
  )
  const physicalRadiusLabel = useMemo(
    () => formatPhysicalRadius(typeof radius === 'number' ? radius : null, latticeConstantValue, latticeConstantPrefix),
    [radius, latticeConstantValue, latticeConstantPrefix]
  )
  const frequencyPerUnit = useMemo(() => normalizedValueToFrequency(1, latticeConstantMeters), [latticeConstantMeters])
  const axisLabelUnit = useMemo(() => {
    if (!frequencyPerUnit) return null
    const descriptor = describeFrequency(frequencyPerUnit)
    return descriptor?.unit ?? null
  }, [frequencyPerUnit])

  const plottedSeries = useMemo(() => {
    if (!series?.length) return []
    const EPSILON = 1e-6
    return series.filter((band) => {
      if (!band.values?.length) return false
      const polarization = extractPolarization(band.label)
      const allZero = band.values.every((value) => Math.abs(value) < EPSILON)
      if (polarization === 'TM' && allZero) return false
      return true
    })
  }, [series])

  const legendEntries = useMemo(() => {
    const entries = new Map<string, string>()
    for (const band of plottedSeries) {
      const polarization = extractPolarization(band.label)
      if (entries.has(polarization)) continue
      entries.set(polarization, band.color)
    }
    return Array.from(entries.entries()).map(([polarization, color]) => ({ polarization, color }))
  }, [plottedSeries])

  const bounds = useMemo(() => {
    if (!plottedSeries.length) return { min: 0, max: 1 }
    const values = plottedSeries.flatMap((band) => band.values)
    if (!values.length) return { min: 0, max: 1 }
    const dataMin = Math.min(...values)
    const dataMax = Math.max(...values)
    return {
      min: Math.min(0, dataMin),
      max: Math.max(1, dataMax),
    }
  }, [plottedSeries])

  const normalizedK = useMemo(() => {
    if (!kPath?.length) return []
    const start = kPath[0]
    const end = kPath[kPath.length - 1]
    const span = end - start || 1
    return kPath.map((value) => (value - start) / span)
  }, [kPath])

  const fallbackRatioStep = kPath && kPath.length > 1 ? 1 / (kPath.length - 1) : 0

  const symmetryStops = useMemo(() => getSymmetryStops(lattice), [lattice])

  const valueToY = useCallback(
    (value: number) => {
      const span = bounds.max - bounds.min || 1
      const ratio = (value - bounds.min) / span
      return padding.top + (1 - ratio) * innerHeight
    },
    [bounds.max, bounds.min, innerHeight, padding.top]
  )

  const materialBands = useMemo(() => {
    if (!materialHighlight?.frequencyWindows?.length || !frequencyPerUnit) return []
    return materialHighlight.frequencyWindows
      .map((window, idx) => {
        const [startHz, endHz] = window
        if (!Number.isFinite(startHz) || !Number.isFinite(endHz) || startHz <= 0 || endHz <= 0) return null
        const normalizedStart = startHz / frequencyPerUnit
        const normalizedEnd = endHz / frequencyPerUnit
        const minValue = Math.min(normalizedStart, normalizedEnd)
        const maxValue = Math.max(normalizedStart, normalizedEnd)
        if (maxValue < bounds.min || minValue > bounds.max) return null
        const clampedMin = Math.max(bounds.min, minValue)
        const clampedMax = Math.min(bounds.max, maxValue)
        if (clampedMax <= clampedMin) return null
        const yTop = valueToY(clampedMax)
        const yBottom = valueToY(clampedMin)
        const height = Math.max(0, yBottom - yTop)
        if (height <= 0) return null
        return { id: `material-band-${idx}`, y: yTop, height }
      })
      .filter((band): band is { id: string; y: number; height: number } => Boolean(band))
  }, [materialHighlight, frequencyPerUnit, bounds.min, bounds.max, valueToY])

  const ratioToX = useCallback(
    (ratio: number) => padding.left + Math.min(1, Math.max(0, ratio)) * innerWidth,
    [innerWidth, padding.left]
  )

  const formatYAxisTick = useCallback(
    (value: number) => {
      if (!frequencyPerUnit) return value.toFixed(2)
      const physicalValue = value * frequencyPerUnit
      const descriptor = describeFrequency(Math.abs(physicalValue))
      if (!descriptor) return value.toFixed(2)
      const sign = physicalValue < 0 ? '-' : ''
      return `${sign}${descriptor.value}`
    },
    [frequencyPerUnit]
  )

  const points = useMemo<ChartPointMeta[]>(() => {
    if (!plottedSeries.length) return []
    return plottedSeries.flatMap((band) => {
      const polarization = extractPolarization(band.label)
      const bandNumber = extractBandNumber(band.label)
      return band.values.map((value, idx) => {
        const ratio = normalizedK[idx] ?? idx * fallbackRatioStep
        const kValue = kPath[idx] ?? ratio
        const nearestStop = findNearestStop(symmetryStops, ratio)
        return {
          key: `${band.id}-${idx}`,
          bandId: band.id,
          pointIndex: idx,
          x: ratioToX(ratio),
          y: valueToY(value),
          color: band.color,
          omega: value,
          kLabel: nearestStop.label,
          polarization,
          bandNumber,
          kRatio: ratio,
          kValue,
          lattice,
        }
      })
    })
  }, [fallbackRatioStep, kPath, lattice, normalizedK, plottedSeries, ratioToX, symmetryStops, valueToY])

  const pointRadius = useMemo(() => {
    if (mode !== 'points') return 0
    const totalPoints = points.length || 1
    const density = totalPoints / (innerWidth * innerHeight || 1)
    const base = size === 'large' ? 2.1 : 1.3
    const factor = Math.max(0.55, Math.min(1.1, 0.45 / Math.sqrt(Math.max(density, 1e-6))))
    return base * factor
  }, [mode, points.length, innerWidth, innerHeight, size])

  const setBandHovered = useBandCoverageStore((store) => store.setBandHovered)
  const setBandSelected = useBandCoverageStore((store) => store.setBandSelected)
  const storeSelectedPoint = useBandCoverageStore((store) => store.bandSelected)

  const toBandDataPoint = useCallback(
    (meta: ChartPointMeta): BandDataPoint => ({
      lattice: meta.lattice,
      pointKey: meta.key,
      bandId: meta.bandId,
      bandNumber: meta.bandNumber,
      polarization: meta.polarization,
      omega: meta.omega,
      kLabel: meta.kLabel,
      kRatio: meta.kRatio,
      kValue: meta.kValue,
      kIndex: meta.pointIndex,
      color: meta.color,
    }),
    []
  )

  const [hoverPoint, setHoverPoint] = useState<null | (ChartPointMeta & { clientX: number; clientY: number })>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const diagramContainerRef = useRef<HTMLDivElement | null>(null)
  const hoverFrame = useRef<number | null>(null)
  const pendingHover = useRef<null | (ChartPointMeta & { clientX: number; clientY: number })>(null)
  const selectedPointMeta = useMemo(() => {
    if (!storeSelectedPoint) return null
    return points.find((point) => point.key === storeSelectedPoint.pointKey) ?? null
  }, [points, storeSelectedPoint])

  const markerPoints = useMemo(() => {
    const entries: Array<{ meta: ChartPointMeta; variant: 'hover' | 'selected' }> = []
    if (selectedPointMeta) entries.push({ meta: selectedPointMeta, variant: 'selected' })
    if (hoverPoint) {
      if (selectedPointMeta?.key === hoverPoint.key) {
        entries[entries.length - 1] = { meta: hoverPoint, variant: 'hover' }
      } else {
        entries.push({ meta: hoverPoint, variant: 'hover' })
      }
    }
    return entries
  }, [hoverPoint, selectedPointMeta])

  const flushHover = useCallback(() => {
    hoverFrame.current = null
    const next = pendingHover.current ?? null
    pendingHover.current = null
    setHoverPoint(next)
    setBandHovered(next ? toBandDataPoint(next) : null)
  }, [setBandHovered, toBandDataPoint])

  const scheduleHover = useCallback(
    (payload: (ChartPointMeta & { clientX: number; clientY: number }) | null) => {
      pendingHover.current = payload
      if (hoverFrame.current !== null) return
      hoverFrame.current = requestAnimationFrame(flushHover)
    },
    [flushHover]
  )

  const pickNearestPoint = useCallback(
    (clientX: number, clientY: number) => {
      if (!svgRef.current || !points.length) return null
      const rect = svgRef.current.getBoundingClientRect()
      const localX = ((clientX - rect.left) / rect.width) * width
      const localY = ((clientY - rect.top) / rect.height) * height
      let best: ChartPointMeta | null = null
      let bestDist = Infinity
      for (const point of points) {
        const dx = point.x - localX
        const dy = point.y - localY
        const dist = dx * dx + dy * dy
        if (dist < bestDist) {
          best = point
          bestDist = dist
        }
      }
      if (!best) return null
      return {
        ...best,
        clientX: clientX - rect.left,
        clientY: clientY - rect.top,
      }
    },
    [height, points, width]
  )

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      const payload = pickNearestPoint(event.clientX, event.clientY)
      if (!payload) return
      scheduleHover(payload)
    },
    [pickNearestPoint, scheduleHover]
  )

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      const payload = pickNearestPoint(event.clientX, event.clientY)
      if (!payload) return
      setBandSelected(toBandDataPoint(payload))
      scheduleHover(payload)
    },
    [pickNearestPoint, scheduleHover, setBandSelected, toBandDataPoint]
  )

  useEffect(() => {
    return () => {
      if (hoverFrame.current !== null) cancelAnimationFrame(hoverFrame.current)
      setBandHovered(null)
    }
  }, [setBandHovered])

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target
      const container = diagramContainerRef.current
      if (!container) return
      if (target instanceof Node && container.contains(target)) {
        return
      }
      setBandSelected(null)
    }

    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [setBandSelected])

  const pathD = useCallback(
    (values: number[]) =>
      values
        .map((value, idx) => {
          const ratio = normalizedK[idx] ?? idx * fallbackRatioStep
          return `${idx === 0 ? 'M' : 'L'} ${ratioToX(ratio)} ${valueToY(value)}`
        })
        .join(' '),
    [fallbackRatioStep, normalizedK, ratioToX, valueToY]
  )

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className={clsx('font-semibold', isLarge ? 'text-2xl' : 'text-lg')} style={{ color: COLORS.textPrimary }}>
            Band diagram
          </div>
          {canToggleMode ? (
            <button
              type="button"
              aria-label={mode === 'lines' ? 'Show data point dots' : 'Show band lines'}
              onClick={onToggleMode}
              className={clsx('rounded-full bg-transparent text-white transition hover:bg-white/10', isLarge ? 'p-3' : 'p-2.5')}
            >
              {mode === 'lines' ? <ScatterChart className={clsx(isLarge ? 'h-5 w-5' : 'h-4 w-4')} /> : <LineChart className={clsx(isLarge ? 'h-5 w-5' : 'h-4 w-4')} />}
            </button>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {expandAvailable ? (
            <button
              type="button"
              aria-label="Expand band diagram"
              onClick={onExpand!}
              className={clsx('rounded-full bg-transparent text-white transition hover:bg-white/10', isLarge ? 'p-3' : 'p-2.5')}
            >
              <Maximize2 className={clsx(isLarge ? 'h-5 w-5' : 'h-4 w-4')} />
            </button>
          ) : null}
        </div>
      </div>
      {showParameterTuple ? (
        <div className={clsx('font-mono flex flex-wrap gap-10', isLarge ? 'text-sm' : 'text-xs')} style={{ color: 'whitesmoke' }}>
          {formattedLattice ? <span className="capitalize">{formattedLattice} lattice</span> : null}
          {formattedRadius ? (
            <span>
              r/a = {formattedRadius}
              {physicalRadiusLabel ? ` (${physicalRadiusLabel})` : ''}
            </span>
          ) : null}
          {formattedEps ? (
            <span>
              <MathSymbol symbol="ε" subscript="bg" /> = {formattedEps}
            </span>
          ) : null}
        </div>
      ) : null}
      <div ref={diagramContainerRef} className="relative w-full" style={{ aspectRatio: `${width} / ${height}` }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid meet"
          onPointerLeave={() => scheduleHover(null)}
          onPointerMove={handlePointerMove}
          onPointerDown={handlePointerDown}
        >
          <rect x={padding.left} y={padding.top} width={innerWidth} height={innerHeight} fill={plotBackground} stroke={COLORS.border} />
          {materialBands.map((band) => (
            <rect
              key={band.id}
              x={padding.left}
              y={band.y}
              width={innerWidth}
              height={band.height}
              fill={materialHighlight?.color ?? '#ffffff'}
              opacity={isLarge ? 0.22 : 0.18}
            />
          ))}
          {symmetryStops.map((stop, index) => {
            const x = ratioToX(stop.position)
            const isEdge = index === 0 || index === symmetryStops.length - 1
            return (
              <g key={`${stop.label}-${index}`}>
                <line
                  x1={x}
                  x2={x}
                  y1={padding.top}
                  y2={padding.top + innerHeight}
                  stroke="#ffffff"
                  strokeOpacity={isEdge ? 0.08 : 0.15}
                  strokeWidth={isEdge ? 1 : 1.5}
                />
                <text
                  x={x}
                  y={padding.top + innerHeight + (isLarge ? 26 : 18)}
                  textAnchor="middle"
                  fontSize={isLarge ? 14 : 11}
                  fill={COLORS.textMuted}
                >
                  {stop.label}
                </text>
              </g>
            )
          })}
          {[0, 0.5, 1].map((ratio) => {
            const value = bounds.min + ratio * (bounds.max - bounds.min)
            const y = valueToY(value)
            return (
              <text
                key={`h-${ratio}`}
                x={padding.left - 12}
                y={y + 4}
                textAnchor="end"
                fontSize={isLarge ? 13 : 11}
                fill={COLORS.textMuted}
              >
                {formatYAxisTick(value)}
              </text>
            )
          })}
          {mode === 'lines'
            ? plottedSeries.map((band) => (
                <path key={band.id} d={pathD(band.values)} fill="none" stroke={band.color} strokeWidth={1.05} strokeLinecap="round" />
              ))
            : points.map((meta) => (
                <circle key={`dot-${meta.key}`} cx={meta.x} cy={meta.y} r={pointRadius} fill={meta.color} fillOpacity={0.85} />
              ))}
          {markerPoints.map(({ meta, variant }) => (
            <circle
              key={`${meta.key}-${variant}`}
              cx={meta.x}
              cy={meta.y}
              r={variant === 'hover' ? 3.2 : 2.6}
              fill="#ffffff"
              stroke={meta.color}
              strokeWidth={variant === 'hover' ? 1.8 : 1.2}
              opacity={variant === 'hover' ? 1 : 0.9}
            />
          ))}
          <text
            x={padding.left - (isLarge ? 46 : 40)}
            y={padding.top + innerHeight / 2}
            textAnchor="middle"
            fontSize={isLarge ? 18 : 13}
            fill={COLORS.textMuted}
            transform={`rotate(-90 ${padding.left - (isLarge ? 46 : 40)} ${padding.top + innerHeight / 2})`}
          >
            ωa/2πc
            {axisLabelUnit ? (
              <tspan fill="#ffffff"> ({axisLabelUnit})</tspan>
            ) : null}
          </text>
        </svg>
        {hoverPoint && (
          <div
            className="pointer-events-none absolute rounded-md px-2 py-1 text-[11px]"
            style={{
              left: hoverPoint.clientX,
              top: hoverPoint.clientY,
              transform: 'translate(-50%, -110%)',
              backgroundColor: 'rgba(15,18,24,0.95)',
              border: `1px solid ${COLORS.border}`,
              color: COLORS.textPrimary,
              whiteSpace: 'nowrap',
              zIndex: 20,
              boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
            }}
          >
            <div style={{ color: hoverPoint.color, fontWeight: 600 }}>{hoverPoint.polarization}</div>
            <div>{`Band ${hoverPoint.bandNumber}`}</div>
            <div style={{ color: COLORS.textMuted }}>ω = {hoverPoint.omega.toFixed(3)}</div>
            {frequencyPerUnit && hoverPoint
              ? (() => {
                  const physicalValue = hoverPoint.omega * frequencyPerUnit
                  if (!Number.isFinite(physicalValue)) return null
                  const descriptor = describeFrequency(Math.abs(physicalValue))
                  const sign = physicalValue < 0 ? '-' : ''
                  const wavelengthMeters = Math.abs(physicalValue) > 0 ? SPEED_OF_LIGHT / Math.abs(physicalValue) : null
                  const wavelengthLabel = formatLengthWithBestPrefix(wavelengthMeters)
                  if (!descriptor && !wavelengthLabel) return null
                  return (
                    <>
                      {descriptor ? (
                        <div style={{ color: COLORS.textMuted }}>
                          ≈ {sign}
                          {descriptor.value} {descriptor.unit}
                        </div>
                      ) : null}
                      {wavelengthLabel ? (
                        <div style={{ color: COLORS.textMuted }}>λ ≈ {wavelengthLabel}</div>
                      ) : null}
                    </>
                  )
                })()
              : null}
          </div>
        )}
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/40 text-xs" style={{ color: COLORS.textPrimary }}>
            <LoadingSpinner />
            <span>Loading band diagram…</span>
          </div>
        )}
      </div>
      <div className={clsx('flex flex-wrap justify-center gap-3', isLarge ? 'text-sm' : 'text-xs')} style={{ color: COLORS.textMuted }}>
        {legendEntries.map((entry) => (
          <span key={entry.polarization} className="flex items-center gap-2">
            <span className="h-2 w-6 rounded-sm" style={{ backgroundColor: entry.color }} />
            {entry.polarization} mode
          </span>
        ))}
      </div>
    </div>
  )
}

function LoadingSpinner() {
  return <span className="h-6 w-6 animate-spin rounded-full border-2 border-current border-t-transparent" />
}

function extractPolarization(label: string) {
  const match = label.match(/TE|TM/i)
  return match ? match[0].toUpperCase() : 'TM'
}

function extractBandNumber(label: string) {
  const match = label.match(/(\d+)/)
  return match ? Number(match[1]) : 1
}

function buildGeometry(lattice: LatticeType, rOverA: number): { circles: GeometryCircle[]; radius: number } {
  const width = 260
  const height = 220
  const padding = 16
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const safeRadius = Number.isFinite(rOverA) ? rOverA : 0.2

  const baseRows = 7
  const rows = lattice === 'hex' ? baseRows + 2 : baseRows
  const cols = 9

  if (lattice === 'hex') {
    const horizontalSpacing = innerWidth / (cols - 0.5)
    const verticalSpacing = innerHeight / (rows - 1)
    const angleFactor = Math.sin(Math.PI / 3)
    const adjustedVertical = verticalSpacing * angleFactor
    const spacing = Math.min(horizontalSpacing, adjustedVertical)
    const offsetX = padding + (innerWidth - spacing * (cols - 0.5)) / 2
    const offsetY = padding + (innerHeight - spacing * angleFactor * (rows - 1)) / 2

    const points: GeometryCircle[] = []
    for (let row = 0; row < rows; row += 1) {
      const y = offsetY + row * spacing * angleFactor
      for (let col = 0; col < cols; col += 1) {
        let x = offsetX + col * spacing
        if (row % 2 === 1) x += spacing / 2
        if (x < padding || x > width - padding) continue
        points.push({ key: `hex-${row}-${col}`, x, y, row, col })
      }
    }

    const cellSize = spacing / 2
    const radius = computeRadius(cellSize, safeRadius)
    return { circles: points, radius }
  }

  const spacingX = innerWidth / (cols - 1)
  const spacingY = innerHeight / (rows - 1)
  const spacing = Math.min(spacingX, spacingY)
  const offsetX = padding + (innerWidth - spacing * (cols - 1)) / 2
  const offsetY = padding + (innerHeight - spacing * (rows - 1)) / 2

  const points: GeometryCircle[] = []
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      points.push({
        key: `sq-${row}-${col}`,
        x: offsetX + col * spacing,
        y: offsetY + row * spacing,
        row,
        col,
      })
    }
  }

  const radius = computeRadius(spacing / 2, safeRadius)
  return { circles: points, radius }
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

function MathSymbol({ symbol, subscript }: { symbol: string; subscript?: string }) {
  return (
    <span className="inline-flex items-baseline">
      <span>{symbol}</span>
      {subscript ? <sub className="ml-px text-[0.65em]">{subscript}</sub> : null}
    </span>
  )
}

function toVizLattice(lattice: LatticeType): 'square' | 'hexagonal' {
  if (lattice === 'hex' || lattice === 'hexagonal') return 'hexagonal'
  return 'square'
}


function computeRadius(cellSize: number, ratio: number) {
  const normalized = clamp01((ratio - 0.1) / (0.48 - 0.1))
  return Math.max(3, cellSize * (0.35 + 0.55 * normalized))
}

function clamp01(value: number) {
  if (Number.isNaN(value)) return 0
  return Math.min(1, Math.max(0, value))
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