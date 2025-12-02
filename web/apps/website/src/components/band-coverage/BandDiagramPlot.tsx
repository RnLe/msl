'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import clsx from 'clsx'
import { LineChart, Maximize2, ScatterChart } from 'lucide-react'
import type { BandDataPoint, LatticeType } from './types'
import { COLORS } from './palette'
import { useBandCoverageStore } from './store'
import {
  SPEED_OF_LIGHT,
  describeFrequency,
  formatLengthWithBestPrefix,
  formatPhysicalRadius,
  normalizedValueToFrequency,
  resolveLatticeConstantMeters,
} from './units'
import { findNearestStop, formatSymmetryLabel, getSymmetryStops } from './symmetry'

export type BandSeries = {
  id: string
  label: string
  color: string
  values: number[]
}

export type MaterialHighlight = {
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

const PRIMARY_BG = '#111111'
const SECONDARY_BG = '#1b1b1b'

export type BandDiagramPlotProps = {
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

export function BandDiagramPlot({
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
        const formattedLabel = formatSymmetryLabel(lattice, nearestStop.label)
        return {
          key: `${band.id}-${idx}`,
          bandId: band.id,
          pointIndex: idx,
          x: ratioToX(ratio),
          y: valueToY(value),
          color: band.color,
          omega: value,
          kLabel: formattedLabel,
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
            const label = formatSymmetryLabel(lattice, stop.label)
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
                  {label}
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

function MathSymbol({ symbol, subscript }: { symbol: string; subscript?: string }) {
  return (
    <span className="inline-flex items-baseline">
      <span>{symbol}</span>
      {subscript ? <sub className="ml-px text-[0.65em]">{subscript}</sub> : null}
    </span>
  )
}
