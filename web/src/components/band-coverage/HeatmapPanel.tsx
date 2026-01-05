'use client'

import { memo, useCallback, useRef, useState } from 'react'
import type {
  CSSProperties,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from 'react'
import clsx from 'clsx'
import { Library, X } from 'lucide-react'
import { HeatmapCanvas } from './HeatmapCanvas'
import type { CoverageMode, CoveragePoint, CoverageResponse, LatticeType } from './types'
import { COLORS, STATUS_COLORS, STATUS_LABELS } from './palette'
import { formatPhysicalRadius } from './units'
import type { LatticeConstantPrefix } from './store'
import type { MaterialLibraryEntry } from './library'

export type SliderMarker = {
  index: number
  color: string
}

export type MaterialAccent = {
  background: string
  text?: string
}

export type HeatmapPanelProps = {
  coverage: CoverageResponse
  activeLattice: LatticeType
  height?: number
  hasData: boolean
  mode: CoverageMode
  transitioning: boolean
  hoveredPoint: CoveragePoint | null
  selectedPoint: CoveragePoint | null
  onHover: (point?: CoveragePoint | null) => void
  onSelect: (point: CoveragePoint) => void
  sliderEpsIndex: number
  sliderRIndex: number
  currentEpsBg: number
  currentRadius: number
  latticeConstantValue: number | null
  latticeConstantPrefix: LatticeConstantPrefix | null
  heatmapStatusMessage: string | null
  materialAxisIndex: number | null
  selectedMaterialAccent: MaterialAccent | null
  selectedMaterialEntry: MaterialLibraryEntry | null
  legendLabels: Record<number, string>
  onUpdateSelection: (next: { epsIndex?: number; rIndex?: number }) => void
  onSliderDragStart: (type: 'row' | 'column') => void
  onSliderDragEnd: () => void
  onClearMaterialSelection: () => void
  onOpenMaterialLibrary: () => void
  heatmapRef: React.RefObject<HTMLDivElement | null>
  sliderHeightStyle?: CSSProperties
}

export function HeatmapPanel({
  coverage,
  activeLattice,
  height,
  hasData,
  mode,
  transitioning,
  hoveredPoint,
  selectedPoint,
  onHover,
  onSelect,
  sliderEpsIndex,
  sliderRIndex,
  currentEpsBg,
  currentRadius,
  latticeConstantValue,
  latticeConstantPrefix,
  heatmapStatusMessage,
  materialAxisIndex,
  selectedMaterialAccent,
  selectedMaterialEntry,
  legendLabels,
  onUpdateSelection,
  onSliderDragStart,
  onSliderDragEnd,
  onClearMaterialSelection,
  onOpenMaterialLibrary,
  heatmapRef,
  sliderHeightStyle,
}: HeatmapPanelProps) {
  const matrix = coverage.status?.[activeLattice]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <RadiusInlineLabel
          value={Number.isFinite(currentRadius) ? currentRadius : undefined}
          physicalLabel={formatPhysicalRadius(
            Number.isFinite(currentRadius) ? currentRadius : null,
            latticeConstantValue,
            latticeConstantPrefix
          )}
        />
        <Legend align="end" labels={legendLabels} />
      </div>

      <div className="space-y-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-stretch">
          <div
            className="md:w-4 md:shrink-0 md:self-stretch"
            style={sliderHeightStyle}
          >
            <AxisSliderVertical
              axisLength={coverage.rOverA.length}
              value={sliderRIndex}
              disabled={!hasData}
              onChange={(value) => onUpdateSelection({ rIndex: value })}
              onDragStart={() => onSliderDragStart('column')}
              onDragEnd={onSliderDragEnd}
              className="h-full"
              style={sliderHeightStyle}
            />
          </div>
          <div className="flex-1 min-w-0 flex flex-col gap-3">
            <div ref={heatmapRef} className="relative">
              <HeatmapCanvas
                data={matrix}
                epsBg={coverage.epsBg}
                rOverA={coverage.rOverA}
                lattice={activeLattice}
                height={height}
                disabled={!hasData}
                state={mode}
                transitioning={transitioning}
                hovered={hoveredPoint}
                selected={selectedPoint}
                onHover={onHover}
                onSelect={onSelect}
                materialColumnIndex={materialAxisIndex ?? undefined}
                materialColor={selectedMaterialAccent?.background}
              />
              {heatmapStatusMessage ? (
                <div
                  className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-lg text-xs font-semibold"
                  style={{
                    backgroundColor: 'rgba(6,8,12,0.55)',
                    color: COLORS.textPrimary,
                    border: `1px solid ${COLORS.border}`,
                    textTransform: 'uppercase',
                    letterSpacing: '0.2em',
                  }}
                >
                  {heatmapStatusMessage}
                </div>
              ) : null}
            </div>
            <div className="flex justify-end">
              <AxisSliderHorizontal
                axisLength={coverage.epsBg.length}
                value={sliderEpsIndex}
                disabled={!hasData}
                onChange={(value) => onUpdateSelection({ epsIndex: value })}
                onDragStart={() => onSliderDragStart('row')}
                onDragEnd={onSliderDragEnd}
                className="w-full"
                markers={
                  materialAxisIndex !== null && selectedMaterialAccent
                    ? [{ index: materialAxisIndex, color: selectedMaterialAccent.background }]
                    : undefined
                }
              />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex-1">
                <AxisLabelBlock
                  title="Background Material"
                  detail={
                    <span>
                      <MathSymbol symbol="ε" subscript="bg" />
                    </span>
                  }
                  valueLabel={
                    Number.isFinite(currentEpsBg)
                      ? `${currentEpsBg.toFixed(2)}`
                      : undefined
                  }
                  align="center"
                />
              </div>
              <div className="flex items-center gap-2">
                {selectedMaterialEntry && selectedMaterialAccent ? (
                  <button
                    type="button"
                    onClick={onClearMaterialSelection}
                    className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white transition hover:border-white/40 hover:bg-white/15"
                    title="Clear material highlight"
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: selectedMaterialAccent.background, boxShadow: '0 0 6px rgba(0,0,0,0.35)' }}
                    />
                    <span className="font-semibold uppercase tracking-[0.2em]">{selectedMaterialEntry.label}</span>
                    <X className="h-3 w-3 opacity-70" />
                  </button>
                ) : null}
                <button
                  type="button"
                  onClick={onOpenMaterialLibrary}
                  className="rounded-full p-2 text-white transition hover:bg-white/15"
                  aria-label="Open material library"
                  title="Open material library"
                >
                  <Library className="h-4 w-4" strokeWidth={1.7} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Legend
// ─────────────────────────────────────────────────────────────────────────────

type LegendProps = {
  align?: 'start' | 'end'
  labels?: Record<number, string>
}

function Legend({ align = 'start', labels = STATUS_LABELS }: LegendProps) {
  const orderedCodes = Object.keys(STATUS_COLORS)
    .map((code) => Number(code))
    .sort((a, b) => a - b)
  return (
    <div
      className="flex flex-wrap items-center gap-4 text-xs"
      style={{ color: COLORS.textMuted, justifyContent: align === 'end' ? 'flex-end' : 'flex-start', flex: '1 1 auto' }}
    >
      {orderedCodes.map((code) => (
        <div key={code} className="flex items-center gap-2">
          <span className="h-3 w-6 rounded-sm" style={{ backgroundColor: STATUS_COLORS[code] }} />
          <span>{labels[code] ?? STATUS_LABELS[code] ?? `Status ${code}`}</span>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Axis Sliders
// ─────────────────────────────────────────────────────────────────────────────

type AxisSliderProps = {
  axisLength: number
  value: number
  onChange: (index: number) => void
  disabled?: boolean
  className?: string
  style?: CSSProperties
  onDragStart?: () => void
  onDragEnd?: () => void
  markers?: SliderMarker[]
}

const AxisSliderHorizontal = memo(function AxisSliderHorizontal({
  axisLength,
  value,
  onChange,
  disabled,
  className,
  onDragStart,
  onDragEnd,
  markers,
}: AxisSliderProps) {
  const safeValue = clampIndex(value, axisLength)
  const maxIndex = Math.max(axisLength - 1, 0)
  return (
    <CustomSlider
      orientation="horizontal"
      value={safeValue}
      maxIndex={maxIndex}
      onChange={onChange}
      disabled={disabled || axisLength === 0}
      className={className}
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      markers={markers}
    />
  )
})

const AxisSliderVertical = memo(function AxisSliderVertical({
  axisLength,
  value,
  onChange,
  disabled,
  className,
  style,
  onDragStart,
  onDragEnd,
  markers,
}: AxisSliderProps) {
  const safeValue = clampIndex(value, axisLength)
  const maxIndex = Math.max(axisLength - 1, 0)
  return (
    <div className={clsx('flex h-full w-full items-stretch', className)} style={style}>
      <CustomSlider
        orientation="vertical"
        value={safeValue}
        maxIndex={maxIndex}
        onChange={onChange}
        disabled={disabled || axisLength === 0}
        className="w-full"
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        markers={markers}
      />
    </div>
  )
})

function clampIndex(value: number, length: number) {
  if (length <= 0 || Number.isNaN(value)) return 0
  return Math.min(length - 1, Math.max(0, Math.round(value)))
}

type CustomSliderProps = {
  orientation: 'horizontal' | 'vertical'
  value: number
  maxIndex: number
  onChange: (value: number) => void
  disabled?: boolean
  className?: string
  onDragStart?: () => void
  onDragEnd?: () => void
  markers?: SliderMarker[]
}

function CustomSlider({
  orientation,
  value,
  maxIndex,
  onChange,
  disabled,
  className,
  onDragStart,
  onDragEnd,
  markers,
}: CustomSliderProps) {
  const trackRef = useRef<HTMLDivElement | null>(null)
  const [dragging, setDragging] = useState(false)
  const pointerIdRef = useRef<number | null>(null)
  const ratio = maxIndex > 0 ? value / maxIndex : 0
  const handlePosition =
    orientation === 'horizontal'
      ? { left: `${ratio * 100}%`, top: '50%', transform: 'translate(-50%, -50%)' }
      : { top: `${(1 - ratio) * 100}%`, left: '50%', transform: 'translate(-50%, -50%)' }
  const handleSize =
    orientation === 'horizontal'
      ? { width: '6px', height: '100%' }
      : { width: '100%', height: '6px' }

  const updateFromPointer = useCallback(
    (clientX: number, clientY: number) => {
      if (!trackRef.current) return
      const rect = trackRef.current.getBoundingClientRect()
      let pct = 0
      if (orientation === 'horizontal') {
        pct = (clientX - rect.left) / rect.width
      } else {
        pct = (rect.bottom - clientY) / rect.height
      }
      pct = Math.min(1, Math.max(0, pct))
      const nextValue = maxIndex <= 0 ? 0 : Math.round(pct * maxIndex)
      onChange(nextValue)
    },
    [maxIndex, onChange, orientation]
  )

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (disabled) return
    event.preventDefault()
    trackRef.current?.focus()
    pointerIdRef.current = event.pointerId
    trackRef.current?.setPointerCapture(event.pointerId)
    setDragging(true)
    onDragStart?.()
    updateFromPointer(event.clientX, event.clientY)
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    updateFromPointer(event.clientX, event.clientY)
  }

  const cleanupPointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging) return
    if (pointerIdRef.current !== null) {
      trackRef.current?.releasePointerCapture(pointerIdRef.current)
      pointerIdRef.current = null
    }
    setDragging(false)
    onDragEnd?.()
  }

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (disabled) return
    let delta = 0
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') delta = 1
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') delta = -1
    if (event.key === 'Home') delta = -Infinity
    if (event.key === 'End') delta = Infinity
    if (event.key === 'PageUp') delta = 5
    if (event.key === 'PageDown') delta = -5
    if (delta === 0) return
    event.preventDefault()
    const next = (() => {
      if (delta === Infinity) return maxIndex
      if (delta === -Infinity) return 0
      return clampIndex(value + delta, maxIndex + 1)
    })()
    onChange(next)
  }

  const thicknessClasses = orientation === 'horizontal' ? 'h-4 w-full' : 'w-5 h-full'
  const cursorStyle = disabled ? 'cursor-not-allowed' : 'cursor-pointer'
  const clampRatio = (value: number) => Math.min(1, Math.max(0, value))
  const markerElements = markers?.map((marker, idx) => {
    if (!Number.isFinite(marker.index)) return null
    const ratioValue = maxIndex > 0 ? clampRatio(marker.index / maxIndex) : 0
    const positionStyle =
      orientation === 'horizontal'
        ? {
            left: `${ratioValue * 100}%`,
            top: 0,
            transform: 'translate(-50%, 0)',
            width: '6px',
            height: '100%',
            borderRadius: '999px',
          }
        : {
            top: `${(1 - ratioValue) * 100}%`,
            left: 0,
            transform: 'translate(0, -50%)',
            height: '3px',
            width: '100%',
          }
    return (
      <div
        key={`marker-${idx}`}
        className="absolute"
        style={{
          ...positionStyle,
          backgroundColor: marker.color,
          opacity: disabled ? 0.4 : 0.85,
          pointerEvents: 'none',
        }}
      />
    )
  })

  return (
    <div
      ref={trackRef}
      role="slider"
      aria-valuemin={0}
      aria-valuemax={maxIndex}
      aria-valuenow={value}
      aria-orientation={orientation}
      aria-disabled={disabled}
      tabIndex={disabled ? -1 : 0}
      className={clsx('relative select-none bg-white/10 focus:outline-none', thicknessClasses, cursorStyle, className)}
      style={{
        border: '1px solid rgba(255,255,255,0.2)',
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={cleanupPointer}
      onPointerLeave={(event) => {
        if (dragging) cleanupPointer(event)
      }}
      onKeyDown={handleKeyDown}
    >
      {markerElements}
      <div
        className="absolute"
        style={{
          ...handlePosition,
          ...handleSize,
          backgroundColor: disabled ? '#4d5560' : COLORS.accent,
          transition: dragging ? 'none' : 'transform 120ms ease, left 120ms ease, top 120ms ease',
        }}
      />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Labels
// ─────────────────────────────────────────────────────────────────────────────

type AxisLabelBlockProps = {
  title: string
  detail: React.ReactNode
  valueLabel?: string
  align?: 'left' | 'right' | 'center'
}

function AxisLabelBlock({ title, detail, valueLabel, align = 'left' }: AxisLabelBlockProps) {
  const alignment =
    align === 'right'
      ? 'justify-end text-right'
      : align === 'center'
        ? 'justify-center text-center'
        : 'justify-start text-left'
  return (
    <div className={clsx('flex flex-wrap items-baseline gap-6 text-sm', alignment)} style={{ color: COLORS.textMuted }}>
      <span className="text-base font-semibold" style={{ color: COLORS.textPrimary }}>
        {title}
      </span>
      <span className="font-serif text-lg italic text-white/90">{detail}</span>
      {valueLabel && (
        <span className="text-lg font-semibold" style={{ color: COLORS.textPrimary }}>
          {valueLabel}
        </span>
      )}
    </div>
  )
}

function RadiusInlineLabel({ value, physicalLabel }: { value?: number; physicalLabel?: string | null }) {
  const label = typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '—'
  return (
    <div className="text-base font-semibold" style={{ color: COLORS.textPrimary }}>
      <span>{label}</span>
      {physicalLabel ? (
        <span className="ml-3 text-sm font-normal" style={{ color: COLORS.textMuted }}>
          ({physicalLabel})
        </span>
      ) : null}
      {'  '}
      {' '}
      <span style={{ color: COLORS.textMuted }}>Radius</span>
      {' '}
      <span className="font-serif italic text-white/90">r/a</span>
    </div>
  )
}

function MathSymbol({ symbol, subscript }: { symbol: string; subscript?: string }) {
  return (
    <span className="font-serif italic">
      {symbol}
      {subscript && (
        <sub className="ml-0.5 text-[0.65em] text-white/80" style={{ fontFamily: 'inherit' }}>
          {subscript}
        </sub>
      )}
    </span>
  )
}

export { Legend, AxisSliderHorizontal, AxisSliderVertical, AxisLabelBlock, RadiusInlineLabel, MathSymbol }
