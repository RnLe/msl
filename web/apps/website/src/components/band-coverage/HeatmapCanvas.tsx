'use client'

import { Layer, Rect, Stage } from 'react-konva'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import type { CoverageMode, CoveragePoint, LatticeType } from './types'
import { useElementSize } from './use-element-size'
import { COLORS, STATUS_COLORS } from './palette'

type HeatmapCanvasProps = {
  data?: number[][]
  epsBg: number[]
  rOverA: number[]
  lattice: LatticeType
  height?: number
  disabled?: boolean
  state: CoverageMode
  transitioning?: boolean
  hovered?: CoveragePoint | null
  selected?: CoveragePoint | null
  onHover?: (point?: CoveragePoint) => void
  onSelect?: (point: CoveragePoint) => void
  materialColumnIndex?: number
  materialColor?: string
}

export function HeatmapCanvas({
  data,
  epsBg,
  rOverA,
  lattice,
  height,
  disabled = false,
  state,
  transitioning = false,
  hovered,
  selected,
  onHover,
  onSelect,
  materialColumnIndex,
  materialColor,
}: HeatmapCanvasProps) {
  const [containerRef, size] = useElementSize<HTMLDivElement>()
  const measuredWidth = size.width
  const stageWidth = Math.max(0, measuredWidth)
  const baseHeight = height ?? Math.round((stageWidth || 640) * 0.55)
  const desiredHeight = height ? baseHeight : Math.round(baseHeight * (2 / 3))
  const stageHeight = Math.max(220, desiredHeight)

  const cells = useMemo(() => {
    if (!data || !stageWidth || !stageHeight) return []
    const width = stageWidth
    const heightPx = stageHeight
    const epsCount = Math.max(1, epsBg.length)
    const rCount = Math.max(1, rOverA.length)
    const cellWidth = width / epsCount
    const cellHeight = heightPx / rCount

    const next = [] as Array<{
      key: string
      x: number
      y: number
      width: number
      height: number
      color: string
      epsIndex: number
      rIndex: number
      statusCode: number
    }>

    for (let epsIndex = 0; epsIndex < epsCount; epsIndex += 1) {
      const column = data[epsIndex] ?? []
      for (let rIndex = 0; rIndex < rCount; rIndex += 1) {
        const statusCode = column[rIndex] ?? 0
        const color = STATUS_COLORS[statusCode] ?? STATUS_COLORS[0]
        const x = epsIndex * cellWidth
        const y = heightPx - (rIndex + 1) * cellHeight
        next.push({
          key: `${epsIndex}-${rIndex}`,
          x,
          y,
          width: cellWidth + 0.5,
          height: cellHeight + 0.5,
          color,
          epsIndex,
          rIndex,
          statusCode,
        })
      }
    }

    return next
  }, [data, stageHeight, stageWidth, epsBg.length, rOverA.length])

  const overlayMessage = disabled
    ? state === 'loading'
      ? 'Loading coverage…'
      : 'No coverage data available yet.'
    : undefined

  const showCanvas = Boolean(data && data.length && stageWidth > 0)

  const getPoint = useCallback(
    (cell: { epsIndex: number; rIndex: number; statusCode: number }): CoveragePoint => ({
      lattice,
      epsIndex: cell.epsIndex,
      rIndex: cell.rIndex,
      epsBg: epsBg[cell.epsIndex] ?? NaN,
      rOverA: rOverA[cell.rIndex] ?? NaN,
      statusCode: cell.statusCode,
    }),
    [epsBg, rOverA, lattice]
  )

  const hoveredKey = hovered && hovered.lattice === lattice ? `${hovered.epsIndex}-${hovered.rIndex}` : undefined
  const selectedKey = selected && selected.lattice === lattice ? `${selected.epsIndex}-${selected.rIndex}` : undefined

  const cellLookup = useMemo(() => {
    const map = new Map<string, (typeof cells)[number]>()
    for (const cell of cells) {
      map.set(cell.key, cell)
    }
    return map
  }, [cells])

  const materialColumnCells = useMemo(() => {
    if (!Number.isInteger(materialColumnIndex) || materialColumnIndex === undefined || materialColumnIndex === null) return []
    return cells.filter((cell) => cell.epsIndex === materialColumnIndex)
  }, [cells, materialColumnIndex])

  const hoverFrameRef = useRef<number | null>(null)
  const pendingHoverRef = useRef<CoveragePoint | undefined>(undefined)
  const lastHoverKeyRef = useRef<string | null>(null)

  const flushHover = useCallback(() => {
    hoverFrameRef.current = null
    const next = pendingHoverRef.current
    pendingHoverRef.current = undefined
    const nextKey = next ? `${next.lattice}-${next.epsIndex}-${next.rIndex}` : null
    if (nextKey === lastHoverKeyRef.current) return
    lastHoverKeyRef.current = nextKey
    onHover?.(next)
  }, [onHover])

  const scheduleHover = useCallback(
    (point?: CoveragePoint) => {
      pendingHoverRef.current = point
      if (hoverFrameRef.current !== null) return
      hoverFrameRef.current = requestAnimationFrame(() => flushHover())
    },
    [flushHover]
  )

  useEffect(() => {
    return () => {
      if (hoverFrameRef.current !== null) {
        cancelAnimationFrame(hoverFrameRef.current)
      }
    }
  }, [])

  const handleSelect = useCallback(
    (point: CoveragePoint) => {
      onSelect?.(point)
    },
    [onSelect]
  )

  const baseLayer = useMemo(() => {
    if (!showCanvas) return null
    return (
      <Layer>
        {cells.map((cell) => (
          <Rect
            key={cell.key}
            x={cell.x}
            y={cell.y}
            width={cell.width}
            height={cell.height}
            fill={cell.color}
            opacity={disabled ? 0.25 : 0.95}
            listening={!disabled}
            onPointerMove={() => !disabled && scheduleHover(getPoint(cell))}
            onPointerLeave={() => !disabled && scheduleHover(undefined)}
            onPointerDown={() => !disabled && handleSelect(getPoint(cell))}
          />
        ))}
      </Layer>
    )
  }, [cells, disabled, getPoint, handleSelect, scheduleHover, showCanvas])

  const hoveredCell = hoveredKey ? cellLookup.get(hoveredKey) : undefined
  const selectedCell = selectedKey ? cellLookup.get(selectedKey) : undefined

  const highlightLayer = useMemo(() => {
    if (!showCanvas) return null
    return (
      <Layer listening={false}>
        {selectedCell && (
          <Rect x={selectedCell.x} y={selectedCell.y} width={selectedCell.width} height={selectedCell.height} stroke="#ffffff" strokeWidth={1.5} />
        )}
        {hoveredCell && (!selectedCell || hoveredCell.key !== selectedCell.key) && (
          <Rect x={hoveredCell.x} y={hoveredCell.y} width={hoveredCell.width} height={hoveredCell.height} stroke="#ffffff" strokeWidth={1} />
        )}
      </Layer>
    )
  }, [hoveredCell, selectedCell, showCanvas])

  const materialOverlayLayer = useMemo(() => {
    if (!showCanvas || !materialColumnCells.length || !materialColor) return null
    const columnX = materialColumnCells[0]?.x ?? 0
    const columnWidth = materialColumnCells[0]?.width ?? 0
    const borderColor = '#ffffff'
    const borderWidth = Math.max(2, Math.min(5, columnWidth * 0.25))
    return (
      <Layer listening={false}>
        {materialColumnCells.map((cell) => (
          <Rect key={`material-${cell.key}`} x={cell.x} y={cell.y} width={cell.width} height={cell.height} fill={materialColor} opacity={disabled ? 0.45 : 1} />
        ))}
        <Rect
          x={columnX}
          y={0}
          width={columnWidth}
          height={stageHeight}
          stroke={borderColor}
          strokeWidth={borderWidth}
          opacity={0.95}
        />
      </Layer>
    )
  }, [disabled, materialColor, materialColumnCells, showCanvas, stageHeight])

  const tooltipData = !disabled && hoveredCell && hovered
    ? {
        left: hoveredCell.x + hoveredCell.width / 2,
        top: hoveredCell.y + 6,
        rValue: hovered.rOverA,
        epsValue: hovered.epsBg,
      }
    : null

  return (
    <div
      ref={containerRef}
      className="relative w-full"
      style={{
        overflow: 'visible',
        minHeight: `${stageHeight}px`,
        backgroundColor: COLORS.card,
        border: `1px solid ${COLORS.border}`,
      }}
      onPointerLeave={() => !disabled && scheduleHover(undefined)}
    >
      {showCanvas ? (
        <Stage width={stageWidth} height={stageHeight} listening={!disabled} className="bg-transparent" style={{ backgroundColor: COLORS.card }}>
          {baseLayer}
          {materialOverlayLayer}
          {highlightLayer}
        </Stage>
      ) : (
        <div className="flex h-full min-h-[220px] w-full items-center justify-center text-sm" style={{ color: COLORS.textMuted }}>
          Preparing canvas…
        </div>
      )}

      {tooltipData && (
        <div
          className="pointer-events-none absolute rounded-md border px-2 py-1 text-[11px]"
          style={{
            left: tooltipData.left,
            top: tooltipData.top,
            transform: 'translate(-50%, -110%)',
            backgroundColor: 'rgba(8,9,12,0.92)',
            borderColor: COLORS.border,
            color: COLORS.textPrimary,
            whiteSpace: 'nowrap',
            zIndex: 20,
            boxShadow: '0 8px 24px rgba(0,0,0,0.45)',
          }}
        >
          <div>
            <span className="font-serif italic">r/a</span>
            <span className="ml-1">= {Number.isFinite(tooltipData.rValue) ? tooltipData.rValue.toFixed(3) : '—'}</span>
          </div>
          <div>
            &epsilon;<sub>bg</sub>
            <span className="ml-1">= {Number.isFinite(tooltipData.epsValue) ? tooltipData.epsValue.toFixed(2) : '—'}</span>
          </div>
        </div>
      )}

      {transitioning && !disabled && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/30 backdrop-blur-[1px] text-[11px] uppercase tracking-[0.4em]" style={{ color: COLORS.textPrimary }}>
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-current border-t-transparent" />
          <span>Updating</span>
        </div>
      )}

      {overlayMessage && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm" style={{ color: COLORS.textMuted }}>
          <div
            className="rounded-full border border-dashed px-4 py-1.5 shadow-lg"
            style={{
              borderColor: COLORS.border,
              backgroundColor: '#111111d0',
              color: COLORS.textPrimary,
              boxShadow: '0 15px 45px rgba(0,0,0,0.45)',
            }}
          >
            {overlayMessage}
          </div>
        </div>
      )}
    </div>
  )
}
