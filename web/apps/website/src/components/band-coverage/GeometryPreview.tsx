'use client'

import { useMemo } from 'react'
import type { LatticeType } from './types'
import { COLORS } from './palette'
import { useBandCoverageStore } from './store'
import {
  formatLengthWithBestPrefix,
  formatLengthWithPrefix,
  resolveLatticeConstantMeters,
} from './units'

const GEOMETRY_WIDTH = 260
const GEOMETRY_HEIGHT = 220
const PRIMARY_BG = '#111111'

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

const EMPTY_LINE_GEOMETRY: GeometryLineMetadata = {
  centerPoint: null,
  rightNeighbor: null,
  neighborDistance: null,
  bottomLeft: null,
  bottomRight: null,
  bottomRatio: null,
  bottomBaseline: null,
}

export type GeometryPreviewProps = {
  lattice: LatticeType
  rOverA: number
}

export function GeometryPreview({ lattice, rOverA }: GeometryPreviewProps) {
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

function buildBottomSpan(
  left: GeometryCircle | null,
  right: GeometryCircle | null
): [GeometryCircle, GeometryCircle] | null {
  if (!left || !right) return null
  return [left, right]
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

function computeRadius(cellSize: number, ratio: number) {
  const normalized = clamp01((ratio - 0.1) / (0.48 - 0.1))
  return Math.max(3, cellSize * (0.35 + 0.55 * normalized))
}

function clamp01(value: number) {
  if (Number.isNaN(value)) return 0
  return Math.min(1, Math.max(0, value))
}
