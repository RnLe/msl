'use client'

import { useCallback, useEffect, useMemo, useRef, useState, useTransition } from 'react'
import type {
  ChangeEvent as ReactChangeEvent,
  KeyboardEvent as ReactKeyboardEvent,
} from 'react'
import clsx from 'clsx'
import { ChevronRight, RotateCcw } from 'lucide-react'
import { HeatmapPanel, MathSymbol, type MaterialAccent } from './HeatmapPanel'
import { SelectionPreview, buildDemoPreview, type BandPreviewPayload } from './SelectionPreview'
import type { CoverageMode, CoveragePoint, CoverageResponse, LatticeType } from './types'
import { COLORS, STATUS_LABELS } from './palette'
import { DEFAULT_EPS_BG_AXIS, DEFAULT_R_OVER_A_AXIS } from './axes'
import { DEMO_HEATMAP_STATUS } from './demoHopeData'
import { useBandCoverageStore, LATTICE_CONSTANT_PREFIXES, type LatticeConstantPrefix } from './store'
import { MATERIAL_LIBRARY, type MaterialLibraryEntry } from './library'
import { getMaterialSpectralProfile } from './materialSpectra'
import { MaterialLibraryModal } from './MaterialLibraryModal'
import { computeLatticeForFrequency, getMaterialAccent, lightenHexColor } from './materialLibraryShared'
import { useCoverageFetcher, useBandPreviewFetcher, computePreviewMax } from './useBandDataFetcher'

const EGG_TRIGGER_CLICKS = 5

type BandCoverageViewerProps = {
  scanId?: string
  apiBase?: string
  height?: number
}

const DEFAULT_SCAN_ID = 'square_hex_eps_r_v1'
const DEFAULT_API_BASE = 'https://data-server-railway-production.up.railway.app'

export function BandCoverageViewer({
  scanId = DEFAULT_SCAN_ID,
  apiBase = process.env.NEXT_PUBLIC_BAND_API_BASE ?? DEFAULT_API_BASE,
  height,
}: BandCoverageViewerProps) {
  const demoCoverage = useMemo(() => buildDemoCoverage(), [])
  const demoPreview = useMemo(() => buildDemoPreview(), [])
  const [activeLattice, setActiveLattice] = useState<LatticeType>('square')
  const [reloadToken, setReloadToken] = useState(0)
  const [eggMode, setEggMode] = useState(false)
  const [eggClickDisplay, setEggClickDisplay] = useState(0)
  const [heatmapStatusMessage, setHeatmapStatusMessage] = useState<string | null>(null)
  const heatmapRef = useRef<HTMLDivElement | null>(null)
  const eggClickCountRef = useRef(0)
  const eggTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const heatmapStatusTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [heatmapHeight, setHeatmapHeight] = useState<number | null>(null)
  const [hoveredLatticeButton, setHoveredLatticeButton] = useState<LatticeType | null>(null)
  const [heatmapTransitioning, setHeatmapTransitioning] = useState(false)
  const heatmapTransitionTimeout = useRef<NodeJS.Timeout | null>(null)
  const [isLatticeSwitchPending, startLatticeTransition] = useTransition()
  const [isPrefixMenuOpen, setPrefixMenuOpen] = useState(false)
  const [isMaterialLibraryOpen, setMaterialLibraryOpen] = useState(false)
  const prefixButtonRef = useRef<HTMLButtonElement | null>(null)
  const prefixMenuRef = useRef<HTMLDivElement | null>(null)
  const latticeConstantInputRef = useRef<HTMLInputElement | null>(null)
  const hovered = useBandCoverageStore((store) => store.hovered)
  const selected = useBandCoverageStore((store) => store.selected)
  const setHovered = useBandCoverageStore((store) => store.setHovered)
  const setSelected = useBandCoverageStore((store) => store.setSelected)
  const activeSlider = useBandCoverageStore((store) => store.activeAxis)
  const setActiveSlider = useBandCoverageStore((store) => store.setActiveAxis)
  const latticeConstantValue = useBandCoverageStore((store) => store.latticeConstantValue)
  const latticeConstantPrefix = useBandCoverageStore((store) => store.latticeConstantPrefix)
  const setLatticeConstantValue = useBandCoverageStore((store) => store.setLatticeConstantValue)
  const setLatticeConstantPrefix = useBandCoverageStore((store) => store.setLatticeConstantPrefix)
  const resetLatticeConstant = useBandCoverageStore((store) => store.resetLatticeConstant)
  const selectedMaterialId = useBandCoverageStore((store) => store.selectedMaterialId)
  const setSelectedMaterialId = useBandCoverageStore((store) => store.setSelectedMaterialId)

  // Coverage data fetching
  const state = useCoverageFetcher({
    scanId,
    apiBase,
    demoCoverage,
    reloadToken,
  })

  const handleHover = useCallback(
    (point?: CoveragePoint | null) => {
      setHovered(point ?? null)
    },
    [setHovered]
  )
  const handleSelectPoint = useCallback(
    (point: CoveragePoint) => {
      setSelected(point)
    },
    [setSelected]
  )

  const triggerHeatmapTransition = useCallback(() => {
    setHeatmapTransitioning(true)
    if (heatmapTransitionTimeout.current) {
      clearTimeout(heatmapTransitionTimeout.current)
    }
    heatmapTransitionTimeout.current = setTimeout(() => {
      setHeatmapTransitioning(false)
    }, 400)
  }, [])

  useEffect(() => {
    return () => {
      if (heatmapTransitionTimeout.current) {
        clearTimeout(heatmapTransitionTimeout.current)
      }
      if (eggTimeoutRef.current) {
        clearTimeout(eggTimeoutRef.current)
      }
      if (heatmapStatusTimeoutRef.current) {
        clearTimeout(heatmapStatusTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!isPrefixMenuOpen) return
    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node
      if (
        prefixMenuRef.current &&
        !prefixMenuRef.current.contains(target) &&
        prefixButtonRef.current &&
        !prefixButtonRef.current.contains(target)
      ) {
        setPrefixMenuOpen(false)
      }
    }
    document.addEventListener('pointerdown', handlePointerDown)
    return () => document.removeEventListener('pointerdown', handlePointerDown)
  }, [isPrefixMenuOpen])



  const normalizedCoverage = useMemo(() => state.coverage ?? demoCoverage, [state.coverage, demoCoverage])
  const coverage = eggMode ? demoCoverage : normalizedCoverage
  const effectiveMode: CoverageMode = eggMode ? 'offline' : state.mode
  const lattices: LatticeType[] = coverage.lattices?.length ? coverage.lattices : ['square', 'hex']
  const matrix = coverage.status?.[activeLattice]
  const hasData = Boolean(matrix && matrix.length)
  const lastUpdated = coverage.updatedAt ? new Date(coverage.updatedAt) : undefined
  const lastUpdatedLabel = lastUpdated ? formatUtcTimestamp(lastUpdated) : undefined
  const isDemo = effectiveMode === 'offline'
  const actualServerReady = Boolean(apiBase) && state.mode === 'ready'
  const serverOnline = !eggMode && actualServerReady
  const statusColor = serverOnline ? '#4ade80' : '#f87171'
  const statusLabel = serverOnline ? 'External Data Server online' : 'External Data Server offline'
  const eggClicksRemaining = Math.max(0, EGG_TRIGGER_CLICKS - eggClickDisplay)
  const apiReady = !eggMode && state.mode === 'ready' && Boolean(apiBase)
  const statusClickable = serverOnline || eggMode || (Boolean(apiBase) && !serverOnline)
  const statusCursor = serverOnline ? 'default' : statusClickable ? 'pointer' : 'default'
  const selectedPoint = selected && selected.lattice === activeLattice ? selected : null
  const hoveredPoint = hovered && hovered.lattice === activeLattice ? hovered : null
  const defaultEpsIndex = coverage.epsBg.length ? Math.floor((coverage.epsBg.length - 1) / 2) : 0
  const defaultRIndex = coverage.rOverA.length ? Math.floor((coverage.rOverA.length - 1) / 2) : 0
  const sliderEpsIndex = selectedPoint?.epsIndex ?? defaultEpsIndex
  const sliderRIndex = selectedPoint?.rIndex ?? defaultRIndex

  // Band preview data fetching
  const { bandState, beginSliderInteraction, endSliderInteraction } = useBandPreviewFetcher({
    scanId,
    apiBase,
    coverageMode: effectiveMode,
    coverage,
    activeLattice,
    sliderEpsIndex,
    sliderRIndex,
  })
  const currentRadius = coverage.rOverA[sliderRIndex]
  const currentEpsBg = coverage.epsBg[sliderEpsIndex]
  const sliderHeightStyle = heatmapHeight ? { height: `${heatmapHeight}px` } : undefined
  const legendLabels = isDemo ? DEMO_LEGEND_LABELS : STATUS_LABELS

  const showHeatmapStatus = useCallback((message: string, duration = 1200) => {
    setHeatmapStatusMessage(message)
    if (heatmapStatusTimeoutRef.current) {
      clearTimeout(heatmapStatusTimeoutRef.current)
    }
    heatmapStatusTimeoutRef.current = setTimeout(() => {
      setHeatmapStatusMessage(null)
    }, duration)
  }, [])

  const resetEggClicks = useCallback(() => {
    if (eggTimeoutRef.current) {
      clearTimeout(eggTimeoutRef.current)
      eggTimeoutRef.current = null
    }
    eggClickCountRef.current = 0
    setEggClickDisplay(0)
  }, [])

  const handleStatusClick = useCallback(() => {
    if (serverOnline) {
      const nextCount = eggClickCountRef.current + 1
      eggClickCountRef.current = nextCount
      setEggClickDisplay(nextCount)
      if (eggTimeoutRef.current) {
        clearTimeout(eggTimeoutRef.current)
      }
      eggTimeoutRef.current = setTimeout(() => {
        resetEggClicks()
      }, 1500)

      if (nextCount >= EGG_TRIGGER_CLICKS) {
        resetEggClicks()
        triggerHeatmapTransition()
        showHeatmapStatus('Updating demo heatmap…')
        setEggMode(true)
      }
      return
    }

    if (eggMode) {
      resetEggClicks()
      triggerHeatmapTransition()
      showHeatmapStatus('Reconnecting to Railway…', 1500)
      setEggMode(false)
      setReloadToken((token) => token + 1)
      return
    }

    if (!apiBase) return
    resetEggClicks()
    showHeatmapStatus('Requesting live coverage…')
    setReloadToken((token) => token + 1)
  }, [apiBase, eggMode, resetEggClicks, serverOnline, triggerHeatmapTransition, showHeatmapStatus])

  const handleStatusKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLButtonElement>) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        handleStatusClick()
      }
    },
    [handleStatusClick]
  )

  const statusTooltipDetail = (() => {
    if (serverOnline) {
      if (eggClickDisplay <= 0) return ''
      if (eggClicksRemaining <= 1) {
        return 'Secret heatmap unlocks on the next click.'
      }
      return `Secret heatmap unlocks in ${eggClicksRemaining} more click${eggClicksRemaining === 1 ? '' : 's'}.`
    }
    if (eggMode) {
      return 'Click to reconnect to Railway.'
    }
    if (!apiBase) {
      return 'Backend URL missing. Set NEXT_PUBLIC_BAND_API_BASE to enable live data.'
    }
    if (!apiReady) {
      return 'Connection lost. Click to retry the Railway API.'
    }
    return 'Railway API unavailable. Click to request the latest scan again.'
  })()



  useEffect(() => {
    if (typeof window === 'undefined') return
    const node = heatmapRef.current
    if (!node || !('ResizeObserver' in window)) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setHeatmapHeight(entry.contentRect.height)
      }
    })
    observer.observe(node)
    setHeatmapHeight(node.getBoundingClientRect().height)
    return () => {
      observer.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!lattices.includes(activeLattice) && lattices.length) {
      setActiveLattice(lattices[0])
    }
  }, [lattices, activeLattice])

  useEffect(() => {
    if (hovered && hovered.lattice !== activeLattice) setHovered(null)
    if (selected && selected.lattice !== activeLattice) {
      const point = buildPointFromIndices(coverage, activeLattice, selected.epsIndex, selected.rIndex)
      setSelected(point)
    }
  }, [activeLattice, hovered, selected, setHovered, setSelected, coverage])

  useEffect(() => {
    return () => {
      setHovered(null)
      setSelected(null)
      setActiveSlider(null)
    }
  }, [setHovered, setSelected, setActiveSlider])

  const ensureSelectionDefaults = useCallback(() => {
    if (!coverage.epsBg.length || !coverage.rOverA.length) return
    const current = useBandCoverageStore.getState().selected
    const matrix = coverage.status?.[activeLattice]
    const epsCount = coverage.epsBg.length
    const rCount = coverage.rOverA.length

    const withinBounds =
      current &&
      current.lattice === activeLattice &&
      current.epsIndex >= 0 &&
      current.epsIndex < epsCount &&
      current.rIndex >= 0 &&
      current.rIndex < rCount &&
      Number.isFinite(current.epsBg) &&
      Number.isFinite(current.rOverA)

    if (withinBounds) {
      const statusCode = matrix?.[current.epsIndex]?.[current.rIndex] ?? 0
      if (current.statusCode !== statusCode) {
        setSelected({ ...current, statusCode })
      }
      return
    }

    const fallback =
      findFirstAvailablePoint(coverage, activeLattice) ??
      buildPointFromIndices(coverage, activeLattice, 0, 0)
    if (!current || current.lattice !== fallback.lattice || current.epsIndex !== fallback.epsIndex || current.rIndex !== fallback.rIndex) {
      setSelected(fallback)
      return
    }
    if (current.statusCode !== fallback.statusCode) {
      setSelected(fallback)
    }
  }, [coverage, activeLattice, setSelected])

  useEffect(() => {
    ensureSelectionDefaults()
  }, [ensureSelectionDefaults])

  const updateSelectionFromAxes = useCallback(
    (next: { epsIndex?: number; rIndex?: number }) => {
      if (!coverage.epsBg.length || !coverage.rOverA.length) return
      const current = useBandCoverageStore.getState().selected
      const baseSelection =
        current && current.lattice === activeLattice
          ? current
          : buildPointFromIndices(coverage, activeLattice, defaultEpsIndex, defaultRIndex)
      const point = buildPointFromIndices(
        coverage,
        activeLattice,
        typeof next.epsIndex === 'number' ? next.epsIndex : baseSelection.epsIndex,
        typeof next.rIndex === 'number' ? next.rIndex : baseSelection.rIndex
      )
      if (
        current &&
        current.lattice === point.lattice &&
        current.epsIndex === point.epsIndex &&
        current.rIndex === point.rIndex
      ) {
        return
      }
      setSelected(point)
    },
    [coverage, activeLattice, defaultEpsIndex, defaultRIndex, setSelected]
  )

  const selectedMaterialEntry = useMemo(() => {
    if (!selectedMaterialId) return null
    return MATERIAL_LIBRARY.find((entry) => entry.id === selectedMaterialId) ?? null
  }, [selectedMaterialId])

  const selectedMaterialAccent = useMemo(() => {
    if (!selectedMaterialEntry) return null
    return getMaterialAccent(selectedMaterialEntry.id)
  }, [selectedMaterialEntry])

  const selectedMaterialProfile = useMemo(
    () => (selectedMaterialId ? getMaterialSpectralProfile(selectedMaterialId) : null),
    [selectedMaterialId]
  )

  const materialAxisIndex = useMemo(() => {
    if (!selectedMaterialEntry || !coverage.epsBg.length) return null
    let bestIndex = -1
    let minDelta = Infinity
    coverage.epsBg.forEach((value, idx) => {
      const delta = Math.abs(value - selectedMaterialEntry.epsilon)
      if (delta < minDelta) {
        minDelta = delta
        bestIndex = idx
      }
    })
    return bestIndex >= 0 ? bestIndex : null
  }, [coverage.epsBg, selectedMaterialEntry])

  useEffect(() => {
    if (materialAxisIndex === null || !selectedMaterialEntry) return
    const current = useBandCoverageStore.getState().selected
    if (current && current.lattice === activeLattice && current.epsIndex === materialAxisIndex) {
      return
    }
    updateSelectionFromAxes({ epsIndex: materialAxisIndex })
  }, [activeLattice, materialAxisIndex, selectedMaterialEntry, updateSelectionFromAxes])

  const materialHighlight = useMemo(() => {
    if (!selectedMaterialProfile?.windows?.length || !selectedMaterialAccent) return null
    return {
      color: selectedMaterialAccent.background,
      frequencyWindows: selectedMaterialProfile.windows.map((window) => window.frequencyHz),
    }
  }, [selectedMaterialProfile, selectedMaterialAccent])

  const scalingPreview = useMemo(() => {
    if (bandState.data && bandState.data.series?.length) return bandState.data
    return demoPreview
  }, [bandState.data, demoPreview])

  const normalizedAxisMax = useMemo(() => computePreviewMax(scalingPreview), [scalingPreview])

  const handleMaterialApply = useCallback(
    (entry: MaterialLibraryEntry) => {
      if (!coverage.epsBg.length) return
      let bestIndex = 0
      let minDelta = Infinity
      coverage.epsBg.forEach((value, idx) => {
        const delta = Math.abs(value - entry.epsilon)
        if (delta < minDelta) {
          minDelta = delta
          bestIndex = idx
        }
      })
      updateSelectionFromAxes({ epsIndex: bestIndex })
    },
    [coverage.epsBg, updateSelectionFromAxes]
  )

  const handleLibrarySelect = useCallback(
    (entry: MaterialLibraryEntry) => {
      setSelectedMaterialId(entry.id)
      handleMaterialApply(entry)
      setMaterialLibraryOpen(false)
    },
    [handleMaterialApply, setMaterialLibraryOpen, setSelectedMaterialId]
  )

  const handleClearMaterialSelection = useCallback(() => {
    setSelectedMaterialId(null)
  }, [setSelectedMaterialId])

  const handleMaterialWindowAlign = useCallback(
    (entry: MaterialLibraryEntry, targetFrequencyHz: number) => {
      const target = computeLatticeForFrequency(normalizedAxisMax, targetFrequencyHz)
      if (!target) return
      setLatticeConstantPrefix(target.prefix)
      setLatticeConstantValue(target.value)
      handleMaterialApply(entry)
      setSelectedMaterialId(entry.id)
      setMaterialLibraryOpen(false)
    },
    [handleMaterialApply, normalizedAxisMax, setLatticeConstantPrefix, setLatticeConstantValue, setSelectedMaterialId, setMaterialLibraryOpen]
  )

  const handleLatticeToggle = useCallback(
    (lattice: LatticeType) => {
      if (lattice === activeLattice) return
      triggerHeatmapTransition()
      const nextPoint = buildPointFromIndices(coverage, lattice, sliderEpsIndex, sliderRIndex)
      setSelected(nextPoint)
      startLatticeTransition(() => {
        setActiveLattice(lattice)
      })
    },
    [activeLattice, coverage, sliderEpsIndex, sliderRIndex, triggerHeatmapTransition, setSelected, startLatticeTransition]
  )

  const handleLatticeConstantValueChange = useCallback(
    (event: ReactChangeEvent<HTMLInputElement>) => {
      const raw = event.target.value
      if (raw === '') {
        setLatticeConstantValue(null)
        return
      }
      const parsed = Number.parseInt(raw, 10)
      if (Number.isNaN(parsed)) {
        return
      }
      const clamped = Math.min(999, Math.max(1, parsed))
      setLatticeConstantValue(clamped)
    },
    [setLatticeConstantValue]
  )

  const handlePrefixSelect = useCallback(
    (next: LatticeConstantPrefix | null) => {
      setLatticeConstantPrefix(next)
      setPrefixMenuOpen(false)
    },
    [setLatticeConstantPrefix, setPrefixMenuOpen]
  )

  const handleLatticeConstantReset = useCallback(() => {
    resetLatticeConstant()
    setPrefixMenuOpen(false)
  }, [resetLatticeConstant, setPrefixMenuOpen])

  const handleLatticeConstantWheel = useCallback(
    (event: WheelEvent) => {
      event.preventDefault()
      event.stopPropagation()
      const direction = event.deltaY < 0 ? 1 : -1
      const baseValue = latticeConstantValue ?? 1
      const nextValue = Math.min(999, Math.max(1, baseValue + direction))
      setLatticeConstantValue(nextValue)
    },
    [latticeConstantValue, setLatticeConstantValue]
  )

  useEffect(() => {
    const input = latticeConstantInputRef.current
    if (!input) return
    const handleWheel = (event: WheelEvent) => {
      handleLatticeConstantWheel(event)
    }
    input.addEventListener('wheel', handleWheel, { passive: false })
    return () => {
      input.removeEventListener('wheel', handleWheel)
    }
  }, [handleLatticeConstantWheel])

  const latticeConstantInputValue = latticeConstantValue !== null ? String(latticeConstantValue) : ''
  const latticeConstantPrefixValue = latticeConstantPrefix ?? ''
  const showLatticeConstantReset = latticeConstantValue !== null || latticeConstantPrefix !== null
  const prefixLabel = latticeConstantPrefixValue || 'prefix'

  return (
    <section
      className="space-y-6 p-6"
      style={{
        backgroundColor: COLORS.background,
        color: COLORS.textPrimary,
      }}
    >
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.35em]" style={{ color: COLORS.textMuted }}>
            Band library
          </p>
        </div>
        <div className="flex items-center justify-end">
          <button
            type="button"
            className="relative group"
            role="status"
            aria-label={statusLabel}
            onClick={statusClickable ? handleStatusClick : undefined}
            onKeyDown={statusClickable ? handleStatusKeyDown : undefined}
            style={{ cursor: statusCursor }}
          >
            <div className="relative h-4 w-4">
              <span
                className="absolute inset-0 rounded-full opacity-80 animate-ping"
                style={{ backgroundColor: statusColor, animationDuration: '2s' }}
              />
              <span
                className="absolute inset-0 rounded-full border border-black/20 shadow-lg"
                style={{ backgroundColor: statusColor }}
              />
            </div>
            <div
              className="pointer-events-none absolute right-0 top-7 z-10 w-64 rounded-lg border border-white/10 bg-[#0d1117] px-4 py-3 text-xs opacity-0 shadow-2xl transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100"
              style={{ color: COLORS.textPrimary }}
            >
              <p className="text-sm font-semibold" style={{ color: COLORS.textPrimary }}>
                External Data Server
              </p>
              <p
                className="mt-2 text-sm font-semibold"
                style={{ color: serverOnline ? '#4ade80' : '#f87171', letterSpacing: '0.2em' }}
              >
                {serverOnline ? 'ONLINE' : 'OFFLINE'}
              </p>
              {statusTooltipDetail ? (
                <p className="mt-1 text-[11px] leading-relaxed" style={{ color: COLORS.textMuted }}>
                  {statusTooltipDetail}
                </p>
              ) : null}
            </div>
          </button>
        </div>
      </header>

      <div className="flex flex-col gap-4">
        <HeatmapPanel
          coverage={coverage}
          activeLattice={activeLattice}
          height={height}
          hasData={hasData}
          mode={state.mode}
          transitioning={heatmapTransitioning || isLatticeSwitchPending}
          hoveredPoint={hoveredPoint}
          selectedPoint={selectedPoint}
          onHover={handleHover}
          onSelect={handleSelectPoint}
          sliderEpsIndex={sliderEpsIndex}
          sliderRIndex={sliderRIndex}
          currentEpsBg={currentEpsBg}
          currentRadius={currentRadius}
          latticeConstantValue={latticeConstantValue}
          latticeConstantPrefix={latticeConstantPrefix}
          heatmapStatusMessage={heatmapStatusMessage}
          materialAxisIndex={materialAxisIndex}
          selectedMaterialAccent={selectedMaterialAccent}
          selectedMaterialEntry={selectedMaterialEntry}
          legendLabels={legendLabels}
          onUpdateSelection={updateSelectionFromAxes}
          onSliderDragStart={beginSliderInteraction}
          onSliderDragEnd={endSliderInteraction}
          onClearMaterialSelection={handleClearMaterialSelection}
          onOpenMaterialLibrary={() => setMaterialLibraryOpen(true)}
          heatmapRef={heatmapRef}
          sliderHeightStyle={sliderHeightStyle}
        />

        <div className="flex flex-wrap items-center justify-between gap-3 py-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-[0.35em]" style={{ color: COLORS.textMuted }}>
              Lattice
            </span>
            <div className="flex flex-wrap gap-1">
              {lattices.map((lattice) => (
                <button
                  key={lattice}
                  type="button"
                  onClick={() => handleLatticeToggle(lattice)}
                  onMouseEnter={() => setHoveredLatticeButton(lattice)}
                  onMouseLeave={() => setHoveredLatticeButton((prev) => (prev === lattice ? null : prev))}
                  className={clsx('px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] transition-all cursor-pointer', {
                    'shadow-[0_6px_18px_rgba(0,0,0,0.4)]': lattice === activeLattice,
                  })}
                  style={{
                    backgroundColor: (() => {
                      const isActive = lattice === activeLattice
                      const isHovered = hoveredLatticeButton === lattice
                      const base = '#111111'
                      if (isActive) {
                        return lightenHexColor(base, 0.18)
                      }
                      if (isHovered) {
                        return lightenHexColor(base, 0.1)
                      }
                      return 'transparent'
                    })(),
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: 999,
                  }}
                >
                  {formatLatticeLabel(lattice)}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-white/80 text-xs">
            {showLatticeConstantReset ? (
              <button
                type="button"
                onClick={handleLatticeConstantReset}
                className="p-1 text-[#bfbfbf] transition hover:text-white"
                aria-label="Reset lattice constant"
                title="Reset lattice constant"
              >
                <RotateCcw className="h-3 w-3" strokeWidth={1.5} />
              </button>
            ) : null}
            <span className="font-semibold uppercase tracking-[0.35em]" style={{ color: COLORS.textMuted }}>
              Lattice Constant
            </span>
            <div className="flex items-center gap-1 text-sm">
              <span className="font-semibold" style={{ color: COLORS.textPrimary }}>
                <MathSymbol symbol="a" /> =
              </span>
              <input
                type="number"
                min={1}
                max={999}
                step={1}
                inputMode="numeric"
                pattern="[0-9]*"
                ref={latticeConstantInputRef}
                value={latticeConstantInputValue}
                onChange={handleLatticeConstantValueChange}
                className="h-6 w-14 rounded-none border-0 bg-transparent px-1 text-sm text-white placeholder:text-white/30 focus:outline-none"
                aria-label="Set lattice constant value"
              />
              <div className="relative">
                <button
                  ref={prefixButtonRef}
                  type="button"
                  onClick={() => setPrefixMenuOpen((open) => !open)}
                  className="flex h-6 min-w-[72px] items-center justify-between px-2 text-left text-sm text-white"
                  style={{ backgroundColor: '#111111', border: 'none', borderRadius: 9999 }}
                  aria-haspopup="listbox"
                  aria-expanded={isPrefixMenuOpen}
                  aria-label="Select lattice constant prefix"
                >
                  <span>{prefixLabel}</span>
                  <ChevronRight className={clsx('h-3 w-3 transition-transform', { 'rotate-90': isPrefixMenuOpen })} />
                </button>
                {isPrefixMenuOpen ? (
                  <div
                    ref={prefixMenuRef}
                    className="absolute right-0 top-full z-50 mt-1 w-32 text-sm text-white shadow-2xl"
                    style={{ backgroundColor: '#111111', border: 'none', zIndex: 60 }}
                    role="listbox"
                  >
                    <button
                      type="button"
                      onClick={() => handlePrefixSelect(null)}
                      className={clsx('flex w-full items-center px-3 py-2 text-left transition hover:bg-[#1b1b1b]', {
                        'bg-[#1b1b1b]': latticeConstantPrefixValue === '',
                      })}
                      role="option"
                      aria-selected={latticeConstantPrefixValue === ''}
                    >
                      prefix
                    </button>
                    {LATTICE_CONSTANT_PREFIXES.map((prefix) => (
                      <button
                        key={prefix}
                        type="button"
                        onClick={() => handlePrefixSelect(prefix)}
                        className={clsx('flex w-full items-center px-3 py-2 text-left transition hover:bg-[#1b1b1b]', {
                          'bg-[#1b1b1b]': prefix === latticeConstantPrefixValue,
                        })}
                        role="option"
                        aria-selected={prefix === latticeConstantPrefixValue}
                      >
                        {prefix}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

      </div>

      <SelectionPreview point={selectedPoint} mode={state.mode} demo={demoPreview} bandState={bandState} materialHighlight={materialHighlight} />
      <MaterialLibraryModal
        open={isMaterialLibraryOpen}
        onClose={() => setMaterialLibraryOpen(false)}
        entries={MATERIAL_LIBRARY}
        activeId={selectedMaterialId}
        onSelect={handleLibrarySelect}
        onAlignMaterial={handleMaterialWindowAlign}
        axisMaxNormalized={normalizedAxisMax}
      />
    </section>
  )
}

function formatLatticeLabel(lattice: LatticeType) {
  if (lattice === 'hex') return 'Triangular'
  return lattice
}

function clampIndex(value: number, length: number) {
  if (length <= 0 || Number.isNaN(value)) return 0
  return Math.min(length - 1, Math.max(0, Math.round(value)))
}

function FractionLabel({ numerator, denominator }: { numerator: string; denominator: string }) {
  return (
    <span className="font-serif italic">
      {numerator}
      <span className="mx-1">/</span>
      {denominator}
    </span>
  )
}

function formatUtcTimestamp(date: Date) {
  return new Intl.DateTimeFormat('en-GB', {
    timeZone: 'UTC',
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(date)
}

function normalizeCoverage(payload: CoverageResponse, scanId: string): CoverageResponse {
  const epsBg = payload.epsBg?.length ? payload.epsBg : payload.axes?.epsBg ?? DEFAULT_EPS_BG_AXIS
  const rOverA = payload.rOverA?.length ? payload.rOverA : payload.axes?.rOverA ?? DEFAULT_R_OVER_A_AXIS
  const status = payload.status ?? {}
  const lattices = payload.lattices?.length ? payload.lattices : Object.keys(status)
  return {
    scanId: payload.scanId ?? scanId,
    lattices: lattices.length ? lattices : ['square', 'hex'],
    epsBg,
    rOverA,
    status,
    updatedAt: payload.updatedAt,
  }
}

const DEMO_UPDATED_AT = '2024-01-01T00:00:00.000Z'
const DEMO_LEGEND_LABELS: Record<number, string> = {
  0: 'Freedom',
  1: 'Peace',
  2: 'Respect',
  3: 'Growth',
}

const DEMO_COVERAGE = buildDemoDataset()

function buildDemoCoverage(): CoverageResponse {
  return DEMO_COVERAGE
}

function densifyAxis(axis: number[], factor: number) {
  if (!Array.isArray(axis) || axis.length === 0 || factor <= 1) {
    return axis
  }

  const result: number[] = []
  for (let i = 0; i < axis.length - 1; i += 1) {
    const start = axis[i]
    const end = axis[i + 1]
    for (let step = 0; step < factor; step += 1) {
      const t = step / factor
      result.push(start + (end - start) * t)
    }
  }
  result.push(axis[axis.length - 1])
  return result
}

function matchAxisToLength(axis: number[], targetLength: number, precision: number) {
  if (!Array.isArray(axis) || axis.length === 0 || targetLength <= 0) {
    return axis
  }

  if (axis.length === targetLength) {
    return axis
  }

  if (axis.length > 1) {
    const rawFactor = (targetLength - 1) / (axis.length - 1)
    const nearest = Math.round(rawFactor)
    if (nearest > 1 && Math.abs(rawFactor - nearest) < 1e-6) {
      return densifyAxis(axis, nearest)
    }
  }

  if (targetLength === 1) {
    return [axis[0]]
  }

  const lastIndex = axis.length - 1
  return Array.from({ length: targetLength }, (_, idx) => {
    const t = idx / (targetLength - 1)
    const scaled = t * lastIndex
    const lower = Math.floor(scaled)
    const upper = Math.min(lastIndex, lower + 1)
    const weight = scaled - lower
    const value = axis[lower] + (axis[upper] - axis[lower]) * weight
    return Number(value.toFixed(precision))
  })
}

function normalizeMatrix<T>(matrix: T[][], expectedRows: number, expectedCols: number, fallback: T): T[][] {
  const rows: T[][] = []
  for (let row = 0; row < expectedRows; row += 1) {
    const sourceRow = matrix[row] ?? []
    const normalizedRow: T[] = []
    for (let col = 0; col < expectedCols; col += 1) {
      const value = sourceRow[col]
      normalizedRow.push(value ?? fallback)
    }
    rows.push(normalizedRow)
  }
  return rows
}

function buildDemoDataset(): CoverageResponse {
  const epsTarget = DEMO_HEATMAP_STATUS.length || DEFAULT_EPS_BG_AXIS.length
  const rTarget = DEMO_HEATMAP_STATUS[0]?.length ?? DEFAULT_R_OVER_A_AXIS.length
  const epsBg = matchAxisToLength(DEFAULT_EPS_BG_AXIS, epsTarget, 2)
  const rOverA = matchAxisToLength(DEFAULT_R_OVER_A_AXIS, rTarget, 3)
  const statusMatrix = normalizeMatrix(DEMO_HEATMAP_STATUS, epsBg.length, rOverA.length, 0)
  return {
    scanId: 'demo_offline',
    lattices: ['square', 'hex'],
    epsBg,
    rOverA,
    status: {
      square: statusMatrix,
      hex: statusMatrix.map((row) => row.slice()),
    },
    updatedAt: DEMO_UPDATED_AT,
  }
}

function buildPointFromIndices(coverage: CoverageResponse, lattice: LatticeType, epsIndex: number, rIndex: number): CoveragePoint {
  const safeEps = clampIndex(epsIndex, coverage.epsBg.length)
  const safeR = clampIndex(rIndex, coverage.rOverA.length)
  return {
    lattice,
    epsIndex: safeEps,
    rIndex: safeR,
    epsBg: coverage.epsBg[safeEps] ?? NaN,
    rOverA: coverage.rOverA[safeR] ?? NaN,
    statusCode: coverage.status?.[lattice]?.[safeEps]?.[safeR] ?? 0,
  }
}

function findFirstAvailablePoint(coverage: CoverageResponse, lattice: LatticeType): CoveragePoint | null {
  const matrix = coverage.status?.[lattice]
  if (!matrix?.length) return null
  const rows = Math.min(matrix.length, coverage.epsBg.length)
  const cols = coverage.rOverA.length
  for (let eps = 0; eps < rows; eps += 1) {
    const row = matrix[eps] ?? []
    const limit = Math.min(row.length, cols)
    for (let r = 0; r < limit; r += 1) {
      if ((row[r] ?? 0) > 0) {
        return buildPointFromIndices(coverage, lattice, eps, r)
      }
    }
  }
  return null
}
