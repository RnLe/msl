'use client'

import { memo, useCallback, useEffect, useMemo, useRef, useState, useTransition } from 'react'
import type {
  CSSProperties,
  ChangeEvent as ReactChangeEvent,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from 'react'
import clsx from 'clsx'
import { ChevronRight, RotateCcw, Library, X } from 'lucide-react'
import { HeatmapCanvas } from './HeatmapCanvas'
import { SelectionPreview, buildDemoPreview, type BandPreviewPayload } from './SelectionPreview'
import type { CoverageMode, CoveragePoint, CoverageResponse, LatticeType } from './types'
import { COLORS, STATUS_COLORS, STATUS_LABELS } from './palette'
import { DEFAULT_EPS_BG_AXIS, DEFAULT_R_OVER_A_AXIS } from './axes'
import { DEMO_HEATMAP_STATUS } from './demoHopeData'
import { useBandCoverageStore, LATTICE_CONSTANT_PREFIXES, type LatticeConstantPrefix } from './store'
import { formatPhysicalRadius } from './units'
import { MATERIAL_LIBRARY, type MaterialCategory, type MaterialLibraryEntry } from './library'

const EGG_TRIGGER_CLICKS = 5
const EPS_MATCH_TOLERANCE = 0.45

const MATERIAL_CATEGORY_ORDER: MaterialCategory[] = ['reference', 'polymer', 'intermediate', 'semiconductor', 'matrix']
const MATERIAL_CATEGORY_LABELS: Record<MaterialCategory, string> = {
  reference: 'Reference / Low Index',
  polymer: 'Polymers',
  intermediate: 'Intermediate Dielectrics',
  semiconductor: 'High-Index Semiconductors',
  matrix: 'Matrix Pairs',
}
const MATERIAL_STICKER_SIZE = 88
const MATERIAL_CARD_PADDING_X = 16
const MATERIAL_CARD_PADDING_Y = 12
const MATERIAL_ACCENTS: Record<string, { background: string; text: string }> = {
  air: { background: '#F6F3EC', text: '#201c13' },
  water: { background: '#6FAFDB', text: '#062238' },
  silica: { background: '#C7D1DB', text: '#101a24' },
  fluoride: { background: '#E3D9FF', text: '#2a1a4a' },
  polymer: { background: '#F6B27A', text: '#3a1d07' },
  ptfe: { background: '#E4E1C4', text: '#2c2711' },
  su8: { background: '#D76B73', text: '#ffffff' },
  alumina: { background: '#B79AC8', text: '#240f33' },
  si3n4: { background: '#A8C686', text: '#1e2610' },
  aln: { background: '#8EB7B5', text: '#082626' },
  gan: { background: '#4B90A6', text: '#ffffff' },
  tio2: { background: '#E0B45A', text: '#2d1a05' },
  chalcogenide: { background: '#7E4F7B', text: '#ffffff' },
  si: { background: '#335C4C', text: '#ffffff' },
  gaas: { background: '#75485E', text: '#ffffff' },
  inp: { background: '#C47D3E', text: '#2c1400' },
  gap: { background: '#CC9F3C', text: '#2f1600' },
  ge: { background: '#494F5C', text: '#ffffff' },
}
const DEFAULT_MATERIAL_ACCENT = { background: '#dedede', text: '#111111' }

function getMaterialAccent(id: string) {
  return MATERIAL_ACCENTS[id] ?? DEFAULT_MATERIAL_ACCENT
}

type BandCoverageViewerProps = {
  scanId?: string
  apiBase?: string
  height?: number
}

type CoverageState = {
  mode: CoverageMode
  coverage?: CoverageResponse
  message?: string
}

type BandState = {
  status: 'idle' | 'loading' | 'ready' | 'error' | 'prefetching'
  data?: BandPreviewPayload
  message?: string
}

type CoverageCacheEntry = Omit<CoverageState, 'coverage'> & { coverage: CoverageResponse }

const COVERAGE_CACHE = new Map<string, CoverageCacheEntry>()
const BAND_PREVIEW_CACHE = new Map<string, BandPreviewPayload>()

type BandEndpointResponse = {
  params: {
    lattice: LatticeType
    epsBg: number
    rOverA: number
  }
  kPath: number[]
  bandsTE?: number[][]
  bandsTM?: number[][]
}

type BandAxisSliceEntry = {
  params?: {
    scanId?: string
    lattice?: LatticeType
    epsBg?: number
    rOverA?: number
  }
  epsIndex?: number
  rIndex?: number
  bandsTE?: number[][]
  bandsTM?: number[][]
}

type BandAxisSliceResponse = {
  lattice: LatticeType
  fixedAxis: 'epsBg' | 'rOverA'
  fixedIndex: number
  fixedValue: number
  kPath: number[]
  entries: BandAxisSliceEntry[]
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
  const [state, setState] = useState<CoverageState>({ mode: 'loading' })
  const [activeLattice, setActiveLattice] = useState<LatticeType>('square')
  const [bandState, setBandState] = useState<BandState>({ status: 'idle' })
  const [reloadToken, setReloadToken] = useState(0)
  const [eggMode, setEggMode] = useState(false)
  const [eggClickDisplay, setEggClickDisplay] = useState(0)
  const [heatmapStatusMessage, setHeatmapStatusMessage] = useState<string | null>(null)
  const heatmapRef = useRef<HTMLDivElement | null>(null)
  const axisPendingKeysRef = useRef<Set<string>>(new Set())
  const axisPrefetchStatusRef = useRef<Map<string, 'loading' | 'ready'>>(new Map())
  const axisPrefetchControllers = useRef<Map<string, AbortController>>(new Map())
  const isMountedRef = useRef(true)
  const singleFetchControllerRef = useRef<AbortController | null>(null)
  const activeSliderRef = useRef<'row' | 'column' | null>(null)
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
  const coverageCacheKey = `${apiBase ?? 'demo'}::${scanId}`
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
      isMountedRef.current = false
      singleFetchControllerRef.current?.abort()
      axisPrefetchControllers.current.forEach((controller) => controller.abort())
      axisPrefetchControllers.current.clear()
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

  useEffect(() => {
    if (!apiBase) {
      const offlineEntry: CoverageCacheEntry = {
        mode: 'offline',
        coverage: demoCoverage,
        message: 'Backend URL not configured (NEXT_PUBLIC_BAND_API_BASE). Showing demo coverage.',
      }
      COVERAGE_CACHE.set(coverageCacheKey, offlineEntry)
      setState(offlineEntry)
      return
    }

    const cached = COVERAGE_CACHE.get(coverageCacheKey)
    if (cached) {
      setState(cached)
      return
    }

    let cancelled = false
    const controller = new AbortController()

    setState((prev) => ({ mode: 'loading', coverage: prev.coverage }))

    async function load() {
      try {
        const response = await fetch(`${apiBase}/scans/${scanId}/coverage`, { signal: controller.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const payload = (await response.json()) as CoverageResponse
        if (cancelled) return
        const normalized = normalizeCoverage(payload, scanId)
        const entry: CoverageCacheEntry = { mode: 'ready', coverage: normalized }
        COVERAGE_CACHE.set(coverageCacheKey, entry)
        setState(entry)
      } catch (error) {
        if (controller.signal.aborted || cancelled) return
        console.warn('[BandCoverageViewer] falling back to demo coverage', error)
        const offlineEntry: CoverageCacheEntry = {
          mode: 'offline',
          coverage: demoCoverage,
          message: 'Could not reach the Railway API. Displaying demo data while offline.',
        }
        COVERAGE_CACHE.set(coverageCacheKey, offlineEntry)
        setState(offlineEntry)
      }
    }

    load()

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [apiBase, coverageCacheKey, demoCoverage, scanId, reloadToken])

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
        setBandState({ status: 'idle' })
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
  }, [apiBase, eggMode, resetEggClicks, serverOnline, setBandState, triggerHeatmapTransition, showHeatmapStatus])

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


  const makeSelectionKey = useCallback(
    (lattice: LatticeType, epsIndex: number, rIndex: number) => `${scanId}:${lattice}:${epsIndex}:${rIndex}`,
    [scanId]
  )

  const makeAxisKey = useCallback(
    (type: 'column' | 'row', lattice: LatticeType, fixedIndex: number) => `${scanId}:${type}:${lattice}:${fixedIndex}`,
    [scanId]
  )

  const syncSelectionWithCache = useCallback(() => {
    if (!isMountedRef.current) return
    const current = useBandCoverageStore.getState().selected
    if (!current) return
    const cacheKey = makeSelectionKey(current.lattice, current.epsIndex, current.rIndex)
    const cached = BAND_PREVIEW_CACHE.get(cacheKey)
    if (cached) {
      setBandState({ status: 'ready', data: cached })
    }
  }, [makeSelectionKey])

  const startSingleFetch = useCallback(
    (point: CoveragePoint) => {
      if (!apiBase || state.mode !== 'ready') return
      if (!point || point.statusCode === 0) {
        setBandState({ status: 'idle', message: 'Band data not available for this geometry yet.' })
        return
      }
      const selectionKey = makeSelectionKey(point.lattice, point.epsIndex, point.rIndex)
      singleFetchControllerRef.current?.abort()
      const controller = new AbortController()
      singleFetchControllerRef.current = controller
      setBandState((prev) => ({ ...prev, status: 'loading' }))

      const query = new URLSearchParams({
        lattice: String(point.lattice),
        epsBg: String(point.epsBg),
        rOverA: String(point.rOverA),
        epsIndex: String(point.epsIndex),
        rIndex: String(point.rIndex),
        include_te: 'true',
        include_tm: 'true',
      })

      ;(async () => {
        try {
          const response = await fetch(`${apiBase}/scans/${scanId}/band?${query.toString()}`, { signal: controller.signal })
          if (!response.ok) throw new Error(`HTTP ${response.status}`)
          const payload = (await response.json()) as BandEndpointResponse
          if (controller.signal.aborted || !isMountedRef.current) return
          const preview = toBandPreview(payload)
          BAND_PREVIEW_CACHE.set(selectionKey, preview)
          setBandState({ status: 'ready', data: preview })
        } catch (error) {
          if (controller.signal.aborted || !isMountedRef.current) return
          console.warn('[BandCoverageViewer] band endpoint unavailable yet', error)
          setBandState({ status: 'error', message: 'Band endpoint is not available yet. Check the Railway deployment once ready.' })
        }
      })()
    },
    [apiBase, state.mode, makeSelectionKey, scanId]
  )

  const requestSelectionIfUncached = useCallback(() => {
    if (state.mode !== 'ready') return
    const current = useBandCoverageStore.getState().selected
    if (!current) return
    const cacheKey = makeSelectionKey(current.lattice, current.epsIndex, current.rIndex)
    const cached = BAND_PREVIEW_CACHE.get(cacheKey)
    if (cached) {
      setBandState({ status: 'ready', data: cached })
      return
    }
    startSingleFetch(current)
  }, [state.mode, makeSelectionKey, startSingleFetch])

  const ingestAxisSlice = useCallback(
    (payload: BandAxisSliceResponse) => {
      if (!payload?.entries?.length) return
      const { lattice, kPath } = payload
      payload.entries.forEach((entry) => {
        const epsValue = typeof entry.epsIndex === 'number' ? coverage.epsBg[entry.epsIndex] : entry.params?.epsBg
        const rValue = typeof entry.rIndex === 'number' ? coverage.rOverA[entry.rIndex] : entry.params?.rOverA
        const epsIndex =
          typeof entry.epsIndex === 'number'
            ? entry.epsIndex
            : findAxisIndex(coverage.epsBg, epsValue)
        const rIndex =
          typeof entry.rIndex === 'number'
            ? entry.rIndex
            : findAxisIndex(coverage.rOverA, rValue)
        if (epsIndex < 0 || rIndex < 0) return
        const preview = buildPreviewFromBands({
          lattice,
          epsBg: epsValue ?? coverage.epsBg[epsIndex] ?? NaN,
          rOverA: rValue ?? coverage.rOverA[rIndex] ?? NaN,
          kPath,
          bandsTE: entry.bandsTE,
          bandsTM: entry.bandsTM,
        })
        const cacheKey = makeSelectionKey(lattice, epsIndex, rIndex)
        BAND_PREVIEW_CACHE.set(cacheKey, preview)
        axisPendingKeysRef.current.delete(cacheKey)
      })
      syncSelectionWithCache()
    },
    [coverage.epsBg, coverage.rOverA, makeSelectionKey, syncSelectionWithCache]
  )

  const prefetchColumn = useCallback(
    (targetEpsIndex?: number) => {
      if (state.mode !== 'ready' || !apiBase) return
      if (!coverage.epsBg.length || !coverage.rOverA.length) return
      const epsIndex = Number.isFinite(targetEpsIndex) ? (targetEpsIndex as number) : sliderEpsIndex
      if (!Number.isFinite(epsIndex)) return
      const axisKey = makeAxisKey('column', activeLattice, epsIndex)
      const status = axisPrefetchStatusRef.current.get(axisKey)
      if (status === 'loading' || status === 'ready') return

    const pendingKeys = coverage.rOverA
      .map((_, rIndex) => makeSelectionKey(activeLattice, epsIndex, rIndex))
      .filter((key) => !BAND_PREVIEW_CACHE.has(key))
    if (!pendingKeys.length) {
      axisPrefetchStatusRef.current.set(axisKey, 'ready')
      syncSelectionWithCache()
      return
    }

    axisPrefetchStatusRef.current.set(axisKey, 'loading')
    pendingKeys.forEach((key) => axisPendingKeysRef.current.add(key))

    const controller = new AbortController()
    axisPrefetchControllers.current.set(axisKey, controller)

    const query = new URLSearchParams({
      lattice: String(activeLattice),
      epsIndex: String(epsIndex),
      include_te: 'true',
      include_tm: 'true',
    })

      ;(async () => {
      try {
        const response = await fetch(`${apiBase}/scans/${scanId}/band/column?${query.toString()}`, { signal: controller.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const payload = (await response.json()) as BandAxisSliceResponse
        if (!isMountedRef.current) return
        ingestAxisSlice(payload)
        axisPrefetchStatusRef.current.set(axisKey, 'ready')
      } catch (error) {
        if (!controller.signal.aborted) {
          console.warn('[BandCoverageViewer] column prefetch failed', error)
          axisPrefetchStatusRef.current.delete(axisKey)
          requestSelectionIfUncached()
        }
      } finally {
        pendingKeys.forEach((key) => axisPendingKeysRef.current.delete(key))
        axisPrefetchControllers.current.delete(axisKey)
        syncSelectionWithCache()
      }
      })()
    },
    [state.mode, apiBase, coverage, sliderEpsIndex, makeAxisKey, activeLattice, makeSelectionKey, scanId, ingestAxisSlice, syncSelectionWithCache, requestSelectionIfUncached]
  )

  const prefetchRow = useCallback(
    (targetRIndex?: number) => {
      if (state.mode !== 'ready' || !apiBase) return
      if (!coverage.epsBg.length || !coverage.rOverA.length) return
      const rIndex = Number.isFinite(targetRIndex) ? (targetRIndex as number) : sliderRIndex
      if (!Number.isFinite(rIndex)) return
      const axisKey = makeAxisKey('row', activeLattice, rIndex)
      const status = axisPrefetchStatusRef.current.get(axisKey)
      if (status === 'loading' || status === 'ready') return

    const pendingKeys = coverage.epsBg
      .map((_, epsIndex) => makeSelectionKey(activeLattice, epsIndex, rIndex))
      .filter((key) => !BAND_PREVIEW_CACHE.has(key))
    if (!pendingKeys.length) {
      axisPrefetchStatusRef.current.set(axisKey, 'ready')
      syncSelectionWithCache()
      return
    }

    axisPrefetchStatusRef.current.set(axisKey, 'loading')
    pendingKeys.forEach((key) => axisPendingKeysRef.current.add(key))

    const controller = new AbortController()
    axisPrefetchControllers.current.set(axisKey, controller)

    const query = new URLSearchParams({
      lattice: String(activeLattice),
      rIndex: String(rIndex),
      include_te: 'true',
      include_tm: 'true',
    })

      ;(async () => {
      try {
        const response = await fetch(`${apiBase}/scans/${scanId}/band/row?${query.toString()}`, { signal: controller.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const payload = (await response.json()) as BandAxisSliceResponse
        if (!isMountedRef.current) return
        ingestAxisSlice(payload)
        axisPrefetchStatusRef.current.set(axisKey, 'ready')
      } catch (error) {
        if (!controller.signal.aborted) {
          console.warn('[BandCoverageViewer] row prefetch failed', error)
          axisPrefetchStatusRef.current.delete(axisKey)
          requestSelectionIfUncached()
        }
      } finally {
        pendingKeys.forEach((key) => axisPendingKeysRef.current.delete(key))
        axisPrefetchControllers.current.delete(axisKey)
        syncSelectionWithCache()
      }
      })()
    },
    [state.mode, apiBase, coverage, sliderRIndex, makeAxisKey, activeLattice, makeSelectionKey, scanId, ingestAxisSlice, syncSelectionWithCache, requestSelectionIfUncached]
  )

  const beginSliderInteraction = useCallback(
    (type: 'row' | 'column') => {
      if (activeSliderRef.current === type) return
      activeSliderRef.current = type
      setActiveSlider(type)
      singleFetchControllerRef.current?.abort()
      setBandState((prev) => (prev.data ? { ...prev, status: 'prefetching' } : prev))
      if (type === 'column') {
        prefetchColumn()
      } else {
        prefetchRow()
      }
    },
    [prefetchColumn, prefetchRow, setActiveSlider]
  )

  const endSliderInteraction = useCallback(() => {
    if (!activeSliderRef.current) return
    activeSliderRef.current = null
    setActiveSlider(null)
    requestSelectionIfUncached()
  }, [requestSelectionIfUncached, setActiveSlider])

  useEffect(() => {
    if (!selectedPoint) return
    if (activeSlider === 'column') {
      prefetchColumn(selectedPoint.epsIndex)
    } else if (activeSlider === 'row') {
      prefetchRow(selectedPoint.rIndex)
    }
  }, [activeSlider, selectedPoint?.epsIndex, selectedPoint?.rIndex, prefetchColumn, prefetchRow])
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
      if (statusCode > 0) {
        return
      }
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

  useEffect(() => {
    if (!selected || !Number.isFinite(selected.epsBg) || !Number.isFinite(selected.rOverA)) {
      setBandState({ status: 'idle' })
      singleFetchControllerRef.current?.abort()
      return
    }

    if (activeSlider) {
      syncSelectionWithCache()
      return
    }

    if (state.mode !== 'ready' || !apiBase) {
      setBandState({ status: 'idle' })
      singleFetchControllerRef.current?.abort()
      return
    }

    const selectionKey = makeSelectionKey(selected.lattice, selected.epsIndex, selected.rIndex)
    const cached = BAND_PREVIEW_CACHE.get(selectionKey)
    if (cached) {
      setBandState({ status: 'ready', data: cached })
      return
    }

    if (axisPendingKeysRef.current.has(selectionKey)) {
      setBandState((prev) => ({ ...prev, status: 'prefetching' }))
      return
    }

    startSingleFetch(selected)
  }, [selected, state.mode, apiBase, makeSelectionKey, startSingleFetch, activeSlider, syncSelectionWithCache])

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

  const activeMaterialId = useMemo(() => {
    if (!Number.isFinite(currentEpsBg)) return null
    let closestEntry: MaterialLibraryEntry | null = null
    let minDelta = Infinity
    MATERIAL_LIBRARY.forEach((entry) => {
      const delta = Math.abs(entry.epsilon - (currentEpsBg as number))
      if (delta < minDelta) {
        minDelta = delta
        closestEntry = entry
      }
    })
    if (!closestEntry) return null
    return minDelta <= EPS_MATCH_TOLERANCE ? closestEntry.id : null
  }, [currentEpsBg])

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
      handleMaterialApply(entry)
      setMaterialLibraryOpen(false)
    },
    [handleMaterialApply, setMaterialLibraryOpen]
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
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-[0.35em]" style={{ color: COLORS.textMuted }}>
              Lattice
            </span>
            <div className="flex flex-wrap gap-2">
              {lattices.map((lattice) => (
                <button
                  key={lattice}
                  type="button"
                  onClick={() => handleLatticeToggle(lattice)}
                  onMouseEnter={() => setHoveredLatticeButton(lattice)}
                  onMouseLeave={() => setHoveredLatticeButton((prev) => (prev === lattice ? null : prev))}
                  className={clsx('px-4 py-1.5 text-sm font-semibold capitalize transition-all cursor-pointer', {
                    'shadow-[0_10px_35px_rgba(0,0,0,0.35)]': lattice === activeLattice,
                  })}
                  style={{
                    backgroundColor: (() => {
                      const isActive = lattice === activeLattice
                      const isHovered = hoveredLatticeButton === lattice
                      const base = '#111111'
                      if (isActive) {
                        return lightenHexColor(base, 0.2)
                      }
                      if (isHovered) {
                        return lightenHexColor(base, 0.12)
                      }
                      return 'transparent'
                    })(),
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: 0,
                  }}
                >
                  {formatLatticeLabel(lattice)}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-white/80">
            {showLatticeConstantReset ? (
              <button
                type="button"
                onClick={handleLatticeConstantReset}
                className="p-1 text-[#bfbfbf] transition hover:text-white"
                aria-label="Reset lattice constant"
                title="Reset lattice constant"
              >
                <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.5} />
              </button>
            ) : null}
            <span className="text-xs font-semibold uppercase tracking-[0.35em]" style={{ color: COLORS.textMuted }}>
              Lattice Constant
            </span>
            <div className="flex items-center gap-2 text-sm">
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
                className="h-7 w-16 rounded-none border-0 bg-transparent px-1 text-sm text-white placeholder:text-white/30 focus:outline-none"
                aria-label="Set lattice constant value"
              />
              <div className="relative">
                <button
                  ref={prefixButtonRef}
                  type="button"
                  onClick={() => setPrefixMenuOpen((open) => !open)}
                  className="flex h-7 min-w-[80px] items-center justify-between px-2 text-left text-sm text-white"
                  style={{ backgroundColor: '#111111', border: 'none', borderRadius: 0 }}
                  aria-haspopup="listbox"
                  aria-expanded={isPrefixMenuOpen}
                  aria-label="Select lattice constant prefix"
                >
                  <span>{prefixLabel}</span>
                  <ChevronRight className={clsx('h-3.5 w-3.5 transition-transform', { 'rotate-90': isPrefixMenuOpen })} />
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

        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <RadiusInlineLabel
              value={Number.isFinite(currentRadius) ? (currentRadius as number) : undefined}
              physicalLabel={formatPhysicalRadius(
                Number.isFinite(currentRadius) ? (currentRadius as number) : null,
                latticeConstantValue,
                latticeConstantPrefix
              )}
            />
            {/* legend labels switch to demo values when offline */}
            <Legend align="end" labels={legendLabels} />
          </div>

          <div className="space-y-4">
            <div className="flex flex-col gap-4 md:flex-row md:items-stretch">
              <div
                className="md:w-4 md:flex-shrink-0 md:self-stretch"
                style={sliderHeightStyle}
              >
                <AxisSliderVertical
                  axisLength={coverage.rOverA.length}
                  value={sliderRIndex}
                  disabled={!hasData}
                  onChange={(value) => updateSelectionFromAxes({ rIndex: value })}
                  onDragStart={() => beginSliderInteraction('column')}
                  onDragEnd={endSliderInteraction}
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
                    state={state.mode}
                    transitioning={heatmapTransitioning || isLatticeSwitchPending}
                    hovered={hoveredPoint}
                    selected={selectedPoint}
                    onHover={handleHover}
                    onSelect={handleSelectPoint}
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
                    onChange={(value) => updateSelectionFromAxes({ epsIndex: value })}
                    onDragStart={() => beginSliderInteraction('row')}
                    onDragEnd={endSliderInteraction}
                    className="w-full"
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
                          ? `${(currentEpsBg as number).toFixed(2)}`
                          : undefined
                      }
                      align="center"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => setMaterialLibraryOpen(true)}
                    className="rounded-full border border-white/10 bg-white/5 p-2 text-white transition hover:border-white/40 hover:bg-white/15"
                    aria-label="Open material library"
                    title="Open material library"
                  >
                    <Library className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>

      <SelectionPreview point={selectedPoint} mode={state.mode} demo={demoPreview} bandState={bandState} />
      <MaterialLibraryModal
        open={isMaterialLibraryOpen}
        onClose={() => setMaterialLibraryOpen(false)}
        entries={MATERIAL_LIBRARY}
        activeId={activeMaterialId}
        onSelect={handleLibrarySelect}
      />
    </section>
  )
}

function lightenHexColor(hex: string, factor: number) {
  if (!hex || typeof hex !== 'string' || !hex.startsWith('#')) return hex
  const normalized = hex.length === 4
    ? hex
        .slice(1)
        .split('')
        .map((char) => char + char)
        .join('')
    : hex.slice(1)

  if (normalized.length !== 6) return hex

  const num = parseInt(normalized, 16)
  if (Number.isNaN(num)) return hex

  const clamp = (value: number) => Math.min(255, Math.max(0, value))
  const adjust = (value: number) => clamp(Math.round(value + (255 - value) * factor))

  const r = adjust((num >> 16) & 0xff)
  const g = adjust((num >> 8) & 0xff)
  const b = adjust(num & 0xff)

  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`
}

type LegendProps = {
  align?: 'start' | 'end'
  labels?: Record<number, string>
}

type MaterialLibraryModalProps = {
  open: boolean
  onClose: () => void
  entries: MaterialLibraryEntry[]
  activeId: string | null
  onSelect: (entry: MaterialLibraryEntry) => void
}

function MaterialLibraryModal({ open, onClose, entries, activeId, onSelect }: MaterialLibraryModalProps) {
  useEffect(() => {
    if (!open) return undefined
    const originalOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = originalOverflow
    }
  }, [open])

  const grouped = useMemo(() => {
    const bucket = new Map<MaterialCategory, MaterialLibraryEntry[]>()
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
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative z-10 w-[92vw] max-w-5xl overflow-hidden border border-[#1c1c1c] bg-[#111111] text-white shadow-[0_40px_120px_rgba(0,0,0,0.65)]">
        <div className="flex items-center justify-between border-b border-[#1f1f1f] px-6 py-4" style={{ backgroundColor: '#131313' }}>
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.4em]" style={{ color: '#8a8a8a' }}>
              Material Library
            </div>
            <div className="text-lg font-semibold" style={{ color: '#ffffff' }}>
              Photonic Crystal Reference Set
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 text-white transition hover:text-white/70"
            aria-label="Close material library"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-6 py-5" style={{ backgroundColor: '#111111' }}>
          <div className="space-y-6">
            <p className="text-sm leading-relaxed" style={{ color: '#d3d3d3' }}>
              This library lists <strong>isotropic, non-metallic dielectrics</strong> for photonic crystals, all modeled with a simple real scalar (<em>ε<sub>r</sub></em>).
              Each card calls out a <strong>typical low-loss λ-window (µm)</strong> (sources below).
              The scalar ε<sub>r</sub> model stays valid anywhere the material is transparent (visible, telecom, or mid-IR alike).
              <span className="block mt-3">
                Click a material to apply its ε<sub>r</sub> to the heatmap.
              </span>
            </p>
            {orderedGroups.map((group) => (
              <div key={group.category} className="space-y-3">
                <div className="text-base font-semibold uppercase tracking-[0.35em]" style={{ color: '#e6e6e6' }}>
                  {MATERIAL_CATEGORY_LABELS[group.category]}
                </div>
                <div className="space-y-2">
                  {group.items.map((entry) => (
                    <button
                      key={entry.id}
                      type="button"
                      onClick={() => onSelect(entry)}
                      className={clsx(
                        'w-full overflow-hidden border text-left transition',
                        activeId === entry.id
                          ? 'border-[#3a3a3a] bg-[#1b1b1b] text-white'
                          : 'border-[#1f1f1f] bg-[#111111] text-white/80 hover:border-[#2c2c2c] hover:bg-[#181818] hover:text-white'
                      )}
                      >
                      {(() => {
                        const accent = getMaterialAccent(entry.id)
                        return (
                          <div className="flex items-stretch">
                            <div
                              className="flex shrink-0 items-center justify-center text-base font-semibold self-stretch"
                              style={{
                                height: '100%',
                                aspectRatio: '1 / 1',
                                minHeight: MATERIAL_STICKER_SIZE,
                                minWidth: MATERIAL_STICKER_SIZE,
                                backgroundColor: accent.background,
                                color: accent.text,
                                boxShadow: '0 6px 16px rgba(0,0,0,0.35)',
                              }}
                            >
                              {entry.label}
                            </div>
                            <div
                              className="flex-1 space-y-2"
                              style={{
                                padding: `${MATERIAL_CARD_PADDING_Y}px ${MATERIAL_CARD_PADDING_X}px`,
                                minHeight: MATERIAL_STICKER_SIZE,
                              }}
                            >
                              <div className="flex flex-wrap items-baseline justify-between gap-2">
                                <div className="text-lg font-semibold" style={{ color: '#ffffff' }}>
                                  {entry.fullName}
                                </div>
                                <div className="text-lg font-mono" style={{ color: '#f3f3f3' }}>
                                  ε ≈ {entry.epsilon.toFixed(2)}
                                  <span className="inline-block" aria-hidden="true" style={{ width: '48px' }} />
                                  n ≈ {entry.refractiveIndex.toFixed(2)}
                                </div>
                              </div>
                              <div className="text-sm" style={{ color: '#c2c2c2' }}>
                                {entry.summary}
                              </div>
                              {entry.aliases?.length ? (
                                <div className="text-[11px] uppercase tracking-[0.3em]" style={{ color: '#8f8f8f' }}>
                                  {entry.aliases.join(' / ')}
                                </div>
                              ) : null}
                              {entry.designWindow ? (
                                <div className="text-xs" style={{ color: '#a6a6a6' }}>
                                  <span className="font-semibold" style={{ letterSpacing: '0.08em' }}>
                                    λ window
                                  </span>
                                  <span className="ml-2 font-mono text-sm" style={{ color: '#f5f5f5' }}>
                                    {entry.designWindow}
                                  </span>
                                </div>
                              ) : null}
                            </div>
                          </div>
                        )
                      })()}
                    </button>
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
                    <a
                      href="https://refractiveindex.info"
                      className="underline"
                      style={{ color: '#1b82c8' }}
                      target="_blank"
                      rel="noreferrer"
                    >
                      RefractiveIndex.INFO
                    </a>
                    <span>Consolidated n, k datasets across UV–mid-IR for bulk materials.</span>
                  </div>
                </li>
                <li>
                  <div className="flex flex-wrap items-baseline gap-4">
                    <a
                      href="https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=6973"
                      className="underline"
                      style={{ color: '#1b82c8' }}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Thorlabs optical polymer substrates
                    </a>
                    <span>PMMA/plexiglass transmission data and THz PTFE references.</span>
                  </div>
                </li>
                <li>
                  <div className="flex flex-wrap items-baseline gap-4">
                    <a
                      href="https://www.crystran.com/optical-materials"
                      className="underline"
                      style={{ color: '#1b82c8' }}
                      target="_blank"
                      rel="noreferrer"
                    >
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
  )
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

function formatLatticeLabel(lattice: LatticeType) {
  if (lattice === 'hex') return 'Triangular'
  return lattice
}

type AxisSliderProps = {
  axisLength: number
  value: number
  onChange: (index: number) => void
  disabled?: boolean
  className?: string
  style?: CSSProperties
  onDragStart?: () => void
  onDragEnd?: () => void
}

const AxisSliderHorizontal = memo(function AxisSliderHorizontal({ axisLength, value, onChange, disabled, className, onDragStart, onDragEnd }: AxisSliderProps) {
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
    />
  )
})

const AxisSliderVertical = memo(function AxisSliderVertical({ axisLength, value, onChange, disabled, className, style, onDragStart, onDragEnd }: AxisSliderProps) {
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
}

function CustomSlider({ orientation, value, maxIndex, onChange, disabled, className, onDragStart, onDragEnd }: CustomSliderProps) {
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
      className={clsx('relative select-none bg-white/10 focus:outline-none focus-visible:ring-1 focus-visible:ring-white/40', thicknessClasses, cursorStyle, className)}
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
      {'  '}
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

function findAxisIndex(axis: number[], value?: number, tolerance = 1e-4) {
  if (!Array.isArray(axis) || !axis.length) return -1
  if (!Number.isFinite(value)) return -1
  let closest = -1
  let minDelta = Infinity
  for (let idx = 0; idx < axis.length; idx += 1) {
    const delta = Math.abs(axis[idx] - (value as number))
    if (delta < tolerance) return idx
    if (delta < minDelta) {
      minDelta = delta
      closest = idx
    }
  }
  return minDelta < tolerance * 10 ? closest : -1
}

type BuildPreviewArgs = {
  lattice: LatticeType
  epsBg: number
  rOverA: number
  kPath?: number[]
  bandsTE?: number[][]
  bandsTM?: number[][]
}

function buildPreviewFromBands({ lattice, epsBg, rOverA, kPath, bandsTE, bandsTM }: BuildPreviewArgs): BandPreviewPayload {
  const resolvedKPath = kPath?.length ? kPath : Array.from({ length: 60 }, (_, idx) => idx / 59)
  const series: BandPreviewPayload['series'] = []

  const addSeries = (bands: number[][] | undefined, label: 'TE' | 'TM', color: string) => {
    if (!bands || !bands.length) return
    const maxBands = Math.min(4, bands.length)
    for (let idx = 0; idx < maxBands; idx += 1) {
      const values = bands[idx] ?? []
      if (!values.length) continue
      series.push({
        id: `${label}-${idx + 1}`,
        label: `${label} band ${idx + 1}`,
        color,
        values,
      })
    }
  }

  addSeries(bandsTE, 'TE', '#c97a7a')
  addSeries(bandsTM, 'TM', '#7096b7')

  return {
    lattice,
    epsBg,
    rOverA,
    kPath: resolvedKPath,
    series,
  }
}

function toBandPreview(payload: BandEndpointResponse): BandPreviewPayload {
  return buildPreviewFromBands({
    lattice: payload.params.lattice,
    epsBg: payload.params.epsBg,
    rOverA: payload.params.rOverA,
    kPath: payload.kPath,
    bandsTE: payload.bandsTE,
    bandsTM: payload.bandsTM,
  })
}
