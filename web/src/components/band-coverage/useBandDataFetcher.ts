'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import type { BandPreviewPayload } from './SelectionPreview'
import type { CoverageMode, CoveragePoint, CoverageResponse, LatticeType } from './types'
import { useBandCoverageStore } from './store'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type BandState = {
  status: 'idle' | 'loading' | 'ready' | 'error' | 'prefetching'
  data?: BandPreviewPayload
  message?: string
}

export type CoverageState = {
  mode: CoverageMode
  coverage?: CoverageResponse
  message?: string
}

type CoverageCacheEntry = Omit<CoverageState, 'coverage'> & { coverage: CoverageResponse }

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

// ─────────────────────────────────────────────────────────────────────────────
// Caches (module-level singletons)
// ─────────────────────────────────────────────────────────────────────────────

const COVERAGE_CACHE = new Map<string, CoverageCacheEntry>()
const BAND_PREVIEW_CACHE = new Map<string, BandPreviewPayload>()

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

function findAxisIndex(axis: number[], value?: number, tolerance = 1e-4): number {
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

function computePreviewMax(preview?: BandPreviewPayload | null): number {
  if (!preview?.series?.length) return 1
  let maxValue = 0
  preview.series.forEach((band) => {
    band.values?.forEach((value) => {
      if (Number.isFinite(value) && (value as number) > maxValue) {
        maxValue = value as number
      }
    })
  })
  return maxValue > 0 ? maxValue : 1
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook: useCoverageFetcher
// ─────────────────────────────────────────────────────────────────────────────

export type UseCoverageFetcherOptions = {
  scanId: string
  apiBase: string | undefined
  demoCoverage: CoverageResponse
  reloadToken: number
}

export function useCoverageFetcher({
  scanId,
  apiBase,
  demoCoverage,
  reloadToken,
}: UseCoverageFetcherOptions): CoverageState {
  const coverageCacheKey = `${apiBase ?? 'demo'}::${scanId}`
  const [state, setState] = useState<CoverageState>({ mode: 'loading' })

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
        console.warn('[useCoverageFetcher] falling back to demo coverage', error)
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

  return state
}

function normalizeCoverage(payload: CoverageResponse, scanId: string): CoverageResponse {
  const epsBg = payload.epsBg?.length ? payload.epsBg : payload.axes?.epsBg ?? []
  const rOverA = payload.rOverA?.length ? payload.rOverA : payload.axes?.rOverA ?? []
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

// ─────────────────────────────────────────────────────────────────────────────
// Hook: useBandPreviewFetcher
// ─────────────────────────────────────────────────────────────────────────────

export type UseBandPreviewFetcherOptions = {
  scanId: string
  apiBase: string | undefined
  coverageMode: CoverageMode
  coverage: CoverageResponse
  activeLattice: LatticeType
  sliderEpsIndex: number
  sliderRIndex: number
}

export type UseBandPreviewFetcherResult = {
  bandState: BandState
  setBandState: React.Dispatch<React.SetStateAction<BandState>>
  beginSliderInteraction: (type: 'row' | 'column') => void
  endSliderInteraction: () => void
  prefetchColumn: (targetEpsIndex?: number) => void
  prefetchRow: (targetRIndex?: number) => void
}

export function useBandPreviewFetcher({
  scanId,
  apiBase,
  coverageMode,
  coverage,
  activeLattice,
  sliderEpsIndex,
  sliderRIndex,
}: UseBandPreviewFetcherOptions): UseBandPreviewFetcherResult {
  const [bandState, setBandState] = useState<BandState>({ status: 'idle' })

  const isMountedRef = useRef(true)
  const singleFetchControllerRef = useRef<AbortController | null>(null)
  const activeSliderRef = useRef<'row' | 'column' | null>(null)
  const axisPendingKeysRef = useRef<Set<string>>(new Set())
  const axisPrefetchStatusRef = useRef<Map<string, 'loading' | 'ready'>>(new Map())
  const axisPrefetchControllers = useRef<Map<string, AbortController>>(new Map())

  const activeSlider = useBandCoverageStore((store) => store.activeAxis)
  const setActiveSlider = useBandCoverageStore((store) => store.setActiveAxis)
  const selected = useBandCoverageStore((store) => store.selected)

  // CRITICAL: Must explicitly set isMountedRef.current = true on mount!
  // React Strict Mode unmounts/remounts components, and refs persist across this cycle.
  // Without this, isMountedRef stays false after remount, causing all fetches to silently fail.
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      singleFetchControllerRef.current?.abort()
      axisPrefetchControllers.current.forEach((controller) => controller.abort())
      axisPrefetchControllers.current.clear()
    }
  }, [])

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
      if (!apiBase || coverageMode !== 'ready') return
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
          console.warn('[useBandPreviewFetcher] band endpoint unavailable yet', error)
          setBandState({ status: 'error', message: 'Band endpoint is not available yet. Check the Railway deployment once ready.' })
        }
      })()
    },
    [apiBase, coverageMode, makeSelectionKey, scanId]
  )

  const requestSelectionIfUncached = useCallback(() => {
    if (coverageMode !== 'ready') return
    const current = useBandCoverageStore.getState().selected
    if (!current) return
    const cacheKey = makeSelectionKey(current.lattice, current.epsIndex, current.rIndex)
    const cached = BAND_PREVIEW_CACHE.get(cacheKey)
    if (cached) {
      setBandState({ status: 'ready', data: cached })
      return
    }
    startSingleFetch(current)
  }, [coverageMode, makeSelectionKey, startSingleFetch])

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
      if (coverageMode !== 'ready' || !apiBase) return
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
            console.warn('[useBandPreviewFetcher] column prefetch failed', error)
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
    [coverageMode, apiBase, coverage, sliderEpsIndex, makeAxisKey, activeLattice, makeSelectionKey, scanId, ingestAxisSlice, syncSelectionWithCache, requestSelectionIfUncached]
  )

  const prefetchRow = useCallback(
    (targetRIndex?: number) => {
      if (coverageMode !== 'ready' || !apiBase) return
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
            console.warn('[useBandPreviewFetcher] row prefetch failed', error)
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
    [coverageMode, apiBase, coverage, sliderRIndex, makeAxisKey, activeLattice, makeSelectionKey, scanId, ingestAxisSlice, syncSelectionWithCache, requestSelectionIfUncached]
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

  // Prefetch on slider movement
  useEffect(() => {
    const selectedPoint = selected && selected.lattice === activeLattice ? selected : null
    if (!selectedPoint) return
    if (activeSlider === 'column') {
      prefetchColumn(selectedPoint.epsIndex)
    } else if (activeSlider === 'row') {
      prefetchRow(selectedPoint.rIndex)
    }
  }, [activeSlider, selected, activeLattice, prefetchColumn, prefetchRow])

  // Sync selection with cache or fetch
  // Wait for: coverage data loaded (coverageMode ready) + valid selection set
  useEffect(() => {
    // If coverage is still loading (not yet fetched), show loading state
    if (coverageMode === 'loading') {
      setBandState({ status: 'loading' })
      singleFetchControllerRef.current?.abort()
      return
    }

    // If offline/demo mode or no API, stay idle
    if (coverageMode !== 'ready' || !apiBase) {
      setBandState({ status: 'idle' })
      singleFetchControllerRef.current?.abort()
      return
    }

    // Wait for valid selection to be set (ensureSelectionDefaults will set this)
    if (!selected || !Number.isFinite(selected.epsBg) || !Number.isFinite(selected.rOverA)) {
      setBandState({ status: 'loading' })
      singleFetchControllerRef.current?.abort()
      return
    }

    if (activeSlider) {
      syncSelectionWithCache()
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
  }, [selected, coverageMode, apiBase, makeSelectionKey, startSingleFetch, activeSlider, syncSelectionWithCache])

  return {
    bandState,
    setBandState,
    beginSliderInteraction,
    endSliderInteraction,
    prefetchColumn,
    prefetchRow,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export utility functions that may be needed elsewhere
// ─────────────────────────────────────────────────────────────────────────────

export { buildPreviewFromBands, findAxisIndex, computePreviewMax, toBandPreview }
