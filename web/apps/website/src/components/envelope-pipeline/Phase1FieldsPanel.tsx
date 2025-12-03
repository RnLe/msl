'use client'

import React, { useMemo, useCallback, useRef, useEffect } from 'react'
import { RotateCcw, Link2 } from 'lucide-react'
import katex from 'katex'
import type { FieldData, Phase1FieldsData } from './types'
import { useEnvelopePipelineStore, DEFAULT_VIEW, type ViewState } from './store'

// Helper to render KaTeX inline
function KaTeX({ expr }: { expr: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(expr, { throwOnError: false, output: 'html' })
    } catch (e) {
      return expr
    }
  }, [expr])
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

// Map lattice types to display names
function formatLatticeType(type: string): string {
  switch (type.toLowerCase()) {
    case 'hex':
    case 'hexagonal':
      return 'Triangular'
    case 'sq':
    case 'square':
      return 'Square'
    case 'rect':
    case 'rectangular':
      return 'Rectangular'
    case 'oblique':
      return 'Oblique'
    default:
      return type.charAt(0).toUpperCase() + type.slice(1)
  }
}

// Color scales for different field types
const VIRIDIS_COLORS = [
  '#440154', '#482878', '#3e4989', '#31688e', '#26828e',
  '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'
]

const RDBU_COLORS = [
  '#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
  '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'
]

const PLASMA_COLORS = [
  '#0d0887', '#41049d', '#6a00a8', '#8f0da4', '#b12a90',
  '#cc4778', '#e16462', '#f2844b', '#fca636', '#fcce25'
]

interface CanvasFieldPlotProps {
  plotId: string
  data: FieldData
  extent: Phase1FieldsData['extent']
  colorScale: string[]
  title: string
  subtitle: string
  colorbarLabel: string
  symmetric?: boolean
  logScale?: boolean
}

function interpolateColor(colors: string[], t: number): string {
  const clampedT = Math.max(0, Math.min(1, t))
  const n = colors.length - 1
  const idx = clampedT * n
  const lower = Math.floor(idx)
  const upper = Math.min(lower + 1, n)
  const frac = idx - lower
  
  const c1 = hexToRgb(colors[lower])
  const c2 = hexToRgb(colors[upper])
  
  return `rgb(${
    Math.round(c1.r + frac * (c2.r - c1.r))
  }, ${
    Math.round(c1.g + frac * (c2.g - c1.g))
  }, ${
    Math.round(c1.b + frac * (c2.b - c1.b))
  })`
}

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : { r: 0, g: 0, b: 0 }
}

function CanvasFieldPlot({
  plotId,
  data,
  extent,
  colorScale,
  title,
  subtitle,
  colorbarLabel,
  symmetric = false,
  logScale = false,
}: CanvasFieldPlotProps) {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [dimensions, setDimensions] = React.useState({ width: 280, height: 260 })
  
  // Local zoom and pan state (used when not linked)
  const [localView, setLocalView] = React.useState<ViewState>(DEFAULT_VIEW)
  const [isDragging, setIsDragging] = React.useState(false)
  const [dragStart, setDragStart] = React.useState({ x: 0, y: 0 })
  const [zoomModeActive, setZoomModeActive] = React.useState(false)
  
  // Store access for linked zoom
  const { 
    linkedZoom, setLinkedZoom,
    sharedView, setSharedView,
    setZoomModeActive: setStoreZoomModeActive,
    hoveredPixel, setHoveredPixel, 
    selectedPixel, setSelectedPixel 
  } = useEnvelopePipelineStore()
  
  // Use shared view when linked, local view otherwise
  const view = linkedZoom ? sharedView : localView
  const setView = linkedZoom ? setSharedView : setLocalView
  
  const isDefaultView = view.zoom === 1 && view.panX === 0 && view.panY === 0
  
  // Effective zoom mode: either local zoom mode is on, or linked zoom is enabled
  const effectiveZoomMode = zoomModeActive || linkedZoom
  
  // Update store when zoom mode changes
  useEffect(() => {
    setStoreZoomModeActive(plotId, zoomModeActive)
    return () => setStoreZoomModeActive(plotId, false)
  }, [plotId, zoomModeActive, setStoreZoomModeActive])
  
  const resetView = useCallback(() => {
    setView(DEFAULT_VIEW)
  }, [setView])
  
  const toggleZoomMode = useCallback(() => {
    setZoomModeActive(prev => !prev)
  }, [])
  
  const handleLinkZoom = useCallback(() => {
    // When linking, sync current view to shared view
    if (!linkedZoom) {
      setSharedView(localView)
    }
    setLinkedZoom(true)
  }, [linkedZoom, localView, setSharedView, setLinkedZoom])
  
  const handleUnlinkZoom = useCallback(() => {
    // When unlinking, copy shared view to local
    setLocalView(sharedView)
    setLinkedZoom(false)
  }, [sharedView, setLinkedZoom])
  
  // Observe container size
  React.useEffect(() => {
    const container = containerRef.current
    if (!container) return
    
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        const w = Math.floor(entry.contentRect.width)
        // Maintain aspect ratio (slightly taller than wide for colorbar)
        const h = Math.floor(w * 0.95)
        setDimensions({ width: w, height: h })
      }
    })
    
    observer.observe(container)
    return () => observer.disconnect()
  }, [])
  
  // Click outside to deselect - only one plot needs to handle this
  useEffect(() => {
    // Only the first plot (V) handles the global click listener to avoid duplicates
    if (plotId !== 'V') return
    
    const handleClickOutside = (e: MouseEvent) => {
      // Check if click is on any canvas element
      const target = e.target as HTMLElement
      if (target.tagName === 'CANVAS') return
      
      // Check if click is on a button (zoom, link, reset buttons)
      if (target.tagName === 'BUTTON' || target.closest('button')) return
      
      // Clear selection
      setSelectedPixel(null)
    }
    
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [plotId, setSelectedPixel])
  
  const { width, height } = dimensions
  const plotPadding = { top: 50, right: 55, bottom: 35, left: 45 }
  const plotWidth = width - plotPadding.left - plotPadding.right
  const plotHeight = height - plotPadding.top - plotPadding.bottom
  
  const { normalizedValues, effectiveMin, effectiveMax } = useMemo(() => {
    let min = data.min
    let max = data.max
    
    if (symmetric) {
      const absMax = Math.max(Math.abs(min), Math.abs(max))
      min = -absMax
      max = absMax
    }
    
    if (logScale) {
      // Transform to log scale
      const transformed = data.values.map(row =>
        row.map(v => {
          const absV = Math.abs(v)
          return absV > 1e-12 ? Math.log10(absV) : -12
        })
      )
      const flatTransformed = transformed.flat()
      min = Math.min(...flatTransformed)
      max = Math.max(...flatTransformed)
      return { normalizedValues: transformed, effectiveMin: min, effectiveMax: max }
    }
    
    return { normalizedValues: data.values, effectiveMin: min, effectiveMax: max }
  }, [data, symmetric, logScale])
  
  // Clamp pan values to keep data fully visible (no empty space)
  const clampPan = useCallback((panX: number, panY: number, zoom: number) => {
    // When zoomed in, the visible area is 1/zoom of the total
    // We want to constrain panning so edges of data align with edges of view
    // Pan range: from 0 (data left edge at view left) to (zoom-1)/zoom (data right edge at view right)
    const maxPan = Math.max(0, (zoom - 1) / (2 * zoom))
    return {
      panX: Math.max(-maxPan, Math.min(maxPan, panX)),
      panY: Math.max(-maxPan, Math.min(maxPan, panY)),
    }
  }, [])
  
  // Get pixel coordinates from mouse position
  const getPixelFromMouse = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    
    const rect = canvas.getBoundingClientRect()
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top
    
    const scaleX = width / rect.width
    const scaleY = height / rect.height
    const canvasX = mouseX * scaleX
    const canvasY = mouseY * scaleY
    
    // Check if mouse is within plot area
    if (canvasX < plotPadding.left || canvasX > width - plotPadding.right ||
        canvasY < plotPadding.top || canvasY > height - plotPadding.bottom) {
      return null
    }
    
    // Get rows/cols from data
    const rows = data.values.length
    const cols = data.values[0]?.length || 0
    
    // Calculate visible range based on zoom and pan
    const visibleWidth = 1 / view.zoom
    const visibleHeight = 1 / view.zoom
    const viewLeft = 0.5 - visibleWidth / 2 - view.panX
    const viewTop = 0.5 - visibleHeight / 2 - view.panY
    
    // Convert canvas position to normalized position
    const screenX = (canvasX - plotPadding.left) / plotWidth
    const screenY = (canvasY - plotPadding.top) / plotHeight
    
    // Convert screen position to data position
    const normX = viewLeft + screenX * visibleWidth
    const normY = viewTop + screenY * visibleHeight
    
    // Convert to row/col indices
    const col = Math.floor(normX * cols)
    const row = rows - 1 - Math.floor(normY * rows)
    
    // Check bounds
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      return null
    }
    
    return { row, col }
  }, [data.values, width, height, plotPadding, plotWidth, plotHeight, view])
  
  // Store current values in refs for the native event handler
  const effectiveZoomModeRef = useRef(effectiveZoomMode)
  const viewRef = useRef(view)
  const setViewRef = useRef(setView)
  const dimensionsRef = useRef({ width, height, plotWidth, plotHeight, plotPadding })
  
  useEffect(() => {
    effectiveZoomModeRef.current = effectiveZoomMode
  }, [effectiveZoomMode])
  
  useEffect(() => {
    viewRef.current = view
  }, [view])
  
  useEffect(() => {
    setViewRef.current = setView
  }, [setView])
  
  useEffect(() => {
    dimensionsRef.current = { width, height, plotWidth, plotHeight, plotPadding }
  }, [width, height, plotWidth, plotHeight, plotPadding])
  
  // Native wheel handler - must use passive: false to prevent scroll
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const handleWheel = (e: WheelEvent) => {
      // Only handle if zoom mode is active (local or linked)
      if (!effectiveZoomModeRef.current) return
      
      const { width, height, plotWidth, plotHeight, plotPadding } = dimensionsRef.current
      const view = viewRef.current
      
      const rect = canvas.getBoundingClientRect()
      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top
      
      // Check if mouse is within plot area
      const scaleX = width / rect.width
      const scaleY = height / rect.height
      const canvasX = mouseX * scaleX
      const canvasY = mouseY * scaleY
      
      if (canvasX < plotPadding.left || canvasX > width - plotPadding.right ||
          canvasY < plotPadding.top || canvasY > height - plotPadding.bottom) {
        return
      }
      
      // Block scroll propagation - must be after plot area check
      e.preventDefault()
      e.stopPropagation()
      
      // Calculate zoom
      const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
      const newZoom = Math.max(1, Math.min(20, view.zoom * zoomFactor))
      
      if (newZoom === view.zoom) return
      
      // Get cursor position in plot area (0-1 screen space)
      const screenX = (canvasX - plotPadding.left) / plotWidth
      const screenY = (canvasY - plotPadding.top) / plotHeight
      
      // Convert screen position to data position (what data point is under cursor)
      // Current view: visibleWidth = 1/zoom, viewLeft = 0.5 - visibleWidth/2 - panX
      const oldVisibleWidth = 1 / view.zoom
      const oldVisibleHeight = 1 / view.zoom
      const oldViewLeft = 0.5 - oldVisibleWidth / 2 - view.panX
      const oldViewTop = 0.5 - oldVisibleHeight / 2 - view.panY
      
      // Data position under cursor
      const dataX = oldViewLeft + screenX * oldVisibleWidth
      const dataY = oldViewTop + screenY * oldVisibleHeight
      
      // After zoom, we want the same data point to be at the same screen position
      // New view: newViewLeft = 0.5 - newVisibleWidth/2 - newPanX
      // dataX = newViewLeft + screenX * newVisibleWidth
      // Solving for newPanX:
      // dataX = 0.5 - (1/newZoom)/2 - newPanX + screenX * (1/newZoom)
      // newPanX = 0.5 - (1/newZoom)/2 + screenX * (1/newZoom) - dataX
      const newVisibleWidth = 1 / newZoom
      const newVisibleHeight = 1 / newZoom
      
      const newPanX = 0.5 - newVisibleWidth / 2 + screenX * newVisibleWidth - dataX
      const newPanY = 0.5 - newVisibleHeight / 2 + screenY * newVisibleHeight - dataY
      
      // Clamp pan to keep data fully visible
      const maxPan = Math.max(0, (newZoom - 1) / (2 * newZoom))
      const clampedPanX = Math.max(-maxPan, Math.min(maxPan, newPanX))
      const clampedPanY = Math.max(-maxPan, Math.min(maxPan, newPanY))
      
      setViewRef.current({
        zoom: newZoom,
        panX: clampedPanX,
        panY: clampedPanY,
      })
    }
    
    // Attach with passive: false to allow preventDefault
    canvas.addEventListener('wheel', handleWheel, { passive: false })
    
    return () => {
      canvas.removeEventListener('wheel', handleWheel)
    }
  }, []) // Empty deps - handler uses refs for current values
  
  // Handle mouse down for panning or selection
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return // Only left click
    
    // If zoom mode is active (local or linked), handle panning
    if (effectiveZoomMode) {
      const canvas = canvasRef.current
      if (!canvas) return
      
      const rect = canvas.getBoundingClientRect()
      const mouseX = e.clientX - rect.left
      const mouseY = e.clientY - rect.top
      
      // Check if mouse is within plot area
      const scaleX = width / rect.width
      const scaleY = height / rect.height
      const canvasX = mouseX * scaleX
      const canvasY = mouseY * scaleY
      
      if (canvasX < plotPadding.left || canvasX > width - plotPadding.right ||
          canvasY < plotPadding.top || canvasY > height - plotPadding.bottom) {
        return
      }
      
      setIsDragging(true)
      setDragStart({ x: e.clientX, y: e.clientY })
    } else {
      // If not in zoom mode, handle pixel selection
      const pixel = getPixelFromMouse(e)
      if (pixel) {
        // Toggle selection: if clicking the same pixel, deselect
        if (selectedPixel && selectedPixel.row === pixel.row && selectedPixel.col === pixel.col) {
          setSelectedPixel(null)
        } else {
          setSelectedPixel(pixel)
        }
      }
    }
  }, [effectiveZoomMode, width, height, plotPadding, getPixelFromMouse, selectedPixel, setSelectedPixel])
  
  // Handle mouse move for panning and hover
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // Always track hover
    const pixel = getPixelFromMouse(e)
    setHoveredPixel(pixel)
    
    // Handle panning if dragging
    if (!isDragging) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const rect = canvas.getBoundingClientRect()
    const scaleX = plotWidth / rect.width
    const scaleY = plotHeight / rect.height
    
    const dx = (e.clientX - dragStart.x) * scaleX / plotWidth / view.zoom
    const dy = (e.clientY - dragStart.y) * scaleY / plotHeight / view.zoom
    
    const clamped = clampPan(view.panX + dx, view.panY + dy, view.zoom)
    setView({
      ...view,
      panX: clamped.panX,
      panY: clamped.panY,
    })
    
    setDragStart({ x: e.clientX, y: e.clientY })
  }, [isDragging, dragStart, view, plotWidth, plotHeight, clampPan, getPixelFromMouse, setHoveredPixel, setView])
  
  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])
  
  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setIsDragging(false)
    setHoveredPixel(null)
  }, [setHoveredPixel])
  
  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // Clear canvas with transparency
    ctx.clearRect(0, 0, width, height)
    
    // Draw the field as pixels with zoom/pan
    const rows = normalizedValues.length
    const cols = normalizedValues[0]?.length || 0
    
    // Calculate visible range based on zoom and pan
    const visibleWidth = 1 / view.zoom
    const visibleHeight = 1 / view.zoom
    const viewLeft = 0.5 - visibleWidth / 2 - view.panX
    const viewTop = 0.5 - visibleHeight / 2 - view.panY
    
    // Save context for clipping
    ctx.save()
    ctx.beginPath()
    ctx.rect(plotPadding.left, plotPadding.top, plotWidth, plotHeight)
    ctx.clip()
    
    // Calculate cell positions with zoom/pan
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = normalizedValues[i][j]
        const t = (value - effectiveMin) / (effectiveMax - effectiveMin || 1)
        const color = interpolateColor(colorScale, t)
        
        // Original normalized position (0-1)
        const normX = j / cols
        const normY = (rows - 1 - i) / rows
        
        // Transform by view
        const screenX = (normX - viewLeft) / visibleWidth
        const screenY = (normY - viewTop) / visibleHeight
        const cellW = (1 / cols) / visibleWidth
        const cellH = (1 / rows) / visibleHeight
        
        // Convert to canvas coordinates
        const x = plotPadding.left + screenX * plotWidth
        const y = plotPadding.top + screenY * plotHeight
        const w = cellW * plotWidth
        const h = cellH * plotHeight
        
        // Only draw if visible
        if (x + w > plotPadding.left && x < plotPadding.left + plotWidth &&
            y + h > plotPadding.top && y < plotPadding.top + plotHeight) {
          ctx.fillStyle = color
          ctx.fillRect(x, y, Math.ceil(w) + 1, Math.ceil(h) + 1)
        }
      }
    }
    
    // Draw hover and selection highlights (still within clip)
    const drawHighlight = (row: number, col: number, style: 'hover' | 'selected') => {
      const normX = col / cols
      const normY = (rows - 1 - row) / rows
      
      const screenX = (normX - viewLeft) / visibleWidth
      const screenY = (normY - viewTop) / visibleHeight
      const cellW = (1 / cols) / visibleWidth
      const cellH = (1 / rows) / visibleHeight
      
      const x = plotPadding.left + screenX * plotWidth
      const y = plotPadding.top + screenY * plotHeight
      const w = cellW * plotWidth
      const h = cellH * plotHeight
      
      // Only draw if visible
      if (x + w > plotPadding.left && x < plotPadding.left + plotWidth &&
          y + h > plotPadding.top && y < plotPadding.top + plotHeight) {
        if (style === 'selected') {
          // Selected: blue border
          ctx.strokeStyle = '#1b97ea'
          ctx.lineWidth = 2
          ctx.strokeRect(x + 1, y + 1, Math.ceil(w) - 2, Math.ceil(h) - 2)
        } else {
          // Hover: white semi-transparent overlay
          ctx.fillStyle = 'rgba(255, 255, 255, 0.3)'
          ctx.fillRect(x, y, Math.ceil(w), Math.ceil(h))
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
          ctx.lineWidth = 1
          ctx.strokeRect(x, y, Math.ceil(w), Math.ceil(h))
        }
      }
    }
    
    // Draw selection first (under hover)
    if (selectedPixel) {
      drawHighlight(selectedPixel.row, selectedPixel.col, 'selected')
    }
    
    // Draw hover on top
    if (hoveredPixel && !(selectedPixel && 
        hoveredPixel.row === selectedPixel.row && 
        hoveredPixel.col === selectedPixel.col)) {
      drawHighlight(hoveredPixel.row, hoveredPixel.col, 'hover')
    }
    
    ctx.restore()
    
    // Draw axes border
    ctx.strokeStyle = '#6b7280'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.rect(plotPadding.left, plotPadding.top, plotWidth, plotHeight)
    ctx.stroke()
    
    // Draw title (leave space for reset button)
    ctx.fillStyle = '#f9fafb'
    ctx.font = 'bold 14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(title, width / 2, 20)
    
    // Draw subtitle
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px system-ui, sans-serif'
    ctx.fillText(subtitle, width / 2, 38)
    
    // Draw axis labels - larger with subscript style
    ctx.fillStyle = '#d1d5db'
    ctx.font = '14px system-ui, sans-serif'
    ctx.textAlign = 'center'
    // X-axis label: Rₓ/a
    ctx.fillText('Rₓ / a', plotPadding.left + plotWidth / 2, height - 6)
    
    // Y-axis label: Rᵧ/a
    ctx.save()
    ctx.translate(14, plotPadding.top + plotHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText('Rᵧ / a', 0, 0)
    ctx.restore()
    
    // Draw colorbar
    const cbWidth = 14
    const cbHeight = plotHeight
    const cbX = width - plotPadding.right + 12
    const cbY = plotPadding.top
    
    for (let i = 0; i < cbHeight; i++) {
      const t = 1 - i / cbHeight
      ctx.fillStyle = interpolateColor(colorScale, t)
      ctx.fillRect(cbX, cbY + i, cbWidth, 1)
    }
    
    ctx.strokeStyle = '#6b7280'
    ctx.strokeRect(cbX, cbY, cbWidth, cbHeight)
    
    // Colorbar labels
    ctx.fillStyle = '#d1d5db'
    ctx.font = '11px system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(effectiveMax.toExponential(1), cbX + cbWidth + 4, cbY + 12)
    ctx.fillText(effectiveMin.toExponential(1), cbX + cbWidth + 4, cbY + cbHeight)
    
    // Colorbar title
    ctx.save()
    ctx.translate(cbX + cbWidth + 40, cbY + cbHeight / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.font = '12px system-ui, sans-serif'
    ctx.fillText(colorbarLabel, 0, 0)
    ctx.restore()
    
  }, [normalizedValues, effectiveMin, effectiveMax, colorScale, title, subtitle, colorbarLabel, width, height, plotPadding, plotWidth, plotHeight, view, hoveredPixel, selectedPixel])
  
  return (
    <div 
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        minHeight: '200px',
      }}
    >
      {/* Button group container */}
      <div style={{
        position: 'absolute',
        top: '4px',
        left: '4px',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
      }}>
        {/* Zoom mode toggle button */}
        <button
          onClick={toggleZoomMode}
          title={zoomModeActive ? "Exit zoom mode" : "Enter zoom mode (scroll to zoom, drag to pan)"}
          style={{
            background: 'transparent',
            border: zoomModeActive ? '1px solid rgba(255, 255, 255, 0.5)' : '1px solid transparent',
            borderRadius: '4px',
            padding: '3px 8px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s',
            fontSize: '11px',
            fontWeight: 500,
            color: zoomModeActive ? '#ffffff' : '#888888',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = '#ffffff'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = zoomModeActive ? '#ffffff' : '#888888'
          }}
        >
          Zoom
        </button>
        
        {/* Link Zoom button - only shown when zoom mode is active */}
        {zoomModeActive && (
          <button
            onClick={linkedZoom ? handleUnlinkZoom : handleLinkZoom}
            title={linkedZoom ? "Unlink zoom (stop syncing)" : "Link zoom across all plots"}
            style={{
              background: 'transparent',
              border: linkedZoom ? '1px solid rgba(255, 255, 255, 0.5)' : '1px solid transparent',
              borderRadius: '4px',
              padding: '3px 8px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '4px',
              transition: 'all 0.15s',
              fontSize: '11px',
              fontWeight: 500,
              color: linkedZoom ? '#ffffff' : '#888888',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#ffffff'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = linkedZoom ? '#ffffff' : '#888888'
            }}
          >
            {linkedZoom ? <Link2 size={12} /> : 'Link'}
          </button>
        )}
        
        {/* Reset button - only shown when zoomed/panned */}
        {!isDefaultView && (
          <button
            onClick={resetView}
            title="Reset zoom and pan"
            style={{
              background: 'transparent',
              border: '1px solid transparent',
              borderRadius: '4px',
              padding: '3px 6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s',
              color: '#888888',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#ffffff'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#888888'
            }}
          >
            <RotateCcw size={12} />
          </button>
        )}
      </div>
      
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          borderRadius: '8px',
          background: 'transparent',
          width: '100%',
          height: 'auto',
          cursor: isDragging ? 'grabbing' : 'default',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  )
}

interface Phase1FieldsPanelProps {
  data: Phase1FieldsData
  loading?: boolean
}

// Loading spinner component for title area
function TitleLoadingSpinner() {
  return (
    <>
      <span style={{
        display: 'inline-block',
        width: '14px',
        height: '14px',
        marginLeft: '10px',
        border: '2px solid #3a3a3a',
        borderTopColor: '#1a87ce',
        borderRadius: '50%',
        animation: 'envelope-spin 1s linear infinite',
        verticalAlign: 'middle',
      }} />
      <style>{`
        @keyframes envelope-spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  )
}

export function Phase1FieldsPanel({ data, loading = false }: Phase1FieldsPanelProps) {
  const candidateInfo = data.candidateParams
  
  return (
    <div style={{ background: 'transparent', borderRadius: '12px', padding: '16px' }}>
      {candidateInfo && (
        <div style={{ marginBottom: '16px', textAlign: 'center' }}>
          <h3 style={{ 
            color: '#f9fafb', 
            fontSize: '20px', 
            fontWeight: 600, 
            margin: 0,
            letterSpacing: '-0.01em',
          }}>
            Candidate {candidateInfo.candidateId} — {formatLatticeType(candidateInfo.latticeType)}
            {candidateInfo.polarization && (
              <span style={{ 
                marginLeft: '8px', 
                fontSize: '14px', 
                color: '#60a5fa',
                fontWeight: 500,
              }}>
                {candidateInfo.polarization}
              </span>
            )}
            {loading && <TitleLoadingSpinner />}
          </h3>
          <p style={{ 
            color: '#d1d5db', 
            fontSize: '16px', 
            margin: '8px 0 0 0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            flexWrap: 'wrap',
          }}>
            <span><KaTeX expr="r/a" /> = {candidateInfo.rOverA.toFixed(2)}</span>
            <span><KaTeX expr="\varepsilon" /> = {candidateInfo.epsBg.toFixed(1)}</span>
            {candidateInfo.thetaDeg !== undefined && (
              <span><KaTeX expr="\theta" /> = {candidateInfo.thetaDeg.toFixed(2)}°</span>
            )}
            {candidateInfo.moireLength !== undefined && (
              <span><KaTeX expr="L_M" /> = {candidateInfo.moireLength.toFixed(1)}a</span>
            )}
          </p>
        </div>
      )}
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '12px',
      }}>
        <CanvasFieldPlot
          plotId="V"
          data={data.V}
          extent={data.extent}
          colorScale={RDBU_COLORS}
          title="Potential Landscape"
          subtitle={`V(R)`}
          colorbarLabel="Δω"
          symmetric
        />
        
        <CanvasFieldPlot
          plotId="vg_norm"
          data={data.vg_norm}
          extent={data.extent}
          colorScale={VIRIDIS_COLORS}
          title="Group Velocity"
          subtitle={`|vg(R)| max: ${data.vg_norm.max.toExponential(2)}`}
          colorbarLabel="|vg|"
        />
        
        <CanvasFieldPlot
          plotId="M_inv_eig1"
          data={data.M_inv_eig1}
          extent={data.extent}
          colorScale={PLASMA_COLORS}
          title="Mass Tensor λ₁"
          subtitle="log₁₀λ(M⁻¹) (small)"
          colorbarLabel="log₁₀λ"
          logScale
        />
        
        <CanvasFieldPlot
          plotId="M_inv_eig2"
          data={data.M_inv_eig2}
          extent={data.extent}
          colorScale={PLASMA_COLORS}
          title="Mass Tensor λ₂"
          subtitle="log₁₀λ(M⁻¹) (large)"
          colorbarLabel="log₁₀λ"
          logScale
        />
      </div>
    </div>
  )
}

// Demo data generator for testing
export function generateDemoPhase1Data(): Phase1FieldsData {
  const gridSize = 32
  const extent = { xMin: -26, xMax: 26, yMin: -26, yMax: 26 }
  
  // Generate synthetic field data
  const V: number[][] = []
  const vg_norm: number[][] = []
  const M_inv_eig1: number[][] = []
  const M_inv_eig2: number[][] = []
  
  for (let i = 0; i < gridSize; i++) {
    const vRow: number[] = []
    const vgRow: number[] = []
    const m1Row: number[] = []
    const m2Row: number[] = []
    
    for (let j = 0; j < gridSize; j++) {
      const x = extent.xMin + (j / (gridSize - 1)) * (extent.xMax - extent.xMin)
      const y = extent.yMin + (i / (gridSize - 1)) * (extent.yMax - extent.yMin)
      
      // Moiré-like potential with hexagonal symmetry
      const theta = 2 * Math.PI / 3
      const k1 = 0.12
      const pot = (
        Math.cos(k1 * x) +
        Math.cos(k1 * (x * Math.cos(theta) + y * Math.sin(theta))) +
        Math.cos(k1 * (x * Math.cos(2 * theta) + y * Math.sin(2 * theta)))
      ) / 3 * 0.05
      
      vRow.push(pot)
      vgRow.push(Math.abs(pot) * 0.1 + 0.001 * Math.random())
      m1Row.push(0.1 + 0.05 * Math.random())
      m2Row.push(10 + 5 * Math.random())
    }
    
    V.push(vRow)
    vg_norm.push(vgRow)
    M_inv_eig1.push(m1Row)
    M_inv_eig2.push(m2Row)
  }
  
  const flatten = (arr: number[][]) => arr.flat()
  
  return {
    V: {
      values: V,
      min: Math.min(...flatten(V)),
      max: Math.max(...flatten(V)),
    },
    vg_norm: {
      values: vg_norm,
      min: Math.min(...flatten(vg_norm)),
      max: Math.max(...flatten(vg_norm)),
    },
    M_inv_eig1: {
      values: M_inv_eig1,
      min: Math.min(...flatten(M_inv_eig1)),
      max: Math.max(...flatten(M_inv_eig1)),
    },
    M_inv_eig2: {
      values: M_inv_eig2,
      min: Math.min(...flatten(M_inv_eig2)),
      max: Math.max(...flatten(M_inv_eig2)),
    },
    extent,
    candidateParams: {
      candidateId: 1737,
      latticeType: 'hex',
      rOverA: 0.3,
      epsBg: 8.0,
      thetaDeg: 1.1,
      moireLength: 52.09,
    },
  }
}
