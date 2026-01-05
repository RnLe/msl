import { create } from 'zustand'

export interface PixelCoord {
  row: number
  col: number
}

export interface ViewState {
  zoom: number
  panX: number
  panY: number
}

export const DEFAULT_VIEW: ViewState = { zoom: 1, panX: 0, panY: 0 }

// Continuous cursor position in real space (R coordinates)
export interface CursorPosition {
  x: number  // R_x in units of a
  y: number  // R_y in units of a
}

interface EnvelopePipelineStore {
  // Hovered pixel (synced across all plots) - uses data grid coordinates (N × N)
  hoveredPixel: PixelCoord | null
  setHoveredPixel: (pixel: PixelCoord | null) => void
  
  // Selected pixel (synced across all plots) - uses data grid coordinates (N × N)
  selectedPixel: PixelCoord | null
  setSelectedPixel: (pixel: PixelCoord | null) => void
  
  // Continuous cursor position from moiré lattice (real-space R coordinates)
  // This updates continuously as the mouse moves, not just per-pixel
  cursorPosition: CursorPosition | null
  setCursorPosition: (pos: CursorPosition | null) => void
  
  // Selected position in real-space R coordinates (set when clicking on moiré lattice)
  // This is the exact R position, not derived from pixel coordinates
  selectedPosition: CursorPosition | null
  setSelectedPosition: (pos: CursorPosition | null) => void
  
  // Clear selection
  clearSelection: () => void
  
  // Linked zoom mode - when true, all plots zoom/pan together
  linkedZoom: boolean
  setLinkedZoom: (linked: boolean) => void
  
  // Shared view state for linked zoom
  sharedView: ViewState
  setSharedView: (view: ViewState) => void
  
  // Track which plots have zoom mode active (for Link Zoom button visibility)
  zoomModeActivePlots: Set<string>
  setZoomModeActive: (plotId: string, active: boolean) => void
}

export const useEnvelopePipelineStore = create<EnvelopePipelineStore>((set) => ({
  hoveredPixel: null,
  setHoveredPixel: (pixel) => set({ hoveredPixel: pixel }),
  
  selectedPixel: null,
  setSelectedPixel: (pixel) => set({ selectedPixel: pixel }),
  
  cursorPosition: null,
  setCursorPosition: (pos) => set({ cursorPosition: pos }),
  
  selectedPosition: null,
  setSelectedPosition: (pos) => set({ selectedPosition: pos }),
  
  clearSelection: () => set({ selectedPixel: null, cursorPosition: null, selectedPosition: null }),
  
  linkedZoom: false,
  setLinkedZoom: (linked) => set({ linkedZoom: linked }),
  
  sharedView: DEFAULT_VIEW,
  setSharedView: (view) => set({ sharedView: view }),
  
  zoomModeActivePlots: new Set(),
  setZoomModeActive: (plotId, active) => set((state) => {
    const newSet = new Set(state.zoomModeActivePlots)
    if (active) {
      newSet.add(plotId)
    } else {
      newSet.delete(plotId)
    }
    return { zoomModeActivePlots: newSet }
  }),
}))
