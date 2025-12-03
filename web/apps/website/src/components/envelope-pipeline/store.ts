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

interface EnvelopePipelineStore {
  // Hovered pixel (synced across all plots)
  hoveredPixel: PixelCoord | null
  setHoveredPixel: (pixel: PixelCoord | null) => void
  
  // Selected pixel (synced across all plots)
  selectedPixel: PixelCoord | null
  setSelectedPixel: (pixel: PixelCoord | null) => void
  
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
  
  clearSelection: () => set({ selectedPixel: null }),
  
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
