'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Phase1FieldsPanel, generateDemoPhase1Data } from './Phase1FieldsPanel'
import { RunExplorer, processManifest } from './RunExplorer'
import type { RunInfo, RunsManifest, Phase1FieldsData, Phase1FieldsMeta } from './types'

interface EnvelopePipelineViewerProps {
  manifestUrl?: string
}

// Loading placeholder for the sidebar
function LoadingExplorer() {
  return (
    <div style={{
      background: '#1a1a1a',
      borderRadius: '8px',
      border: '1px solid #2a2a2a',
      padding: '20px',
      height: '400px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={{
        width: '24px',
        height: '24px',
        border: '2px solid #2a2a2a',
        borderTopColor: '#1a87ce',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
      }} />
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

/**
 * Load phase0 metadata (candidate parameters)
 */
async function loadPhase0Meta(candidatePath: string): Promise<Phase1FieldsData['candidateParams'] | null> {
  try {
    const res = await fetch(`${candidatePath}/phase0_meta.json`)
    if (!res.ok) return null
    const meta = await res.json()
    
    // Map from snake_case JSON to camelCase
    return {
      candidateId: meta.candidate_id ?? 0,
      latticeType: meta.lattice_type ?? 'unknown',
      rOverA: meta.r_over_a ?? 0,
      epsBg: meta.eps_bg ?? 1,
      thetaDeg: meta.theta_deg,
      moireLength: meta.moire_length,
      polarization: meta.polarization,
      bandIndex: meta.band_index,
      kLabel: meta.k_label,
      omega0: meta.omega0,
    }
  } catch (err) {
    console.error('Failed to load phase0 meta:', err)
    return null
  }
}

/**
 * Load phase1 field data from binary format
 */
async function loadPhase1Fields(candidatePath: string): Promise<Phase1FieldsData | null> {
  try {
    // Fetch metadata
    const metaRes = await fetch(`${candidatePath}/phase1_fields_meta.json`)
    if (!metaRes.ok) return null
    const meta: Phase1FieldsMeta = await metaRes.json()
    
    // Fetch binary data
    const binRes = await fetch(`${candidatePath}/phase1_fields.bin`)
    if (!binRes.ok) return null
    const buffer = await binRes.arrayBuffer()
    
    const [rows, cols] = meta.shape
    const floatsPerField = rows * cols
    
    // Parse Float32 arrays from buffer
    const parseField = (offset: number, min: number, max: number) => {
      const data = new Float32Array(buffer, offset, floatsPerField)
      // Convert to 2D array (row-major)
      const values: number[][] = []
      for (let i = 0; i < rows; i++) {
        const row: number[] = []
        for (let j = 0; j < cols; j++) {
          row.push(data[i * cols + j])
        }
        values.push(row)
      }
      return { values, min, max }
    }
    
    return {
      V: parseField(meta.fields.V.offset, meta.fields.V.min, meta.fields.V.max),
      vg_norm: parseField(meta.fields.vg_norm.offset, meta.fields.vg_norm.min, meta.fields.vg_norm.max),
      M_inv_eig1: parseField(meta.fields.M_inv_eig1.offset, meta.fields.M_inv_eig1.min, meta.fields.M_inv_eig1.max),
      M_inv_eig2: parseField(meta.fields.M_inv_eig2.offset, meta.fields.M_inv_eig2.min, meta.fields.M_inv_eig2.max),
      extent: meta.extent,
    }
  } catch (err) {
    console.error('Failed to load phase1 fields:', err)
    return null
  }
}

export function EnvelopePipelineViewer({ manifestUrl = '/data/runs-manifest.json' }: EnvelopePipelineViewerProps) {
  const [manifest, setManifest] = useState<RunsManifest | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedRun, setSelectedRun] = useState<RunInfo | undefined>()
  const [fieldData, setFieldData] = useState<Phase1FieldsData | null>(null)
  const [fieldsLoading, setFieldsLoading] = useState(false)
  
  // Fetch manifest on mount
  useEffect(() => {
    fetch(manifestUrl)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load manifest: ${res.status}`)
        return res.json()
      })
      .then((data: RunsManifest) => {
        setManifest(processManifest(data))
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load runs manifest:', err)
        setError(err.message)
        setLoading(false)
      })
  }, [manifestUrl])
  
  // Load field data when a run is selected
  const handleRunSelect = useCallback(async (run: RunInfo) => {
    setSelectedRun(run)
    setFieldsLoading(true)
    
    // Load both phase1 fields and phase0 metadata in parallel
    const [data, candidateParams] = await Promise.all([
      loadPhase1Fields(run.candidate.path),
      loadPhase0Meta(run.candidate.path),
    ])
    
    if (data) {
      // Merge candidate params if available
      setFieldData({
        ...data,
        candidateParams: candidateParams ?? undefined,
      })
    } else {
      // Fall back to demo data if loading fails
      setFieldData(generateDemoPhase1Data())
    }
    setFieldsLoading(false)
  }, [])
  
  // Show demo data initially
  const displayData = fieldData ?? generateDemoPhase1Data()
  
  return (
    <div style={{ 
      display: 'flex', 
      gap: '24px', 
      alignItems: 'flex-start',
      minHeight: '600px',
    }}>
      <div style={{ 
        width: '168px', 
        flexShrink: 0, 
        position: 'sticky', 
        top: '80px', 
        maxHeight: 'calc(100vh - 100px)' 
      }}>
        {loading ? (
          <LoadingExplorer />
        ) : error ? (
          <div style={{
            background: '#1a1a1a',
            borderRadius: '8px',
            border: '1px solid #ef4444',
            padding: '20px',
            color: '#f87171',
            fontSize: '12px',
          }}>
            <strong>Error loading manifest</strong>
            <div style={{ marginTop: '8px', color: '#a3a3a3', fontSize: '11px' }}>
              {error}
            </div>
            <div style={{ marginTop: '12px', color: '#737373', fontSize: '10px' }}>
              Run: <code>pnpm run build:runs-manifest</code>
            </div>
          </div>
        ) : manifest ? (
          <RunExplorer 
            manifest={manifest} 
            onRunSelect={handleRunSelect}
            selectedRun={selectedRun}
          />
        ) : null}
      </div>
      
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ padding: '1rem 0' }}>
          <Phase1FieldsPanel data={displayData} loading={fieldsLoading} />
        </div>
      </div>
    </div>
  )
}
