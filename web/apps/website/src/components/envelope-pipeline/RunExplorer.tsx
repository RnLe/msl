'use client'

import React, { useMemo } from 'react'
import type { CandidateInfo, RunInfo, RunsManifest, PipelineType } from './types'

// Phase file requirements for each phase
const PHASE_REQUIREMENTS: Record<string, string[]> = {
  phase0: ['phase0_meta.json'],
  phase1: ['phase1_band_data.h5', 'phase1_field_stats.json'],
  phase2: ['phase2_operator.npz', 'phase2_R_grid.npy', 'phase2_operator_meta.json', 'phase2_report.md'],
  phase3: ['phase3_eigenstates.h5', 'phase3_eigenvalues.csv', 'phase3_report.md'],
  phase4: ['phase4_bandstructure.csv', 'phase4_gamma_modes.h5', 'phase4_validation_summary.csv'],
  phase5: ['phase5_q_factor_results.csv', 'phase5_report.md'],
}

// Ordered list of phase keys for display
const PHASE_ORDER = ['phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5']

interface PhaseDotsProps {
  phaseStatus: Record<string, boolean>
}

function PhaseDots({ phaseStatus }: PhaseDotsProps) {
  return (
    <div style={{ display: 'flex', gap: '4px' }}>
      {PHASE_ORDER.map((phase, idx) => {
        const isComplete = phaseStatus[phase] ?? false
        return (
          <div
            key={phase}
            title={`Phase ${idx}: ${isComplete ? 'Complete' : 'Incomplete'}`}
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: isComplete ? '#22c55e' : '#ef4444',
              boxShadow: isComplete 
                ? '0 0 6px rgba(34, 197, 94, 0.6)' 
                : '0 0 4px rgba(239, 68, 68, 0.4)',
            }}
          />
        )
      })}
    </div>
  )
}

function PipelineLabel({ type }: { type: PipelineType }) {
  const isBlaze = type === 'blaze'
  return (
    <span
      style={{
        fontSize: '10px',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        color: isBlaze ? '#f97316' : '#6366f1',
        background: isBlaze ? 'rgba(249, 115, 22, 0.15)' : 'rgba(99, 102, 241, 0.15)',
        padding: '2px 6px',
        borderRadius: '4px',
      }}
    >
      {type}
    </span>
  )
}

interface RunItemProps {
  run: RunInfo
  isSelected: boolean
  onSelect: (run: RunInfo) => void
}

function RunItem({ run, isSelected, onSelect }: RunItemProps) {
  return (
    <button
      onClick={() => onSelect(run)}
      style={{
        display: 'block',
        width: '100%',
        padding: '10px 12px',
        background: isSelected ? '#262626' : 'transparent',
        border: isSelected ? '1px solid #333333' : '1px solid transparent',
        borderRadius: '6px',
        cursor: 'pointer',
        textAlign: 'left',
        marginBottom: '4px',
        transition: 'all 0.15s',
      }}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.background = '#222222'
          e.currentTarget.style.borderColor = '#2a2a2a'
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.background = 'transparent'
          e.currentTarget.style.borderColor = 'transparent'
        }
      }}
    >
      {/* Row 1: Timestamp */}
      <div style={{ 
        fontSize: '11px', 
        fontWeight: 500, 
        color: isSelected ? '#e5e7eb' : '#a3a3a3',
        marginBottom: '6px',
      }}>
        {run.timestamp}
      </div>
      
      {/* Row 2: Phase dots + Pipeline label */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        gap: '8px',
      }}>
        <PhaseDots phaseStatus={run.candidate.phaseStatus} />
        <PipelineLabel type={run.pipelineType} />
      </div>
    </button>
  )
}

interface RunExplorerProps {
  manifest: RunsManifest
  onRunSelect?: (run: RunInfo) => void
  selectedRun?: RunInfo
}

export function RunExplorer({ manifest, onRunSelect, selectedRun }: RunExplorerProps) {
  // Sort runs by timestamp (newest first)
  const sortedRuns = useMemo(() => {
    return [...manifest.runs].sort((a, b) => b.timestamp.localeCompare(a.timestamp))
  }, [manifest.runs])
  
  const handleSelect = (run: RunInfo) => {
    onRunSelect?.(run)
  }
  
  return (
    <div style={{
      background: '#1a1a1a',
      borderRadius: '8px',
      border: '1px solid #2a2a2a',
      overflow: 'hidden',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid #2a2a2a',
        background: '#222222',
      }}>
        <h3 style={{ 
          margin: 0, 
          fontSize: '11px', 
          fontWeight: 600, 
          color: '#a3a3a3',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}>
          Pipeline Runs
        </h3>
      </div>
      
      {/* Run list */}
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '6px',
      }}>
        {sortedRuns.length === 0 ? (
          <div style={{ 
            padding: '20px', 
            textAlign: 'center', 
            color: '#737373',
            fontSize: '11px',
          }}>
            No pipeline runs found.
            <br />
            <span style={{ fontSize: '10px', marginTop: '4px', display: 'block' }}>
              Run the manifest generator script.
            </span>
          </div>
        ) : (
          sortedRuns.map((run) => (
            <RunItem
              key={run.path}
              run={run}
              isSelected={selectedRun?.path === run.path}
              onSelect={handleSelect}
            />
          ))
        )}
      </div>
    </div>
  )
}

// Helper function to compute phase status from file list
export function computePhaseStatus(files: string[]): Record<string, boolean> {
  const status: Record<string, boolean> = {}
  
  for (const phase of Object.keys(PHASE_REQUIREMENTS)) {
    const required = PHASE_REQUIREMENTS[phase]
    status[phase] = required.every(req => files.includes(req))
  }
  
  return status
}

// Demo manifest for testing (updated to new format)
export function generateDemoManifest(): RunsManifest {
  return {
    generatedAt: new Date().toISOString(),
    runs: [
      {
        name: 'phase0_real_run_20251119_172848',
        path: '/runs/phase0_real_run_20251119_172848',
        timestamp: '2025-11-19 17:28:48',
        pipelineType: 'mpb',
        candidate: {
          id: 'candidate_1737',
          path: '/runs/phase0_real_run_20251119_172848/candidate_1737',
          files: [
            'phase0_meta.json',
            'phase1_band_data.h5', 'phase1_field_stats.json',
            'phase2_operator.npz', 'phase2_R_grid.npy', 'phase2_operator_meta.json', 'phase2_report.md',
            'phase3_eigenstates.h5', 'phase3_eigenvalues.csv', 'phase3_report.md',
            'phase4_bandstructure.csv', 'phase4_gamma_modes.h5', 'phase4_validation_summary.csv',
            'phase5_q_factor_results.csv', 'phase5_report.md',
          ],
          phaseStatus: {},
        },
      },
      {
        name: 'phase0_real_run_20251120_151835',
        path: '/runs/phase0_real_run_20251120_151835',
        timestamp: '2025-11-20 15:18:35',
        pipelineType: 'mpb',
        candidate: {
          id: 'candidate_1737',
          path: '/runs/phase0_real_run_20251120_151835/candidate_1737',
          files: [
            'phase0_meta.json',
            'phase1_band_data.h5', 'phase1_field_stats.json',
            'phase2_operator.npz', 'phase2_R_grid.npy',
          ],
          phaseStatus: {},
        },
      },
      {
        name: 'phase0_blaze_20251201_140949',
        path: '/runs/phase0_blaze_20251201_140949',
        timestamp: '2025-12-01 14:09:49',
        pipelineType: 'blaze',
        candidate: {
          id: 'candidate_0000',
          path: '/runs/phase0_blaze_20251201_140949/candidate_0000',
          files: [
            'phase0_meta.json',
            'phase1_band_data.h5', 'phase1_field_stats.json',
            'phase2_blaze_data.h5', 'phase2_operator_meta.json', 'phase2_R_grid.npy', 'phase2_report.md',
            'phase3_eigenstates.h5', 'phase3_eigenvalues.csv', 'phase3_report.md',
          ],
          phaseStatus: {},
        },
      },
      {
        name: 'phase0_real_run_20251113_153358',
        path: '/runs/phase0_real_run_20251113_153358',
        timestamp: '2025-11-13 15:33:58',
        pipelineType: 'mpb',
        candidate: {
          id: 'candidate_3431',
          path: '/runs/phase0_real_run_20251113_153358/candidate_3431',
          files: ['phase0_meta.json'],
          phaseStatus: {},
        },
      },
    ],
  }
}

// Process manifest to compute phase status for all candidates
export function processManifest(manifest: RunsManifest): RunsManifest {
  return {
    ...manifest,
    runs: manifest.runs.map(run => ({
      ...run,
      candidate: {
        ...run.candidate,
        phaseStatus: computePhaseStatus(run.candidate.files),
      },
    })),
  }
}
