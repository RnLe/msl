export interface FieldData {
  values: number[][]
  min: number
  max: number
}

export interface Phase1FieldsData {
  V: FieldData
  vg_norm: FieldData
  M_inv_eig1: FieldData
  M_inv_eig2: FieldData
  extent: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }
  candidateParams?: {
    candidateId: number
    latticeType: string
    rOverA: number
    epsBg: number
    thetaDeg?: number
    moireLength?: number
    polarization?: string
    bandIndex?: number
    kLabel?: string
    omega0?: number
  }
}

/** Metadata for binary field data file */
export interface Phase1FieldsMeta {
  shape: [number, number]
  extent: {
    xMin: number
    xMax: number
    yMin: number
    yMax: number
  }
  fields: {
    V: { offset: number; min: number; max: number }
    vg_norm: { offset: number; min: number; max: number }
    M_inv_eig1: { offset: number; min: number; max: number }
    M_inv_eig2: { offset: number; min: number; max: number }
  }
  totalBytes: number
}

export interface CandidateInfo {
  id: string
  path: string
  files: string[]
  phaseStatus: Record<string, boolean>
}

export type PipelineType = 'mpb' | 'blaze'

export interface RunInfo {
  name: string
  path: string
  timestamp: string
  pipelineType: PipelineType
  candidate: CandidateInfo  // Single candidate per run (topmost)
  configFile?: string
}

export interface RunsManifest {
  generatedAt: string
  runsDir?: string
  runs: RunInfo[]
}
