export { Phase1FieldsPanel, generateDemoPhase1Data } from './Phase1FieldsPanel'
export { EnvelopePipelineViewer } from './EnvelopePipelineViewer'

export { 
  RunExplorer, 
  generateDemoManifest,
  processManifest
} from './RunExplorer'

// Lattice visualization components
export { MoireLatticeCanvas } from './MoireLatticeCanvas'
export { BilayerZoomCanvas } from './BilayerZoomCanvas'
export { RegistryShiftCanvas } from './RegistryShiftCanvas'
export { TwoAtomicBasisCanvas } from './TwoAtomicBasisCanvas'
export { LatticeVisualizationPanel } from './LatticeVisualizationPanel'

// Lattice utilities
export {
  getBasisVectors,
  rotateBasis,
  computeMoireLength,
  generateLatticePoints,
  computeRegistryShift,
  cartesianToFractional,
  fractionalToCartesian,
  wrapToUnitCell,
  registryToAtomPosition,
  atomPositionsToShift,
  normalizeLatticeType,
  LATTICE_COLORS,
} from './lattice-utils'

export type { 
  RunInfo, 
  RunsManifest, 
  Phase1FieldsData, 
  Phase1FieldsMeta,
  FieldData,
  PipelineType 
} from './types'

export type {
  Vec2,
  BasisVectors,
  LatticeType,
  LatticeParams,
} from './lattice-utils'
