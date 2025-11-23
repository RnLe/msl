export type LatticeType = 'square' | 'hex' | (string & {})

export type CoverageResponse = {
  scanId: string
  lattices: LatticeType[]
  epsBg: number[]
  rOverA: number[]
  status: Record<string, number[][]>
  updatedAt?: string
  axes?: {
    epsBg?: number[]
    rOverA?: number[]
  }
}

export type CoverageMode = 'loading' | 'ready' | 'offline'

export type CoveragePoint = {
  lattice: LatticeType
  epsIndex: number
  rIndex: number
  epsBg: number
  rOverA: number
  statusCode: number
}

export type BandDataPoint = {
  lattice: LatticeType
  pointKey: string
  bandId: string
  bandNumber: number
  polarization: string
  omega: number
  kLabel: string
  kRatio: number
  kValue: number
  kIndex: number
  color: string
}
