const EPS_BG_RANGE = { min: 1.8, max: 14.0, step: 0.1, precision: 2 }
const R_OVER_A_RANGE = { min: 0.1, max: 0.48, step: 0.01, precision: 3 }

function buildAxis({ min, max, step, precision }: { min: number; max: number; step: number; precision: number }): number[] {
  const values: number[] = []
  const count = Math.floor((max - min) / step + 0.5)
  for (let idx = 0; idx <= count; idx += 1) {
    const value = min + idx * step
    if (value > max + 1e-9) break
    values.push(Number(value.toFixed(precision)))
  }
  if (values[values.length - 1] !== Number(max.toFixed(precision))) {
    values.push(Number(max.toFixed(precision)))
  }
  return values
}

export const DEFAULT_EPS_BG_AXIS = buildAxis(EPS_BG_RANGE)
export const DEFAULT_R_OVER_A_AXIS = buildAxis(R_OVER_A_RANGE)

export const AXES_METADATA = {
  epsilonBackground: EPS_BG_RANGE,
  rOverA: R_OVER_A_RANGE,
}
