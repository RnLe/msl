'use client';

import React, { useCallback, useMemo, useState } from 'react';

const SCORE_TERMS = [
  { weightKey: 'w_flat', scoreKey: 'S_flat', label: 'Curvature (S_flat)', color: '#2563eb' },
  { weightKey: 'w_gap', scoreKey: 'S_gap', label: 'Spectral Gap (S_gap)', color: '#0ea5e9' },
  { weightKey: 'w_parab', scoreKey: 'S_parab', label: 'Parabolic Region (S_parab)', color: '#10b981' },
  { weightKey: 'w_vg', scoreKey: 'S_vg', label: 'Parabola Span (S_vg)', color: '#f97316' },
] as const;
const CORE_WEIGHT_KEYS: Array<'w_flat' | 'w_parab' | 'w_vg'> = ['w_flat', 'w_parab', 'w_vg'];
const TERM_BY_WEIGHT = SCORE_TERMS.reduce<Record<WeightKey, (typeof SCORE_TERMS)[number]>>((acc, term) => {
  acc[term.weightKey] = term;
  return acc;
}, {} as Record<WeightKey, (typeof SCORE_TERMS)[number]>);

type WeightKey = (typeof SCORE_TERMS)[number]['weightKey'];
type RangeTuple = [number, number];
type PolyPoint = { x: number; y: number };
type SampledCurve = { points: PolyPoint[]; peak: PolyPoint; extrema: PolyPoint[] };
type ChartDimensions = { width: number; height: number; margin?: number };

type MetricInputs = {
  curvature: number;
  gap: number;
  symmetryError: number;
  saddleSeverity: number;
  edgeSpread: number;
};

type MiniPlot = {
  id: string;
  curvature: number;
  slope: number;
  asymmetry: number;
  points: PolyPoint[];
  peak: PolyPoint;
  extrema: PolyPoint[];
  yRange: RangeTuple;
  metrics: MetricInputs;
  scores: Record<string, number>;
  contributions: number[];
  total: number;
  distribution: Record<'w_flat' | 'w_parab' | 'w_vg', number>;
};

const GRID_ROWS = 4;
const GRID_COLS = 5;
const DOMAIN: RangeTuple = [-1.8, 1.8];
const SAMPLE_STEPS = 160;
const DEFAULT_CONSTANTS = {
  kappa0: 1.0,
  Delta0: 0.1,
  v0: 1.0,
  epsBgMax: 12.0,
  alphaParab: 0.3,
  betaParab: 1.5,
};
const DEFAULT_CURVATURE_RANGE: RangeTuple = [0.45, 1.9];
const DEFAULT_SLOPE_RANGE: RangeTuple = [0.0, 1.3];
const HERO_DIMENSIONS: ChartDimensions = { width: 560, height: 240, margin: 32 };
const INITIAL_WEIGHTS: Record<WeightKey, number> = {
  w_flat: 0.6,
  w_gap: 0.35,
  w_parab: 0.25,
  w_vg: 0.15,
};
const WEIGHT_RANGE: RangeTuple = [-1.5, 1.5];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeToRange(value: number, range: RangeTuple) {
  const [min, max] = range;
  if (!Number.isFinite(value)) return 0;
  const span = Math.max(1e-6, max - min);
  return clamp((value - min) / span, 0, 1);
}

function lerp(a: number, b: number, t: number) {
  if (!Number.isFinite(t)) return a;
  return a + (b - a) * t;
}

const PARAB_SYMMETRY_SCALE = 1.5;
const EDGE_SPREAD_SCALE = 2.5;

function computeScores(metrics: MetricInputs) {
  const { curvature, gap, symmetryError, saddleSeverity, edgeSpread } = metrics;
  const { kappa0, Delta0 } = DEFAULT_CONSTANTS;
  const S_flat = clamp(Math.abs(curvature) / Math.max(kappa0, 1e-6), 0, 1.2);
  const S_gap = clamp(gap / (gap + Delta0), 0, 1);
  const saddlePenalty = saddleSeverity > 0 ? 1 : 0;
  const symmetryPenalty = clamp(symmetryError / PARAB_SYMMETRY_SCALE, 0, 1);
  const S_parab = clamp(1 - (0.6 * symmetryPenalty + 0.4 * saddlePenalty), 0, 1);
  const S_vg = clamp(1 - edgeSpread / EDGE_SPREAD_SCALE, 0, 1);
  return { S_flat, S_gap, S_parab, S_vg };
}

function quarticValue(x: number, curvature: number, slope: number, asymmetry: number) {
  const a = Math.max(0.25, curvature * 0.4);
  const b = -Math.abs(curvature) * (0.65 + 0.2 * (1 - Math.abs(asymmetry)));
  const c = slope * 0.85 + asymmetry * 0.35;
  const d = asymmetry * 0.08;
  return a * x ** 4 + b * x ** 2 + c * x + d;
}

function findLocalExtrema(points: PolyPoint[]) {
  const extrema: PolyPoint[] = [];
  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const next = points[i + 1];
    const isMax = curr.y >= prev.y && curr.y >= next.y && (curr.y > prev.y || curr.y > next.y);
    const isMin = curr.y <= prev.y && curr.y <= next.y && (curr.y < prev.y || curr.y < next.y);
    if (isMax || isMin) extrema.push(curr);
  }
  if (!extrema.length && points.length) {
    extrema.push(points.reduce((acc, candidate) => (candidate.y > acc.y ? candidate : acc), points[0]));
  }
  return extrema;
}

function sampleQuartic(curvature: number, slope: number, asymmetry: number): SampledCurve {
  const [xmin, xmax] = DOMAIN;
  const pts: PolyPoint[] = [];
  for (let i = 0; i <= SAMPLE_STEPS; i++) {
    const t = i / SAMPLE_STEPS;
    const x = lerp(xmin, xmax, t);
    const y = quarticValue(x, curvature, slope, asymmetry);
    pts.push({ x, y });
  }
  const extrema = findLocalExtrema(pts);
  const peak = extrema.reduce((acc, point) => (point.y > acc.y ? point : acc), extrema[0]);
  return { points: pts, extrema, peak };
}

function computeDistribution(curvature: number, slope: number) {
  const flatBias = curvature + 0.35;
  const vgBias = Math.abs(slope) + 0.2;
  const parabBias = Math.max(0.2, curvature * 0.45 + (1 - Math.min(1, Math.abs(slope)))) + 0.2;
  const sum = flatBias + vgBias + parabBias;
  return {
    w_flat: flatBias / sum,
    w_parab: parabBias / sum,
    w_vg: vgBias / sum,
  };
}

function weightSnippet(weights: Record<WeightKey, number>) {
  const constants = DEFAULT_CONSTANTS;
  const parts = [
    `kappa_0: ${constants.kappa0.toFixed(3)}`,
    `Delta_0: ${constants.Delta0.toFixed(3)}`,
    `v_0: ${constants.v0.toFixed(3)}`,
    `eps_bg_max: ${constants.epsBgMax.toFixed(3)}`,
    `alpha_parab: ${constants.alphaParab.toFixed(3)}`,
    `beta_parab: ${constants.betaParab.toFixed(3)}`,
    ...SCORE_TERMS.map((term) => `${term.weightKey}: ${weights[term.weightKey].toFixed(3)}`),
  ];
  return parts.join('\n');
}

function buildMiniPlotFromRatios(
  rowRatio: number,
  colRatio: number,
  opts: {
    curvatureRange: RangeTuple;
    slopeRange: RangeTuple;
    gapMagnitude: number;
    weights: Record<WeightKey, number>;
  },
  id: string,
): MiniPlot {
  const curvature = lerp(opts.curvatureRange[0], opts.curvatureRange[1], colRatio);
  const slope = lerp(opts.slopeRange[0], opts.slopeRange[1], rowRatio);
  const asymmetry = (rowRatio - 0.5) * 1.3 + (colRatio - 0.5) * 0.7;
  const { points, extrema, peak } = sampleQuartic(curvature, slope, asymmetry);
  const cellPeakX = peak.x;
  const yCenter = peak.y;
  const clampX = (value: number) => clamp(value, DOMAIN[0], DOMAIN[1]);
  const evaluateAt = (offset: number) => quarticValue(clampX(cellPeakX + offset), curvature, slope, asymmetry);
  const nearOffset = 0.35;
  const farOffset = 0.9;
  const yNearLeft = evaluateAt(-nearOffset);
  const yNearRight = evaluateAt(nearOffset);
  const symmetryError = Math.abs(yNearLeft - yNearRight) / (Math.abs(yCenter) + 1);
  const saddleSeverity = (yNearLeft - yCenter) * (yNearRight - yCenter) < 0 ? 1 : 0;
  const yFarLeft = evaluateAt(-farOffset);
  const yFarRight = evaluateAt(farOffset);
  const edgeSpread = Math.max(Math.abs(yFarLeft - yCenter), Math.abs(yFarRight - yCenter));
  const yCandidates = points.map((pt) => pt.y).concat([peak.y + opts.gapMagnitude, peak.y - opts.gapMagnitude]);
  const minY = Math.min(...yCandidates);
  const maxY = Math.max(...yCandidates);
  const span = Math.max(1e-3, maxY - minY);
  const yRange: RangeTuple = [minY, minY + span];
  const metrics: MetricInputs = {
    curvature,
    gap: opts.gapMagnitude,
    symmetryError,
    saddleSeverity,
    edgeSpread,
  };
  const scores = computeScores(metrics);
  const contributions = SCORE_TERMS.map((term) => opts.weights[term.weightKey] * scores[term.scoreKey]);
  const total = contributions.reduce((acc, value) => acc + value, 0);
  const distribution = computeDistribution(curvature, Math.abs(slope));
  return {
    id,
    curvature,
    slope,
    points,
    peak,
    extrema,
    yRange,
    metrics,
    scores,
    contributions,
    total,
    distribution,
    asymmetry,
  };
}

function RangeSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: RangeTuple;
  min: number;
  max: number;
  step: number;
  onChange: (next: RangeTuple) => void;
}) {
  const [start, end] = value;
  const handleStart = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = Math.min(parseFloat(event.target.value), end - step);
    onChange([clamp(next, min, max), end]);
  };
  const handleEnd = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = Math.max(parseFloat(event.target.value), start + step);
    onChange([start, clamp(next, min, max)]);
  };
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs font-medium text-slate-600 dark:text-slate-300">
        <span>{label}</span>
        <span>
          {start.toFixed(2)} – {end.toFixed(2)}
        </span>
      </div>
      <div className="flex flex-col gap-1">
        <input type="range" min={min} max={max} step={step} value={start} onChange={handleStart} />
        <input type="range" min={min} max={max} step={step} value={end} onChange={handleEnd} />
      </div>
    </div>
  );
}

function ScoreBar({
  contributions,
  total,
  weights,
  scores,
}: {
  contributions: number[];
  total: number;
  weights: Record<WeightKey, number>;
  scores: Record<string, number>;
}) {
  const magnitude = contributions.reduce((acc, value) => acc + Math.abs(value), 0) || 1;
  const normalized = contributions.map((value) => Math.abs(value) / magnitude);
  const clampedTotal = clamp(0.5 + 0.5 * Math.tanh(total), 0, 1);
  return (
    <div className="mt-1 flex flex-col gap-1">
      <div className="relative">
        <div className="group flex h-2 overflow-hidden rounded-full">
          {normalized.map((portion, idx) => (
            <div
              key={SCORE_TERMS[idx].weightKey}
              className="h-full"
              style={{ width: `${portion * 100}%`, backgroundColor: SCORE_TERMS[idx].color }}
            />
          ))}
        </div>
        <div className="pointer-events-none absolute bottom-full left-0 z-10 hidden min-w-[240px] translate-y-[-6px] rounded-xl bg-slate-900/95 p-3 text-xs text-white shadow-lg group-hover:block">
          <div className="space-y-1.5">
            {SCORE_TERMS.map((term, idx) => {
              const contribution = contributions[idx] ?? 0;
              const weight = weights[term.weightKey] ?? 0;
              const scoreValue = scores[term.scoreKey] ?? 0;
              return (
                <div key={`${term.weightKey}-tooltip`} className="flex items-center justify-between gap-3">
                  <span className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: term.color }} />
                    {term.scoreKey}
                  </span>
                  <span className="text-right text-[11px]">
                    w={weight.toFixed(2)} · s={scoreValue.toFixed(2)} · c={contribution.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-right text-[11px] opacity-80">total={clampedTotal.toFixed(2)}</p>
        </div>
      </div>
      <div className="relative h-1.5 rounded-full bg-white/20">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-white"
          style={{ width: `${clampedTotal * 100}%` }}
          aria-label={`Score ${clampedTotal.toFixed(2)}`}
        />
      </div>
    </div>
  );
}

function MiniPlotSvg({
  cell,
  gapValue,
  selected,
  dimensions,
}: {
  cell: MiniPlot;
  gapValue: number;
  selected: boolean;
  dimensions?: ChartDimensions;
}) {
  const { width, height, margin } = {
    width: 190,
    height: 120,
    margin: 8,
    ...dimensions,
  };
  const yRange = cell.yRange ?? [0, 1];
  const normalizedPoints = cell.points.map((p) => ({ x: p.x, y: normalizeToRange(p.y, yRange) }));
  const normalizedExtrema = cell.extrema.map((ext) => ({ x: ext.x, y: normalizeToRange(ext.y, yRange) }));
  const normalizedPeak = normalizeToRange(cell.peak.y, yRange);
  const gapUpper = normalizeToRange(cell.peak.y + gapValue, yRange);
  const gapLower = normalizeToRange(cell.peak.y - gapValue, yRange);
  const projectX = (x: number) => {
    const [xmin, xmax] = DOMAIN;
    return margin + ((x - xmin) / (xmax - xmin)) * (width - 2 * margin);
  };
  const projectY = (normalized: number) => height - margin - normalized * (height - 2 * margin);
  const path = normalizedPoints
    .map((pt, idx) => `${idx === 0 ? 'M' : 'L'} ${projectX(pt.x).toFixed(2)} ${projectY(pt.y).toFixed(2)}`)
    .join(' ');
  const scoreLabel = cell.total.toFixed(2);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Quartic mini plot">
      <rect
        x={margin}
        y={margin}
        width={width - 2 * margin}
        height={height - 2 * margin}
        rx={12}
        fill={selected ? 'rgba(37,99,235,0.08)' : 'rgba(148,163,184,0.15)'}
        stroke={selected ? '#2563eb' : 'rgba(148,163,184,0.4)'}
        strokeWidth={1.2}
      />
      <path d={path} fill="none" stroke="white" strokeWidth={2.1} />
      {normalizedExtrema.map((ext, idx) => {
        const isPeak = Math.abs(ext.y - normalizedPeak) < 1e-3 || ext.y > normalizedPeak - 1e-3;
        const fill = isPeak ? '#fbbf24' : '#0ea5e9';
        const stroke = isPeak ? '#f59e0b' : '#0284c7';
        return (
          <circle
            key={`ext-${cell.id}-${idx}`}
            cx={projectX(ext.x)}
            cy={projectY(ext.y)}
            r={isPeak ? 5 : 4}
            fill={fill}
            stroke={stroke}
            strokeWidth={isPeak ? 1.4 : 1}
          />
        );
      })}
      <circle cx={projectX(cell.peak.x)} cy={projectY(gapUpper)} r={4} fill="#0ea5e9" opacity={0.85} />
      <circle cx={projectX(cell.peak.x)} cy={projectY(gapLower)} r={4} fill="#0ea5e9" opacity={0.85} />
      <text
        x={width / 2}
        y={height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        className="font-semibold"
        fill="rgba(255,255,255,0.8)"
      >
        {scoreLabel}
      </text>
    </svg>
  );
}

function HeroPlot({ cell, gapValue }: { cell: MiniPlot; gapValue: number }) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5 shadow-inner dark:border-slate-800 dark:bg-slate-900/40">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Reference quartic</p>
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            κ={cell.curvature.toFixed(2)} &nbsp; | &nbsp; |dω/dk|={Math.abs(cell.slope).toFixed(2)}
          </h3>
        </div>
        <div className="rounded-full bg-white/70 px-4 py-1 text-xs font-semibold text-slate-700 shadow dark:bg-slate-800 dark:text-slate-200">
          Gap markers: ±{gapValue.toFixed(2)}
        </div>
      </div>
      <div className="mt-4">
        <MiniPlotSvg cell={cell} gapValue={gapValue} selected dimensions={HERO_DIMENSIONS} />
      </div>
      <p className="mt-3 text-xs text-slate-600 dark:text-slate-400">
        Always-visible preview of the x⁴ band wall before opening the explorer.
      </p>
    </div>
  );
}

export default function Phase0ScoringExplorer() {
  const [isOpen, setIsOpen] = useState(false);
  const [weights, setWeights] = useState<Record<WeightKey, number>>(INITIAL_WEIGHTS);
  const [gapMagnitude, setGapMagnitude] = useState(0.12);
  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [curvatureRange, setCurvatureRange] = useState<RangeTuple>(DEFAULT_CURVATURE_RANGE);
  const [slopeRange, setSlopeRange] = useState<RangeTuple>(DEFAULT_SLOPE_RANGE);
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'error'>('idle');

  const updateWeight = useCallback((key: WeightKey, value: number) => {
    const next = clamp(value, WEIGHT_RANGE[0], WEIGHT_RANGE[1]);
    setWeights((prev) => ({ ...prev, [key]: parseFloat(next.toFixed(3)) }));
  }, []);

  const grid = useMemo<MiniPlot[]>(() => {
    const entries: MiniPlot[] = [];
    const rowDenom = GRID_ROWS > 1 ? GRID_ROWS - 1 : 1;
    const colDenom = GRID_COLS > 1 ? GRID_COLS - 1 : 1;
    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const rowRatio = rowDenom === 0 ? 0.5 : row / rowDenom;
        const colRatio = colDenom === 0 ? 0.5 : col / colDenom;
        entries.push(
          buildMiniPlotFromRatios(
            rowRatio,
            colRatio,
            { curvatureRange, slopeRange, gapMagnitude, weights },
            `r${row}-c${col}`,
          ),
        );
      }
    }
    return entries;
  }, [curvatureRange, slopeRange, gapMagnitude, weights]);

  const heroCell = grid[Math.max(0, Math.min(grid.length - 1, Math.floor(grid.length / 2)))] ?? null;
  const snippet = useMemo(() => weightSnippet(weights), [weights]);

  const copyToClipboard = useCallback(async (content: string) => {
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      setCopyState('error');
      setTimeout(() => setCopyState('idle'), 1800);
      return;
    }
    try {
      await navigator.clipboard.writeText(content);
      setCopyState('copied');
      setTimeout(() => setCopyState('idle'), 1800);
    } catch (error) {
      console.warn('Clipboard unavailable', error);
      setCopyState('error');
      setTimeout(() => setCopyState('idle'), 1800);
    }
  }, []);

  const handleCellSelect = (cell: MiniPlot) => {
    const coreMagnitude = CORE_WEIGHT_KEYS.reduce((acc, key) => acc + Math.abs(weights[key]), 0) || 1;
    const nextWeights: Record<WeightKey, number> = { ...weights };
    CORE_WEIGHT_KEYS.forEach((key) => {
      nextWeights[key] = parseFloat((cell.distribution[key] * coreMagnitude).toFixed(3));
    });
    setWeights(nextWeights);
    setSelectedCellId(cell.id);
    copyToClipboard(weightSnippet(nextWeights));
  };

  const statusLabel = useMemo(() => {
    if (copyState === 'copied') return 'Weights copied to clipboard';
    if (copyState === 'error') return 'Clipboard unavailable';
    return 'Tap a mini-plot to adopt and copy its preset';
  }, [copyState]);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">Phase 0 Score Explorer</p>
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-white">Modal quartic wall</h2>
        </div>
        <button
          type="button"
          onClick={() => setIsOpen(true)}
          className="inline-flex items-center justify-center rounded-full bg-slate-900 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-slate-900/10 transition hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Launch explorer
        </button>
      </div>
      {heroCell && (
        <div className="mt-6">
          <HeroPlot cell={heroCell} gapValue={gapMagnitude} />
        </div>
      )}
      {isOpen && (
        <div className="fixed inset-0 z-50 bg-black/40 p-4">
          <div className="flex h-full w-full items-center justify-center">
            <div
              className="relative h-full w-full max-w-none rounded-3xl border border-slate-200 bg-white px-6 py-5 shadow-2xl dark:border-slate-800 dark:bg-slate-900"
              style={{ height: 'min(900px, 92vh)' }}
            >
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="absolute right-4 top-4 rounded-full border border-slate-200 px-3 py-1 text-sm text-slate-600 transition hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200"
              >
                Close
              </button>
              <div className="flex h-full flex-col gap-4 pt-5">
                <header className="flex flex-col gap-1">
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Score presets & quartic walls</h3>
                  <p className="text-sm text-slate-600 dark:text-slate-300">{statusLabel}</p>
                </header>
                <div className="flex h-full min-h-0 gap-4">
                  <aside className="w-full max-w-[320px] flex-shrink-0 space-y-4 overflow-y-auto border-r border-slate-200 pr-4 dark:border-slate-800">
                    <div className="space-y-3 rounded-2xl border border-slate-200 p-3 dark:border-slate-800">
                      <label className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-slate-500">
                        <span className="flex items-center gap-2">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: TERM_BY_WEIGHT.w_gap.color }} />
                          Spectral gap weight (w_gap)
                        </span>
                        <span className="text-slate-900 dark:text-white">{weights.w_gap.toFixed(2)}</span>
                      </label>
                      <input
                        type="range"
                        min={WEIGHT_RANGE[0]}
                        max={WEIGHT_RANGE[1]}
                        step={0.01}
                        value={weights.w_gap}
                        onChange={(event) => updateWeight('w_gap', parseFloat(event.target.value))}
                      />
                      <div className="space-y-1 rounded-xl bg-slate-50 p-2 text-xs text-slate-600 dark:bg-slate-800/40 dark:text-slate-300">
                        <div className="flex items-center justify-between font-semibold uppercase tracking-wide text-[11px] text-slate-500">
                          Gap magnitude (±Δ)
                          <span className="text-slate-900 dark:text-white">{gapMagnitude.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min={0.02}
                          max={0.4}
                          step={0.005}
                          value={gapMagnitude}
                          onChange={(event) => setGapMagnitude(parseFloat(event.target.value))}
                        />
                        <p>Dots mark ±{gapMagnitude.toFixed(2)} around each main extremum.</p>
                      </div>
                    </div>
                    <div className="space-y-3 rounded-2xl border border-slate-200 p-3 dark:border-slate-800">
                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">Core term weights</div>
                      {CORE_WEIGHT_KEYS.map((key) => {
                        const term = TERM_BY_WEIGHT[key];
                        return (
                          <div key={key} className="space-y-1">
                            <label className="flex items-center justify-between text-xs font-medium text-slate-600 dark:text-slate-200">
                              <span className="flex items-center gap-2">
                                <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: term.color }} />
                                {term.label}
                              </span>
                              <span className="text-slate-900 dark:text-white">{weights[key].toFixed(2)}</span>
                            </label>
                            <input
                              type="range"
                              min={WEIGHT_RANGE[0]}
                              max={WEIGHT_RANGE[1]}
                              step={0.01}
                              value={weights[key]}
                              onChange={(event) => updateWeight(key, parseFloat(event.target.value))}
                            />
                          </div>
                        );
                      })}
                      <p className="text-[11px] text-slate-500">
                        Positive weights reward a property; negative weights penalize it.
                      </p>
                    </div>
                    <div className="rounded-2xl border border-slate-200 p-3 dark:border-slate-800">
                      <RangeSlider
                        label="Curvature axis range"
                        value={curvatureRange}
                        min={0.2}
                        max={2.3}
                        step={0.05}
                        onChange={setCurvatureRange}
                      />
                      <div className="mt-4">
                        <RangeSlider
                          label="First-derivative axis range"
                          value={slopeRange}
                          min={0}
                          max={1.8}
                          step={0.05}
                          onChange={setSlopeRange}
                        />
                      </div>
                    </div>
                    <div className="space-y-2 rounded-2xl border border-slate-200 p-3 dark:border-slate-800">
                      <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-slate-500">
                        <span>Preset weights yaml</span>
                        <button
                          type="button"
                          onClick={() => copyToClipboard(snippet)}
                          className="rounded-full border border-slate-300 px-3 py-1 text-[11px] font-semibold text-slate-600 transition hover:bg-slate-100 dark:border-slate-600 dark:text-slate-200"
                        >
                          Copy
                        </button>
                      </div>
                      <pre className="overflow-x-auto rounded-xl border border-slate-200 bg-slate-100 p-3 text-[11px] font-mono text-slate-900 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100">
                        {snippet}
                      </pre>
                      <p className="text-[11px] text-slate-500">{statusLabel}</p>
                    </div>
                  </aside>
                  <section className="flex-1 min-w-0">
                    <div className="flex h-full flex-col">
                      <div className="grid h-full min-h-0 grid-cols-2 gap-2 overflow-y-auto lg:grid-cols-4 xl:grid-cols-5">
                        {grid.map((cell) => {
                          const selected = cell.id === selectedCellId;
                          return (
                            <button
                              key={cell.id}
                              type="button"
                              onClick={() => handleCellSelect(cell)}
                              className={`group flex h-full flex-col gap-1 rounded-xl p-0 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 ${
                                selected
                                  ? 'ring-2 ring-slate-900 dark:ring-white'
                                  : 'ring-1 ring-transparent'
                              }`}
                            >
                              <MiniPlotSvg cell={cell} gapValue={gapMagnitude} selected={selected} />
                              <ScoreBar contributions={cell.contributions} total={cell.total} weights={weights} scores={cell.scores} />
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </section>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
