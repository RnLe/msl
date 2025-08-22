'use client';

import * as React from 'react';
import { useEffect, useMemo, useState } from 'react';
import { getWasmModule } from '../providers/wasmLoader';
import type { WasmMoire2D, WasmLattice2D } from '../../public/wasm/moire_lattice_wasm';

type LatticeChoice = 'square' | 'hexagonal' | 'rectangular' | 'oblique';

interface MoireVectors {
  a: { x: number; y: number };
  b: { x: number; y: number };
}

type RegistryCenterJS = {
  label: string;
  tau: { x: number; y: number };
  position: { x: number; y: number };
};

// Add a TS shape for the new WASM result that bundles L and centers
interface RegistryCentersResultJS {
  L: number[]; // flattened row-major [l00, l01, l10, l11]
  centers: RegistryCenterJS[];
}

interface DebugLine {
  key: string;
  value: string;
}

export default function StackingRegistriesDemo() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wasm, setWasm] = useState<any>(null);

  const [latticeType, setLatticeType] = useState<LatticeChoice>('hexagonal');
  const [angleDeg, setAngleDeg] = useState<number>(5);
  const [d0x, setD0x] = useState<number>(0);
  const [d0y, setD0y] = useState<number>(0);

  const [moire, setMoire] = useState<WasmMoire2D | null>(null);
  const [base, setBase] = useState<WasmLattice2D | null>(null);
  const [moireVectors, setMoireVectors] = useState<MoireVectors | null>(null);

  const [centers, setCenters] = useState<RegistryCenterJS[]>([]);
  const [debug, setDebug] = useState<DebugLine[]>([]);
  // Keep a Cartesian anchor for the 'top' center to ensure continuity across angle changes
  const topAnchorPosRef = React.useRef<{ x: number; y: number } | null>(null);
  const lastFracRef = React.useRef<Record<string, { u: number; v: number }>>({});

  // Reset continuity anchor when lattice type changes fundamentally
  useEffect(() => {
    topAnchorPosRef.current = null;
  }, [latticeType]);

  useEffect(() => {
    (async () => {
      try {
        const m = await getWasmModule();
        setWasm(m);
        setLoading(false);
      } catch (e: any) {
        setError(e?.message || String(e));
        setLoading(false);
      }
    })();
  }, []);

  // Build lattices and compute centers whenever inputs change
  useEffect(() => {
    if (!wasm) return;
    try {
      // Base lattice
      const lattice: WasmLattice2D =
        latticeType === 'square'
          ? wasm.create_square_lattice(1)
          : latticeType === 'hexagonal'
            ? wasm.create_hexagonal_lattice(1)
            : latticeType === 'rectangular'
              ? wasm.create_rectangular_lattice(1.5, 1)
              : wasm.create_oblique_lattice(
                  1.0,                   // a
                  Math.SQRT2,            // b (irrational)
                  73.2                   // gamma in degrees (irregular angle)
                );
      setBase(lattice);

      // Simple twisted bilayer
      const moireL: WasmMoire2D = wasm.create_twisted_bilayer(lattice, angleDeg);
      setMoire(moireL);

      // Prefer API that returns both centers and the correct wrapping basis L
      let centersArr: RegistryCenterJS[] = [];
      let vectors: MoireVectors | null = null;

      if (wasm.compute_registry_centers_monatomic_with_l) {
        // New path: get L and centers together
        const res = wasm.compute_registry_centers_monatomic_with_l(
          moireL,
          d0x,
          d0y
        ) as RegistryCentersResultJS;
        centersArr = res?.centers ?? [];
        const L = res?.L ?? [];
        if (L.length === 4) {
          // L is row-major: [l00, l01, l10, l11]
          vectors = {
            a: { x: L[0], y: L[2] },
            b: { x: L[1], y: L[3] },
          };
        }
      }

      if (!vectors) {
        // Fallbacks for older builds
        if (wasm.compute_registry_centers_monatomic_unwrapped) {
          centersArr = wasm.compute_registry_centers_monatomic_unwrapped(
            moireL,
            d0x,
            d0y
          ) as RegistryCenterJS[];
        } else {
          centersArr = wasm.compute_registry_centers_monatomic(
            moireL,
            d0x,
            d0y
          ) as RegistryCenterJS[];
        }
        // Use moiré object's primitive vectors as a fallback basis
        const pv = moireL.primitive_vectors() as MoireVectors;
        vectors = pv;
      }

      setMoireVectors(vectors);
      setCenters(centersArr || []);

      // Update debug info
      const dbg: DebugLine[] = [];
      const { a: va, b: vb } = vectors!;
      const len = (v: { x: number; y: number }) => Math.hypot(v.x, v.y).toFixed(6);
      dbg.push({ key: 'moire.a', value: `(${va.x.toFixed(6)}, ${va.y.toFixed(6)}) |L|=${len(va)}` });
      dbg.push({ key: 'moire.b', value: `(${vb.x.toFixed(6)}, ${vb.y.toFixed(6)}) |L|=${len(vb)}` });
      dbg.push({ key: 'twist_deg', value: angleDeg.toFixed(6) });
      dbg.push({ key: 'd0', value: `(${d0x.toFixed(6)}, ${d0y.toFixed(6)})` });
      dbg.push({ key: 'centers.count', value: String(centersArr?.length ?? 0) });

      // Fractional conversion using the basis vectors (prefer L if available)
      const a11 = va.x, a21 = va.y;
      const a12 = vb.x, a22 = vb.y;
      const detRaw = a11 * a22 - a21 * a12;
      const det = Math.abs(detRaw) < 1e-12 ? (detRaw >= 0 ? 1e-12 : -1e-12) : detRaw;
      const toFrac = (x: number, y: number) => ({
        u: (a22 * x - a12 * y) / det,
        v: (-a21 * x + a11 * y) / det,
      });

      centersArr.forEach((c) => {
        const uv = toFrac(c.position.x, c.position.y);
        dbg.push({ key: `center.${c.label}.cart`, value: `(${c.position.x.toFixed(6)}, ${c.position.y.toFixed(6)})` });
        dbg.push({ key: `center.${c.label}.frac`, value: `(${uv.u.toFixed(6)}, ${uv.v.toFixed(6)})` });
      });

      setDebug(dbg);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || String(e));
    }
  }, [wasm, latticeType, angleDeg, d0x, d0y]);

  // Get sublattice and moiré lattice points
  const latticePoints = useMemo(() => {
    if (!base || !moire || !moireVectors) return { lattice1: [], lattice2: [], moireLattice: [] };
    
    try {
      // Use the moiré primitive basis (prefer L) to build a tiling and drive extents
      const { a, b } = moireVectors;
      const v1 = a; const v2 = b;

      // Characteristic moiré size and a conservative search window for initial tiling
      const moireSize = Math.max(
        Math.hypot(v1.x, v1.y),
        Math.hypot(v2.x, v2.y)
      );
      const searchSize = moireSize * 3.0; // base window; moiré points already fill canvas well

      // Generate moiré lattice points in a square region of fractional indices
      const moireLatticePoints: Array<{ x: number; y: number }> = [];
      const K = 6; // produces (2K+1)^2 points; cheap and sufficient
      for (let i = -K; i <= K; i++) {
        for (let j = -K; j <= K; j++) {
          const x = i * v1.x + j * v2.x;
          const y = i * v1.y + j * v2.y;
          if (Math.abs(x) <= searchSize && Math.abs(y) <= searchSize) {
            moireLatticePoints.push({ x, y });
          }
        }
      }

      // Derive a rectangle that matches the moiré tiling extents and slightly enlarge it
      const mx = moireLatticePoints.length ? moireLatticePoints.map(p => p.x) : [0];
      const my = moireLatticePoints.length ? moireLatticePoints.map(p => p.y) : [0];
      const moireMinX = Math.min(...mx);
      const moireMaxX = Math.max(...mx);
      const moireMinY = Math.min(...my);
      const moireMaxY = Math.max(...my);
      const rectWidth  = (moireMaxX - moireMinX) * 1.1 + 1e-6;
      const rectHeight = (moireMaxY - moireMinY) * 1.1 + 1e-6;

      // Fetch base-lattice points over the same visual extents so they fill the space, too
      const lattice1 = moire.lattice_1();
      const lattice2 = moire.lattice_2();
      const lattice1Points = lattice1.get_direct_lattice_points_in_rectangle(rectWidth, rectHeight);
      const lattice2Points = lattice2.get_direct_lattice_points_in_rectangle(rectWidth, rectHeight);

      return {
        lattice1: lattice1Points || [],
        lattice2: lattice2Points || [],
        moireLattice: moireLatticePoints,
      };
    } catch (e) {
      console.error('Error getting lattice points:', e);
      return { lattice1: [], lattice2: [], moireLattice: [] };
    }
  }, [base, moire, moireVectors]);

  // Prepare geometry for drawing
  const geom = useMemo(() => {
    if (!moireVectors) return null;
    const { a, b } = moireVectors;
    const v1 = { x: a.x, y: a.y };
    const v2 = { x: b.x, y: b.y };

    // Generate multiple moiré cells (3x3 grid)
    const cells = [] as Array<{ pts: {x:number;y:number}[]; isCenter: boolean }>;
    for (let i = -1; i <= 1; i++) {
      for (let j = -1; j <= 1; j++) {
        const offset = { x: i * v1.x + j * v2.x, y: i * v1.y + j * v2.y };
        const pts = [
          { x: offset.x, y: offset.y },
          { x: offset.x + v1.x, y: offset.y + v1.y },
          { x: offset.x + v1.x + v2.x, y: offset.y + v1.y + v2.y },
          { x: offset.x + v2.x, y: offset.y + v2.y },
        ];
        cells.push({ pts, isCenter: i === 0 && j === 0 });
      }
    }

    // Include lattice and moiré points to compute tight bounds
    const cellPts = cells.flatMap(c => c.pts);
    const moirePts = latticePoints.moireLattice || [];
    const l1Pts = latticePoints.lattice1 || [];
    const l2Pts = latticePoints.lattice2 || [];
    const allPts = [...cellPts, ...moirePts, ...l1Pts, ...l2Pts];

    const minX = Math.min(...allPts.map(p => p.x));
    const maxX = Math.max(...allPts.map(p => p.x));
    const minY = Math.min(...allPts.map(p => p.y));
    const maxY = Math.max(...allPts.map(p => p.y));

    const width = Math.max(maxX - minX, 1e-9);
    const height = Math.max(maxY - minY, 1e-9);

    return { v1, v2, cells, bounds: { minX, maxX, minY, maxY, width, height } };
  }, [moireVectors, latticePoints]);

  const svg = useMemo(() => {
    if (!geom) return null;
    // Fixed logical viewport; SVG scales responsively to fill container
    const W = 1600;
    const H = 800;
    const pad = 20; // minimal padding to maximize content area

    // Start from data bounds then expand to match canvas aspect to avoid distortion
    const availW = W - 2 * pad;
    const availH = H - 2 * pad;
    const targetAspect = availW / availH;

    let { minX, maxX, minY, maxY, width, height } = geom.bounds;
    
    // Center the bounds around origin first
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Apply a zoom factor to scale up the content (1.5 = 50% larger)
    const zoomFactor = 0.67; // Inverse zoom: smaller value = bigger display
    width = width * zoomFactor;
    height = height * zoomFactor;
    
    // For a square display area, use the larger dimension
    const maxDim = Math.max(width, height);
    
    // Make it square but maintain aspect ratio for display
    if (targetAspect > 1) {
      // Wider than tall - expand width to fill horizontal space
      width = maxDim * Math.sqrt(targetAspect);
      height = maxDim / Math.sqrt(targetAspect);
    } else {
      // Taller than wide or square
      width = maxDim;
      height = maxDim;
    }
    
    // Recalculate bounds based on new dimensions
    minX = centerX - width / 2;
    maxX = centerX + width / 2;
    minY = centerY - height / 2;
    maxY = centerY + height / 2;

    // Uniform scale preserves angles/shape
    const s = Math.min(availW / width, availH / height);
    
    // Scale factor for point sizes (proportional to zoom)
    const pointScale = 1.5; // 50% larger points

    const toScreen = (x: number, y: number) => {
      // Translate back from centered origin
      const u = W / 2 + (x - centerX) * s;
      const v = H / 2 - (y - centerY) * s; // Flip Y so up is positive
      return { u, v };
    };

    // Precompute fractional/Cartesian transforms for chosen basis
    const a11 = geom.v1.x, a21 = geom.v1.y;
    const a12 = geom.v2.x, a22 = geom.v2.y;
    let det = a11 * a22 - a21 * a12;
    if (Math.abs(det) < 1e-12) {
      det = (det >= 0 ? 1 : -1) * 1e-12; // avoid numerical blow-up
    }
    const toFrac = (x: number, y: number) => ({
      u: (a22 * x - a12 * y) / det,
      v: (-a21 * x + a11 * y) / det,
    });
    const toCart = (u: number, v: number) => ({
      x: u * a11 + v * a12,
      y: u * a21 + v * a22,
    });

    // Draw moiré cells (keep a 3x3 grid as requested)
    const cellPolygons = geom.cells.map((cell, idx) => {
      const poly = cell.pts.map(p => {
        const { u, v } = toScreen(p.x, p.y);
        return `${u},${v}`;
      }).join(' ');
      
      return (
        <polygon
          key={`cell-${idx}`}
          points={poly}
          fill={cell.isCenter ? '#7c3aed15' : 'none'}
          stroke="#7c3aed"
          strokeWidth={cell.isCenter ? 3 : 1.5} // scaled stroke width
          strokeDasharray={cell.isCenter ? undefined : '4.5,4.5'} // scaled dash
          opacity={cell.isCenter ? 1 : 0.3}
        />
      );
    });

    const margin = 50;

    // Draw sublattice points
    const lattice1Dots = latticePoints.lattice1
      .map((p: any, i: number) => {
        const { u, v } = toScreen(p.x || 0, p.y || 0);
        if (u < -margin || u > W + margin || v < -margin || v > H + margin) return null;
        return (
          <circle
            key={`l1-${i}`}
            cx={u}
            cy={v}
            r={3.75 * pointScale / 1.5} // scaled radius
            fill="#4682B4"
            opacity={0.6}
            stroke="white"
            strokeWidth={0.75} // scaled stroke
          />
        );
      })
      .filter(Boolean);

    const lattice2Dots = latticePoints.lattice2
      .map((p: any, i: number) => {
        const { u, v } = toScreen(p.x || 0, p.y || 0);
        if (u < -margin || u > W + margin || v < -margin || v > H + margin) return null;
        return (
          <circle
            key={`l2-${i}`}
            cx={u}
            cy={v}
            r={3.75 * pointScale / 1.5} // scaled radius
            fill="#FF8C00"
            opacity={0.6}
            stroke="white"
            strokeWidth={0.75} // scaled stroke
          />
        );
      })
      .filter(Boolean);

    // Draw moiré lattice points (red)
    const moireDots = latticePoints.moireLattice
      .map((p: any, i: number) => {
        const { u, v } = toScreen(p.x || 0, p.y || 0);
        if (u < -margin || u > W + margin || v < -margin || v > H + margin) return null;
        return (
          <circle
            key={`moire-${i}`}
            cx={u}
            cy={v}
            r={4.5 * pointScale / 1.5} // scaled radius
            fill="#ef4444"
            opacity={0.9}
            stroke="white"
            strokeWidth={1.5} // scaled stroke
          />
        );
      })
      .filter(Boolean);

    // Draw registry centers aligned to the unit cell grid (2x2 around center)
    const centerDots = (() => {
      if (!centers || centers.length === 0) return null;

      // Only draw registry points on a 2x2 unit-cell grid around the center
      const iMin = -1, iMax = 0;
      const jMin = -1, jMax = 0;

      const tiles = centers.flatMap((c: RegistryCenterJS, idx: number) => {
        // Fractional coords in moiré basis
        const uv = toFrac(c.position.x, c.position.y);
        // Wrap into [0,1) to align with the drawn unit cell anchored at the origin
        const u0 = uv.u - Math.floor(uv.u);
        const v0 = uv.v - Math.floor(uv.v);

        const color = c.label.startsWith('top') ? '#2b6cb0' :
                      c.label.startsWith('bridge') ? '#d97706' : '#4c1d95';

        const nodes: React.ReactNode[] = [];
        for (let ii = iMin; ii <= iMax; ii++) {
          for (let jj = jMin; jj <= jMax; jj++) {
            const P = toCart(u0 + ii, v0 + jj);
            const { u, v } = toScreen(P.x, P.y);
            if (u < -margin || u > W + margin || v < -margin || v > H + margin) continue;
            const isCenterTile = ii === 0 && jj === 0;
            nodes.push(
              <g key={`ctr-${idx}-${ii}-${jj}`}>
                <circle 
                  cx={u} 
                  cy={v} 
                  r={isCenterTile ? 9 : 6} // scaled radius
                  fill={color} 
                  opacity={isCenterTile ? 0.9 : 0.7} 
                  stroke="white" 
                  strokeWidth={2.25} // scaled stroke
                />
                {isCenterTile && (
                  <text x={u + 15} y={v - 12} fontSize={16.5} fill="#374151" fontWeight="500">{c.label}</text>
                )}
              </g>
            );
          }
        }
        return nodes;
      });

      return tiles;
    })();

    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', height: 'auto', background: 'transparent', border: '1px solid var(--tw-prose-hr)', boxSizing: 'border-box', display: 'block', margin: '0 auto' }}
      >
        {/* Draw sub-lattice first */}
        <g opacity={0.8}>
          {lattice1Dots}
          {lattice2Dots}
        </g>
        {/* Keep the 3x3 moiré unit-cell grid */}
        {cellPolygons}
        {/* Moiré lattice and registry centers above */}
        {moireDots}
        <circle cx={toScreen(0,0).u} cy={toScreen(0,0).v} r={4.5} fill="#111827" />
        {centerDots}
      </svg>
    );
  }, [geom, centers, latticePoints]);

  if (loading) return <div className="text-sm text-gray-500">Loading WASM…</div>;
  if (error) return <div className="text-sm text-red-600">{error}</div>;

  return (
    <div className="w-full max-w-none space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Lattice</label>
          <select value={latticeType} onChange={e => setLatticeType(e.target.value as LatticeChoice)}
                  className="px-2 py-1 border rounded bg-transparent">
            <option value="hexagonal">Triangular (hexagonal)</option>
            <option value="square">Square</option>
            <option value="rectangular">Rectangular (1.5:1)</option>
            <option value="oblique">Oblique (irregular)</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">Twist (deg)</label>
          <input type="range" min={5} max={30} step={0.5} value={angleDeg}
                 onChange={e => setAngleDeg(parseFloat(e.target.value))} className="w-56" />
          <div className="text-xs text-gray-700 mt-1">{angleDeg.toFixed(2)}°</div>
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">d₀ x</label>
          <input type="number" step="0.05" value={d0x} onChange={e => setD0x(parseFloat(e.target.value || '0'))}
                 className="px-2 py-1 border rounded w-24 bg-transparent" />
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">d₀ y</label>
          <input type="number" step="0.05" value={d0y} onChange={e => setD0y(parseFloat(e.target.value || '0'))}
                 className="px-2 py-1 border rounded w-24 bg-transparent" />
        </div>
      </div>

      {/* Viz */}
      <div className="w-full overflow-x-hidden">
        {svg}
      </div>

      {/* Debug panel */}
      <div className="text-xs font-mono whitespace-pre-wrap bg-gray-50 border rounded p-3 text-gray-800">
        {debug.map((d) => (
          <div key={d.key}><span className="text-gray-500">{d.key}:</span> {d.value}</div>
        ))}
      </div>

      {/* Legend */}
      <div className="text-xs text-gray-700">
        <div className="flex items-center gap-4 flex-wrap">
          <span className="inline-flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{background:'#2b6cb0'}}></span> top</span>
          <span className="inline-flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{background:'#d97706'}}></span> bridge</span>
          <span className="inline-flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{background:'#4c1d95'}}></span> hollow</span>
          <span className="inline-flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{background:'#10b981'}}></span> moiré lattice</span>
          <span className="ml-4 text-gray-500">Dashed lines: neighboring moiré cells</span>
        </div>
      </div>
    </div>
  );
}
