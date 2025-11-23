# Interactive Band Library Viewer - Summary & Implementation Guide

## 1. High-Level Architecture

### Components

- **Data store**: single HDF5 file (`band_library.h5`) that holds axes, status cubes, and band-frequency tensors.
- **Backend (Railway)**: FastAPI microservice with a persistent 5 GB volume mounted at `/data`. It exposes `GET /scans/{scan_id}/coverage`, `GET /scans/{scan_id}/band`, and `GET /health`.
- **Frontend (Next.js + Nextra)**: static MDX page that hydrates the React viewer. The UI now renders one Konva-based heatmap at a time with lattice-toggle buttons, the dark palette requested by design (background `#111111`, cards `#1b2328`, accent `#1a98ee`), a status legend, and a geometry plus band-diagram preview area. When the Railway API is unreachable, the viewer switches to a deterministic demo dataset but keeps every interaction available.

### Data separation

- **Small / cacheable JSON**: axis arrays (`eps_bg`, `r_over_a`), lattice list, and the compressed status matrices (`status[lattice][epsIdx][rIdx]`). These travel straight to the browser.
- **Large / structured**: full band diagrams (`k_path`, `bands_TE`, `bands_TM`). These stay on the server and are streamed on demand through `/band` for the currently locked geometry.

---

## 2. Backend on Railway

### 2.1. Storage & infra

- Railway Hobby plan with a dedicated FastAPI service.
- Attach a 5 GB persistent volume at `/data`; upload the `band_library.h5` archive once (Railway UI, `scp`, or init script).
- Environment variables:
  - `H5_PATH=/data/band_library.h5`
  - `BANDLIB_ALLOWED_ORIGINS=https://<your-gh-pages-domain>`
  - `BANDLIB_DEFAULT_SCAN_ID=square_hex_eps_r_v1` (optional convenience)
  - `PORT=8000` (Railway injects one, but mirroring it keeps Dockerfile/start scripts simple)

### 2.2. Tech stack

- Base image: `python:3.11-slim` (plays nicely with `h5py`). When deploying via the Railway UI you can also select the `alpine` builder, but keep the Dockerfile rooted in this slim image to avoid extra musl build hassle.
- Pip dependencies: `fastapi`, `uvicorn[standard]`, `h5py`, `numpy`, `orjson`, `python-multipart`, plus logging/metrics helpers if needed.
- System deps (when using Alpine builder): `python3`, `py3-pip`, `build-base`, `hdf5-dev`. Railway runs your start command, so add an early `apk add --no-cache ...` if you skip the Dockerfile route.

### 2.3. API shape

#### `GET /health`

Simple uptime check returning `{ "status": "ok" }`.

#### `GET /scans/{scan_id}/coverage`

Response example:

```json
{
  "scanId": "square_hex_eps_r_v1",
  "lattices": ["square", "hex"],
  "epsBg": [1.8, 1.9, 2.0, 2.1],
  "rOverA": [0.1, 0.11, 0.12],
  "status": {
    "square": [[0, 1, 3], [0, 2, 3]],
    "hex": [[0, 0, 1], [0, 3, 3]]
  }
}
```

Status codes per pixel:

- 0 -> no data
- 1 -> TM only
- 2 -> TE only
- 3 -> TE + TM

Implementation sketch:

- `with h5py.File(H5_PATH, "r") as f:`
- look up `scan = f["scans"][scan_id]`
- read `axes/eps_bg`, `axes/r_over_a`, and `status/geom_status`
- compress `(lattice, pol, hole, eps, r)` -> `(lattice, eps, r)` using the mapping above
- cache the JSON blob for the lifetime of the worker

#### `GET /scans/{scan_id}/band`

Query parameters: `lattice`, `eps`, `r`, `include_te`, `include_tm`.

Response example (trimmed):

```json
{
  "params": {
    "scanId": "square_hex_eps_r_v1",
    "lattice": "hex",
    "epsBg": 10.2,
    "rOverA": 0.25
  },
  "kPath": [0.0, 0.01, 0.02, 0.03],
  "bandsTE": [[0.24, 0.25, 0.26, 0.27], [0.31, 0.32, 0.33, 0.34]],
  "bandsTM": [[0.36, 0.37, 0.38, 0.39]]
}
```

Implementation sketch:

- resolve lattice/parameter indices from the axis arrays
- slice the `freq` dataset -> `(bands, k)` for TE (pol = 0) and TM (pol = 1)
- add the cumulative `kPath` if it has not been materialized yet

### 2.4. Dockerfile (minimal)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] h5py numpy orjson

COPY app ./app

ENV H5_PATH=/data/band_library.h5

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Railway builds the image, mounts the volume at `/data`, and injects the environment variables.

### 2.5. Repository layout & Railway integration

- Create a dedicated repository (for example `band-library-service`) with this structure:

  ```text
  band-library-service/
  ├── app/
  │   ├── __init__.py
  │   └── main.py        # FastAPI endpoints described above
  ├── requirements.txt   # mirrors the pip list
  └── Dockerfile         # snippet from §2.4
  ```

- Push the repo to GitHub/GitLab. In Railway choose **New Project → Deploy from Repo**, grant access, and pick the repo/branch.
- Set the service builder to **Dockerfile** so the slim image + dependencies are used verbatim.
- Subsequent commits trigger automatic rebuilds; version the HDF5 schema changes in this repo so migrations stay visible.
- If you prefer the UI “Docker image” flow (e.g., `alpine:latest`), keep this repo anyway. Railway clones it to `/app`, runs your start command, and you can use `sh -c "apk add ... && pip install -r requirements.txt && uvicorn ..."` to keep parity with the Dockerfile approach.

---

## 3. Frontend: Data & State

### 3.1. API base URL

Set `NEXT_PUBLIC_BAND_API_BASE=https://<railway-app>.up.railway.app` (dev and prod). The viewer reads it at runtime and decides between live or demo data automatically. The URL now doubles as the cache key for the coverage payload: once fetched, the client memoizes the response per `(apiBase, scanId)` pair so only the first load hits the network.

### 3.2. Shared types

```ts
export type LatticeType = 'square' | 'hex';

export type CoverageResponse = {
  scanId: string;
  lattices: LatticeType[];
  epsBg: number[];
  rOverA: number[];
  status: Record<string, number[][]>;
  updatedAt?: string;
};

export type BandResponse = {
  params: {
    lattice: LatticeType;
    epsBg: number;
    rOverA: number;
  };
  kPath: number[];
  bandsTE?: number[][];
  bandsTM?: number[][];
};
```

### 3.3. Local state machine

- `state.mode`: `loading | ready | offline`. `offline` means the demo grid is showing because the coverage fetch failed or no API base exists. Coverage payloads are cached in-memory per `(apiBase, scanId)` so re-mounting the component stays instant.
- `activeLattice`: UI toggle between square and hex (only one heatmap rendered at a time). Hover/selection state is reset only when the stored lattice no longer exists in the dataset.
- `hovered` vs `selected`: hover highlights cells; clicking locks them and feeds the selection panel plus preview block. Hover callbacks are throttled via `requestAnimationFrame` so pointer storms no longer trash React renders.
- `bandState`: tracks the `/band` fetch status (`idle | loading | ready | error`) along with the chart-ready payload. Responses are cached per `(scanId, lattice, epsIndex, rIndex)` and re-used if the user revisits a geometry.
- `demoLoading`: in offline mode we trigger a fixed 50 ms “fake loading” timer so the band diagram spinner appears even without network latency, mirroring production UX.

---

## 4. Heatmap Canvas (react-konva)

### 4.1. Layout & palette

- `<HeatmapCanvas />` uses `react-konva` (`Stage` -> `Layer` -> `Rect`) plus a ResizeObserver hook so the canvas always fills the card width.
- Colors: cards `#1b2328`, base `#111111`, borders `#263038`, accent `#1a98ee`, hover accent `#4fb4ff`. These live in `palette.ts` and are reused by the legend and selection panel.
- Axes: generated once from the YAML config. eps_bg runs 1.8 -> 14.0 in 0.1 steps (122 samples). r/a runs 0.10 -> 0.48 in 0.01 steps (39 samples). Demo mode reuses the exact arrays.

### 4.2. Interaction model

- Lattice toggle buttons sit above the canvas. Switching lattice resets hover/selection unless the current point already belongs to the new lattice.
- Konva rectangles track pointer move/leave/tap with throttled handlers:
  - move -> `setHovered(point)` (coalesced once per animation frame)
  - leave -> `setHovered(null)`
  - tap/click -> `setSelected(point)`
- Cell geometry is memoized so hover/selection changes only touch a lightweight highlight layer, keeping INP low even on dense matrices.
- A translucent overlay shows "Loading coverage..." during fetches or "No coverage data yet" when a lattice has zero computed entries.

### 4.3. Status encoding

- `STATUS_COLORS`:
  - 0 -> `#1b2328` (missing)
  - 1 -> `#1a98ee` (TM only)
  - 2 -> `#ee5da1` (TE only)
  - 3 -> `#42f5c8` (TE + TM)
- Hover adds a soft outline; locked selections use the bright accent outline.

---

## 5. Selection & Preview Stack

### 5.1. Selection panel

- Lives directly under the heatmap, summarizing lattice, eps_bg, r/a, and status.
- When nothing is selected it shows guidance text. In demo mode a "Demo" badge reminds readers that the preview below is static.

- Wrapper component that renders:
  - **Geometry preview**: SVG lattice (square or hex) generated from the current r/a entirely on the client.
  - **Band diagram plot**: SVG chart with TE/TM lines. Demo mode uses deterministic sine/cos curves so the UI never looks empty.
- Props: `point`, `mode`, `demo`, and `bandState`. When `mode !== 'offline'` and `/band` succeeds, real data replaces the demo curves automatically. While `bandState` is loading (or while the demo spinner delay runs) a translucent overlay with a rotating icon keeps users aware that a new band set is on the way.

### 5.3. Preparing for the Railway `/band` endpoint

- When coverage data loads and the user locks a geometry, the viewer already fires `fetch(`${apiBase}/scans/${scanId}/band?lattice=...&eps=...&r=...`)`.
- Responses convert into `series[]` (one entry per TE/TM band) before feeding the plot component. Up to four bands per polarization are drawn for legibility.
- Errors fall back to a friendly message while the demo preview remains visible.

---

## 6. Geometry & Band Plot Rendering

### 6.1. Geometry preview

- Centered SVG (260 x 220 px) with either a square grid (cartesian offsets) or a hex grid (axial coordinates mapped to pixels). Hole radius uses `spacing * r_over_a * 0.8` with clamping.
- Entirely client-side so demo mode works instantly.

### 6.2. Band diagram plot

- Another SVG (320 x 220 px) with padded axes. `kPath` is normalized to [0, 1] horizontally; frequencies drive the vertical scale.
- Reference grid lines (three x ticks, three y ticks) keep the plot readable.
- Series metadata carries color + label so the legend mirrors the plotted lines.

---

## 7. MDX & Documentation Integration

- `content/documentation/band_library_viewer.mdx` embeds `<BandCoverageViewer />`, explains the Docker workflow (`docker compose up msl_website_dev`), and documents how to export `NEXT_PUBLIC_BAND_API_BASE` before starting Docker when the backend exists.
- Because everything is client-driven, the MDX page stays static and does not need `getStaticProps`.
- The palette matches the dark colors, so toggling Nextra dark mode does not clash with the embedded component.

---

## 8. Deployment Checklist

### Backend (Railway)

- [ ] Create the repository (`band-library-service`) with `app/main.py`, `requirements.txt`, and the Dockerfile from §2.4.
- [ ] Push to GitHub and link it in Railway via **Deploy from Repo** (builder = Dockerfile or Alpine + start command).
- [ ] Attach the 5 GB volume and seed `band_library.h5`.
- [ ] Configure `H5_PATH`, `BANDLIB_ALLOWED_ORIGINS`, `BANDLIB_DEFAULT_SCAN_ID`, and `PORT`.
- [ ] Deploy and note the public URL (for example `https://band-api.up.railway.app`).

### Frontend (GH Pages / Docker dev)

- [ ] From `web/`, run `docker compose up msl_website_dev` to start Next.js inside the dev container.
- [ ] (Optional) copy `apps/website/.env.development` -> `.env.local` and set `NEXT_PUBLIC_BAND_API_BASE` to the Railway URL.
- [ ] Visit `http://localhost:3000/documentation/band_library_viewer` to confirm demo functionality.
- [ ] When the backend is ready, rebuild the site with the env var so the viewer automatically switches to live coverage plus band diagrams.
