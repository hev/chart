# RFC 0001: Atlas — the routing decision, rendered in the embedding space

Status: Draft (2026-07-03) — a demo-composition RFC, chart's first repo-local
RFC. Like the Layer demo RFCs it is an **architecture mapping, not a release
commitment**. It composes chart with **hev map** (`../map`, binary `hevmap`) and
introduces **no new gateway machinery and no new engine machinery** — anything
here that seems to need either is a finding to file upstream, not code to write
here (see § Where the findings go). Placement note: chart's design of record is
Layer RFC 0076 (`../layer/docs/rfcs/0076-clinical-notes-query-routing-demo.md`);
this RFC lives in chart because its subject is a chart demo surface, owned and
built here, with hev map as a supplier.

> "Atlas" is a provisional view name in the same spirit as chart's provisional
> repo name. Renaming is a string change.

## Summary

chart's identity is **showing the gateway's decision** — the routing badge is
the product. hev map's identity is **showing the embedding space** — UMAP → 3D
point cloud, HDBSCAN clusters, auto-labels, k-NN. Compose them and you get a
demo surface no other search product has: **the routing decision rendered as
geometry.**

Atlas is a view in chart's UI: the indexed PMC-Patients corpus as a labeled 3D
point cloud. Run a search and the hits light up in the cloud. The route type
*visibly manifests as shape*:

- `semantic` — *"elderly woman with progressive dyspnea and bilateral
  lower-extremity edema"* → hits form one tight glowing neighborhood. ANN,
  visible.
- `hybrid_text` — *"metformin 500mg"* → hits scatter as sparks across many
  clusters. Lexical matches live everywhere in the space; that is *why* the
  router refused the semantic route.
- `fused` — both populations at once: the tight semantic core plus the lexical
  outliers RRF pulled in.

Next to the existing routing badge, a **dispersion chip** puts a number on the
shape: *"hits span 1 cluster"* vs *"hits span 11 clusters."* The badge says what
the router chose; the map shows why it was right. Nomic Atlas and the
TensorBoard projector visualize embeddings; nobody visualizes **retrieval
strategy**.

Two structural bonuses fall out of the composition:

1. **The Gemma cascade labels the map for free.** hev map's most expensive
   pipeline step is per-point LLM labeling (OpenAI, one call per doc). chart
   already wrote `specialty`, `diagnosis_category`, and `events` onto every row
   via the GPU cascade (RFC 0076 § classify-events; Layer RFC 0072). The
   exporter already carries those attributes through
   (`map/internal/source/turbopuffer.go` `rowToDoc` keeps all non-vector
   attributes). Color clusters by specialty, sub-label points by diagnosis, and
   the OpenAI dependency disappears for this corpus. This is chart's "one GPU
   pass, many labels" cost pitch made visible.
2. **hev map becomes a Layer design-preview customer.** Its exporter speaks
   `POST /v2/namespaces/{ns}/query` — exactly the Turbopuffer-shaped wire the
   gateway proxies. Pointing the export at
   `https://aws-us-east-1.hevlayer.com/v2/namespaces/chart-notes/query` is the
   "turn onto Layer" that `map`'s capsule in `lyr/CLAUDE.md` names as its
   on-ramp. Paginated full-vector export is traffic the gateway has likely
   never taken seriously; whatever breaks is fresh design-preview signal for
   `hev/layer`.

## The demo (the 90 seconds of wow)

1. Open Atlas: a galaxy of ~11k patient notes, clusters tinted and labeled by
   specialty — *"this is the corpus, seen the way the embedding model sees
   it."*
2. Type `metformin 500mg` → badge `hybrid_text`, sparks scatter everywhere →
   *"lexical matches live all over the space; a semantic route would drown
   this."*
3. Type the dyspnea prose query → badge `semantic`, one neighborhood ignites →
   *"here the router went to ANN — and you can see why."*
4. Toggle **medication discontinuation** → glowing points across every cluster
   → *"one Gemma GPU pass labeled all of this, in-cluster, ~4–5× cheaper than a
   provider batch API"* (the RFC 0076 cost pitch, now visible).
5. Click a hit → its k-NN neighborhood blooms with labels → *"cases like this
   one."*

Every beat renders a decision the stack already made — routing, cascade labels,
neighbors. Atlas adds no retrieval machinery; it is a second way of *seeing*
the same responses the existing UI renders as badges and rails.

The Trio framing is unchanged from RFC 0076: the public twin proves the UX and
the plumbing on published case reports; the same Atlas over Trio's real notes
runs inside their environment, where "the notes never leave the cluster" is the
half of the pitch that matters most. hev map's own GTM line — 80M clinical
patient notes — is literally this corpus domain at Trio scale.

## Architecture

The deploy split (`lyr/AGENTS.md`) holds: **no Python on the live request
path.** Atlas is a batch-produced static artifact plus client-side rendering.

```
                    (batch, offline — hev map's Docker Compose stack, local or cluster job)
gateway ── export ──► hevmap pipeline: UMAP → HDBSCAN → k-NN → labels-from-facets
  │                          │
  │                          ▼
  │                 scripts/build_atlas.sh  (chart) — reads hevmap's
  │                 GET /api/v1/maps/{id}/points, emits the Atlas artifact
  │                          │
  │                          ▼
  │                 web/static/atlas/points.json   (gitignored; uploaded as a
  │                                                 Worker static asset at deploy)
  │
  │                 (live request path — unchanged)
  └── query ──► src/worker.js / search/app.py ──► routing + hybrid echo + row ids
                             │
                             ▼
                web/static/atlas.html — vendored three.js point cloud:
                highlight hit ids, compute dispersion chip, events glow,
                neighbor bloom. Pure client JS; the Worker serves bytes.
```

### The pipeline run (batch)

- **Source:** the `chart-notes` namespace **through the gateway wire** —
  `hevmap connect --type turbopuffer --host https://aws-us-east-1.hevlayer.com
  --namespace chart-notes` with a Layer inbound key. hev map neither knows nor
  cares which backend Layer fronts — **Turbopuffer today** (the hev search
  cutover in `deploy/vectorstore.yaml` is declared but not applied); that
  opacity is the point, and it is what makes this a real customer workload
  rather than a backdoor read. **The export never goes around the gateway**
  (see § The Layer on-ramp for the fallback discipline).
- **Labels:** from the row attributes the cascade already wrote —
  `specialty` (cluster tint), `diagnosis_category` (point sub-label), `events`
  / `has_med_discontinuation` (the glow layer), `age_band` / `gender`
  (filter dims). No OpenAI calls. TF-IDF stays as hev map's fallback for
  clusters whose members predate the cascade backfill.
- **Determinism:** pin the UMAP seed so a rebuild over the same index produces
  the same galaxy (demo muscle memory matters; presenters learn the shape).
  Re-run the pipeline when the index materially changes (slice → full corpus).

### The artifact

`points.json` (or a length-prefixed binary if size warrants), one record per
indexed note:

```
{ id, x, y, z, cluster_id, specialty, diagnosis_category,
  events[], has_med_discontinuation, age_band, gender,
  neighbors: [{id, dist} × k] }
```

plus a `clusters` block (`cluster_id → label, count, centroid`). Explicitly
**no note text and no titles** — datasets are never checked in and never shipped
in bulk (`lyr/AGENTS.md`); hover/click previews fetch the row through the
existing search backend by id, same as every other UI surface. At ~11.4k notes
the artifact is single-digit MB; at the 167k full corpus, packed binary + gzip
keeps it in the tens of MB, which is acceptable for an explicitly heavyweight
view behind a click (lazy-loaded, never on `index.html`'s critical path). The
file is **gitignored** and uploaded as a Workers Static Asset at deploy time,
like a build output — because it is one.

### The viewer

- A new `web/static/atlas.html` in chart, keeping the family's no-framework
  rule for UI code: vanilla JS + a **vendored three.js** (a rendering library,
  not an app framework — the same line shelf draws). hev map's `viewer/`
  scaffold is already three.js-shaped (Vite + TS + `three@0.170`); its sigma
  variant stays the 2D alternative and is not used here — for wow, 3D wins.
- One `THREE.Points` cloud with per-vertex color handles 167k points without
  breaking a sweat; no per-point DOM.
- **Hit highlighting is client-only:** the existing search response already
  returns row ids and the routing echo; the viewer maps ids → positions,
  brightens them, dims the rest, and computes the dispersion chip in JS. The
  demo chips in `web/static/queries.json` double as the scripted Atlas beats.
- **The query is deliberately not projected into the space.** Projecting a new
  vector needs a live UMAP `transform()` — Python on the request path, and a
  lie waiting to happen (out-of-sample projection is the least trustworthy
  thing UMAP does). Highlighting *returned hits* tells the routing story
  without either cost.
- The honesty block (published case reports ≠ EHR; non-commercial; not for
  clinical use) renders on Atlas exactly as on the main page — a UI
  requirement, not a footnote (RFC 0076).

### The dispersion chip, honestly

The chip reports **HDBSCAN cluster membership of the top-k hits** — e.g. "top
20 hits: 1 cluster" vs "top 20 hits: 11 clusters" — not distances. Two reasons:

- Cluster membership is a *descriptive* statistic of the precomputed map;
  3D UMAP distances are famously unfaithful to high-dimensional distances, and
  we will not hang a quality claim on them.
- The routing *number* remains the ReCDS qrels harness (`eval/`), full stop.
  Atlas is illustration, not evaluation. If the dispersion pattern proves
  stable across the built bimodal query set (`eval/bimodal_queries.md`), a
  descriptive table of route → median cluster-spread can join the eval README
  as color — clearly labeled as descriptive, never as the headline number.

## The Layer on-ramp (hev map becomes a customer)

The export leg is the strategic payload of this RFC. Details, verified against
`map/internal/source/turbopuffer.go`:

- The exporter paginates `POST {base}/v2/namespaces/{ns}/query` with
  `rank_by: ["id","asc"]` + last-id cursor, requesting vectors and all
  attributes. The `BaseURL` override exists in the source config but is **not
  yet exposed** — `hevmap connect --region` formats `{region}.turbopuffer.com`
  URLs only. **map-side change #1:** a `--host` flag on `connect`, plumbed
  through the connection record to the source. Small, and generically useful
  (any tpuf-shaped endpoint).
- **map-side change #2:** a "labels from attributes" path — `--label-field
  specialty --point-label-field diagnosis_category` (names illustrative) that
  skips the LLM labeling steps when the source rows already carry labels. This
  is a real hev map feature with a life beyond chart: *bring your own labels*
  is exactly what a customer with an existing classifier pipeline wants.
- Auth: chart's inbound keys are Layer-issued scoped keys
  (`deploy/apikey.yaml`); the export needs query entitlement on `chart-notes`
  and nothing else. hev map just sends the bearer it is given.
- **Expected friction, and where it goes:** paginated full-vector reads
  through the gateway are a cold path — pagination-cursor fidelity, vector
  round-tripping, response-size limits, rate shaping on bulk reads are all
  plausible gaps. Each one is a GitHub issue on `hev/layer` (bug/doc) or a
  Layer RFC (capability gap), with this export as the motivating workload. If
  the backing store has cut over to hev search and the gap is engine-side
  (scan behavior, cursor semantics), it routes to `hev/search` instead — same
  split as everything else.
- **Fallback discipline:** the rows live in the default Turbopuffer store
  today, so a direct-tpuf export exists as an escape hatch: if the gateway
  export path is blocked, it may unblock a *build* while the issue is open —
  but the issue must exist first, and the gateway path is a phase gate below,
  not a nice-to-have. The fallback is temporary by construction: once chart
  cuts over to hev search (`deploy/vectorstore.yaml`, pending), the gateway
  export becomes the only road, and any gap is a blocking upstream issue.
  Getting the export solid *before* the cutover is cheap insurance.

## What this explicitly does not do

- **No live dimensionality reduction, no query projection.** Batch artifact +
  client highlighting only.
- **No retrieval machinery in the viewer.** Fusion, routing, fuzzy, facets stay
  in Layer; the engine internals stay in hev search. Atlas renders echoes.
- **No hev map services on chart's request path.** The hevmap server, compute
  sidecar, and Aerospike run only during the batch build (Docker Compose,
  local or a cluster Job); nothing of hev map deploys with chart.
- **No eval claim.** Qrels remain the number (RFC 0076 § eval); the dispersion
  chip is descriptive.
- **No new gateway or engine code written from here.** Gaps get filed, not
  patched (chart `CLAUDE.md` contract).

## Plan

Phases are ordered by derisk value; each has a gate. Phase 0 is an afternoon
and answers whether the whole composition works before any viewer code exists.

### Phase 0 — export smoke through the gateway

Run hev map's export against `chart-notes` via the gateway (using the `BaseURL`
test override directly if `--host` isn't built yet), on whatever slice is live.

- **Gate:** N rows exported with intact 768-d vectors and cascade attributes,
  through `aws-us-east-1.hevlayer.com`, cursor pagination completing cleanly.
- **Deliverable either way:** if it fails, the first `hev/layer` (or
  `hev/search`) issue of the project, with the exact request/response repro.
  That issue is a success condition of this phase, not a failure of it.

### Phase 1 — map build with cascade labels

map-side changes land (`--host`, labels-from-attributes), then a full pipeline
run over the live slice: UMAP (pinned seed) → HDBSCAN → k-NN → attribute
labels. `scripts/build_atlas.sh` (new, chart) pulls the points and emits the
artifact.

- **Gate:** map status `ready`; ≥95% of points carry a facet-derived label;
  artifact validates against its schema and stays within the size budget
  (≤10 MB gzipped for the slice).
- **Dependency:** the slice must be indexed live with cascade labels present —
  i.e. chart `PLAN.md` Phase 2/3 (slice + base facets) and the Phase 4
  classifier smoke for the `events` dimension. Atlas is a reason to finish
  those gates, not a way around them. Without Phase 4, the build runs with
  specialty/age/gender only and the events layer waits.

### Phase 2 — the Atlas view: highlight + dispersion

`web/static/atlas.html` + vendored three.js: render the cloud with specialty
tints and cluster labels; wire the existing search flow so a query highlights
hit ids and renders the dispersion chip beside the routing badge; lazy-load the
artifact; honesty block; `help.html` gains an Atlas section.

- **Gate:** the three scripted queries (from `queries.json`) produce visibly
  distinct hit geometry matching their route, in both backends (FastAPI dev and
  Worker prod, lockstep rule); Atlas adds zero requests to the default page's
  critical path.

### Phase 3 — the events glow + neighbor bloom

The medication-discontinuation toggle (glow layer from `events` /
`has_med_discontinuation` attributes in the artifact) and click-a-hit → k-NN
neighborhood bloom with labels (neighbors are already per-point in the
artifact).

- **Gate:** glow layer counts reconcile with the `events` facet snapshot
  (same provenance discipline as the rail); neighbor bloom works offline from
  the artifact alone (no extra backend calls except the text preview by id).
- **Dependency:** chart `PLAN.md` Phase 4 cascade gates (accepted classifier
  cost report, `CHART_REQUIRE_EVENT_FACETS=1` smoke green).

### Phase 4 (stretch) — the agent constellation

With the Agentic toggle on, render each of the agent's reformulated query
variants' hit sets in a distinct hue — the agent's fan-out as a constellation.
Pure client work over `$agent` provenance the response already carries
(RFC 0074 / Agents API).

- **Gate:** provenance in the standard response suffices; if the per-variant
  hit attribution isn't in the echo, that's an Agents API gap → `hev/layer`
  issue/RFC, and this phase waits on it.

### Cost and footprint

No GPU, no OpenAI, no new always-on deploy. UMAP+HDBSCAN on 11.4k×768 is
minutes on a laptop; the 167k full corpus is tens of minutes, still a batch on
hev map's Compose stack (its eventual K8s+KEDA path is map's own roadmap, not a
prerequisite). The artifact is static bytes on the existing Worker. Marginal
demo cost ≈ zero — which is itself on-message.

## Where the findings go

| Finding | Destination |
|---|---|
| Gateway export gaps (pagination, vector round-trip, limits, auth scoping) | `hev/layer` issue; capability gap → Layer RFC (`layer/docs/rfcs/`) |
| Engine-side export/scan behavior once the backend is hev search | `hev/search` issue / RFC (`search/docs/rfcs/`) |
| hev map features (`--host`, labels-from-attributes, artifact export) | `map` repo (its own issues/roadmap) |
| Atlas view, artifact builder, demo script | here (chart) |
| Agents API provenance gap (Phase 4) | `hev/layer` issue / RFC |

## Open questions

- **Artifact format:** JSON first; switch to packed binary (Float32 positions,
  varint ids) only when the full corpus makes it worth it.
- **Full-corpus UX:** at 167k points, cluster labels need LOD (show top-N by
  count, reveal on zoom). Slice-scale first; LOD when Phase 6 lands.
- **Where the batch build runs long-term:** local Compose is fine for the
  demo; if the rebuild should be reproducible in-cluster, that's a plain K8s
  Job in chart's `deploy/`, not a new runtime.
- **Name:** "Atlas" collides with Nomic Atlas; fine for a view name inside
  chart, worth revisiting if it ever fronts marketing.
