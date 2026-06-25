# `eval/` — the ReCDS eval harness

chart is the **first Layer demo with real relevance judgments**. PMC-Patients
ships the **ReCDS** benchmark in BEIR/TREC format, so the routing/hybrid claim
gets a number — the SciFact eval-spine pattern (RFC 0057), now on judged clinical
retrieval. Two tasks:

- **ReCDS-PPR** (patient→patient, 155k-patient corpus, fully on HF) — the
  tractable lane, and a product feature too ("find similar patients").
- **ReCDS-PAR** (patient→article, 11.7M-article corpus on Figshare) — heavier;
  the patient→article second act.

`zhengyun21/PMC-Patients-ReCDS` carries `queries.jsonl` / `corpus.jsonl` / TREC
`qrels.tsv` (graded 1/2 from the PubMed citation graph), so it drops into
`ir_measures` / `pytrec_eval` directly.

## Two claims, kept separate (RFC 0076 § Evaluation)

The honesty is the point — conflating these would overclaim:

1. **Retrieval quality / hybrid thesis — quantifiable now (ReCDS-PPR).**
   sparse+dense fusion ≥ either leg alone on judged clinical retrieval. The
   paper's own best system is RRF of sparse+dense, which independently validates
   Layer's hybrid thesis; BM25 is a strong PPR baseline, validating the keyword
   leg. Report MRR / nDCG@10 / Recall@1k vs. those baselines. The harness
   encodes the public leaderboard's RRF and BM25 baselines as fractions and
   prints per-metric deltas.

2. **Routing value — the headline, NOT yet quantified.** ReCDS queries are whole
   patient summaries: long, uniformly NL-shaped, so run as-is they all route
   `semantic` and never exercise the keyword/fuzzy route or the routing *decision*.
   The routing number needs a **bimodal judged query set** — short clinician-style
   queries (drug / code / abbreviation, with relevant-note labels) alongside the
   long NL ones. That set does not exist off the shelf and must be built (the open
   question in `bimodal_queries.md`). Until then, the keyword route is shown
   qualitatively via the chips (`web/static/queries.json`), the semantic/fused
   routes quantitatively via ReCDS.

## Holdout discipline

`similar_patient_ids` both *powers* the "find similar patients" feature and *is*
the PPR qrels. Keep the eval split out of the live feature's candidate set or the
number is circular (RFC 0076 open question).

Audit that overlap before interpreting a PPR number:

```bash
uv run --extra eval python -m eval.holdout \
  --split dev \
  --examples 10 \
  --max-overlap-edges 0 \
  --out eval/out/holdout-report.json
```

The report counts undirected patient-patient edges shared by the live
`similar_patient_ids` feature and ReCDS-PPR qrels, with examples to remove or hold
out before making a retrieval-quality claim. `--max-overlap-edges` makes the
audit a hard gate; use `0` for a strict non-circular ReCDS number. The gate also
requires non-empty feature and qrel edge sets, so an empty audit cannot pass as
"no leakage."

## Run

```bash
uv run --extra eval python -m eval.recds \
  --task ppr \
  --strategies auto,semantic,bm25,fused \
  --out eval/out/recds-report.json
```

For live smoke on a slice, keep both query count and depth bounded:

```bash
CHART_EVAL_LIMIT=25 CHART_EVAL_TOP_K=100 scripts/eval_live.sh
```

The full Phase-5 gate remains `--limit 500 --top-k 1000` after the full PPR
corpus is indexed. On the current 2k-note slice, quality metrics are expected to
be zero because most judged relevant patients are outside the slice. The harness
still exercises the gateway and reports per-query failures. Each strategy summary
prints query coverage (`attempted`, `succeeded`, `failed`, `scored`) plus up to
five query errors and a truncation count. The CLI exits before gateway setup if
the selected split/limit contains no judged queries. In live testing on
2026-06-24, forced lexical ReCDS routes returned gateway 502s for long patient
summaries. The harness caps the lexical side of forced `bm25`/`fused` queries
while still embedding the full query text, and `--progress-every` keeps long
gate runs observable; keep non-green gate results visible in the JSON output
instead of overwriting them with wrapper failures.

After the full index exists (`plan_audit` requirement: `full_retrieval_corpus`),
make query replay failures fatal:

```bash
CHART_EVAL_LIMIT=500 \
CHART_EVAL_TOP_K=1000 \
CHART_EVAL_HOLDOUT_MAX_OVERLAP=0 \
CHART_EVAL_REQUIRE_NO_FAILURES=1 \
CHART_EVAL_REQUIRE_FUSED_DOMINATES=1 \
CHART_EVAL_HOLDOUT_REPORT=eval/out/holdout-report.json \
CHART_EVAL_RECDS_REPORT=eval/out/recds-report.json \
scripts/eval_live.sh
```

The final audit prints these compact next-step equivalents for the same gates:

```bash
uv run --extra eval python -m eval.holdout --split dev --max-overlap-edges 0 --out eval/out/holdout-report.json
CHART_EVAL_LIMIT=500 CHART_EVAL_TOP_K=1000 CHART_EVAL_REQUIRE_NO_FAILURES=1 CHART_EVAL_REQUIRE_FUSED_DOMINATES=1 scripts/eval_live.sh
```

`CHART_EVAL_HOLDOUT_MAX_OVERLAP` makes the wrapper run `eval.holdout` before
query replay. Equivalently, call the harnesses directly with
`eval.holdout --max-overlap-edges`, then `eval.recds --require-no-failures` and
`--require-fused-dominates`. Those flags prove replay completed cleanly and make
the PLAN.md sparse+dense claim a hard gate by requiring fused to meet or beat
both BM25 and semantic on RR@10, nDCG@10, and Recall@1000. The printed baseline
deltas still carry the published-baseline comparison. The aggregate ReCDS report
also includes provenance for the pinned `zhengyun21/PMC-Patients-ReCDS` revision,
embedding model, embedding dimension, and namespace; the final audit requires
that provenance before accepting Phase 5. `scripts/eval_live.sh` also writes the
holdout report to `CHART_EVAL_HOLDOUT_REPORT`, the aggregate ReCDS report to
`CHART_EVAL_RECDS_REPORT`, and, when `eval/out/bimodal` exists, the routing-set report to
`CHART_EVAL_BIMODAL_REPORT`.
