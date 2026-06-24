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
   leg. Report MRR / nDCG@10 / Recall@1k vs. those baselines.

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

## Run

```bash
uv run python -m eval.recds --task ppr --strategies auto,semantic,bm25,fused
```
