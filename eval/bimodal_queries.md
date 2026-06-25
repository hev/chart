# Building the bimodal judged query set (the open question)

The routing headline is only *quantified* once a judged set contains **both**
query shapes. ReCDS supplies the long, NL-shaped half (whole patient summaries).
The missing half is short, clinician-style queries — and their relevance labels.

This is the open question RFC 0076 names. Sketch of the cheap, credible
construction (to be settled before claiming a routing number):

1. **Derive short queries from note entities.** Run an entity pass over a sample
   of notes (drug names, ICD/CPT codes, abbreviations, eponyms). Each entity → a
   short query; the **source note is its relevant doc** (a weak but defensible
   label). This exercises the keyword/fuzzy route at scale, cheaply.
2. **Add a typo tier.** Perturb a fraction (the RFC 0057 fuzzy path) — the all-typo
   query that semantic-only blanks on.
3. **Hand-judge a small gold slice.** A few dozen queries judged by hand to
   sanity-check the derived labels, since (1) is weak supervision.
4. **Mix with ReCDS.** Combine the short set with a sample of ReCDS long queries
   into one judged mix. The claim: per-query `Auto` routing matches or beats any
   single fixed strategy *across the mix*, because the keyword route wins the
   short-token queries the semantic route mishandles and vice versa.

Open sub-questions:
- Is "source note is the relevant doc" too weak? Compare against the hand-judged
  slice before trusting the number.
- Mixing ratio (short : long) — it determines whether `Auto` beats fixed-semantic;
  report the ratio, don't tune it to win.
- Keep this set OUT of the live "find similar patients" candidate set (the holdout
  discipline) — though these are note-retrieval queries, not PPR, so the overlap
  is small.

## Builder

`eval/bimodal.py` implements the weak-supervision version of this plan. It writes
BEIR-style `queries.jsonl` and `qrels.tsv`, metadata documenting the short / long /
typo mix plus the pinned PMC-Patients and ReCDS source revisions, and `review.csv`
for the required hand-judged slice:

```bash
uv run --extra eval python -m eval.bimodal \
  --short-notes 500 \
  --long-limit 500 \
  --review-limit-per-kind 25 \
  --out eval/out/bimodal
```

The labels are not publication-grade until the hand-judged slice exists, but the
artifact is enough to run fixed-strategy comparisons and expose routing failures.
Fill `human_judgment` and `review_notes` in `eval/out/bimodal/review.csv` before
claiming a routing number from the weak labels.

Replay it through the live namespace with the same eval harness:

```bash
uv run --extra eval python -m eval.recds --beir-dir eval/out/bimodal --strategies auto,semantic,bm25,fused
```

The eval JSON includes aggregate metrics plus `metrics_by_kind`, so the routing
claim can be checked separately for `short`, `typo`, and `long` queries instead
of hiding a weak route behind the mixed average. When present, the final audit
requires the bimodal report to carry the same pinned dataset/ReCDS provenance, so
a stale routing-set build cannot accidentally stand in for the current public
pins.
