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
