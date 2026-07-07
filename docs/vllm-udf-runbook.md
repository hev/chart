# Running vLLM in a Layer Function — what it actually takes

Everything below was learned the hard way activating `chart-classify-events`
(Gemma-2-9B on a `g5.xlarge`/A10G, vLLM ≥0.23) on 2026-07-02: five distinct
environmental failures, each discovered only after fixing the previous one,
each costing an image rebuild (~15 min) plus the hev/layer#148 re-registration
dance. This is the checklist that would have made it one build.

The distilled version of this list is proposed as a Layer-provided base image
in `../layer/docs/rfcs/0094-gpu-inference-base-image.md`; until that ships,
every self-hosted-model UDF should copy `deploy/Dockerfile.gpu`'s `classifier`
stage rather than starting from a bare CUDA image.

## The five failures, in the order they hide behind each other

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `Failed to find C compiler` at engine start | torch-inductor compiles kernels at runtime; `nvidia/cuda:*-runtime` ships no toolchain | `apt-get install gcc libc6-dev` |
| 2 | `fatal error: Python.h: No such file or directory` (buried — vLLM swallows the compiler stderr; reproduce the triton build inside the pod to see it) | triton's `cuda_utils` is a C **Python extension**; `python3` ≠ `python3-dev` | `apt-get install python3-dev` |
| 3 | `FAILED: [code=127] ...csrc_sampling.cuda.o` → EngineCore death loop | FlashInfer JIT-compiles CUDA kernels and needs `nvcc`, which only `-devel` images ship | `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampler; or use a `-devel` base and pay ~3GB) |
| 4 | `ValueError: To serve at least one request with the model's max seq len (8192), 2.63 GiB KV cache is needed (0.75 GiB available)` | 17.2GiB bf16 weights + CUDA-graph capture on a 24GB A10G leave almost nothing for KV at the model-config default seq len | `max_model_len=4096` (or whatever your rows need), `gpu_memory_utilization=0.95`, `enforce_eager=True` |
| 5 | Anonymous HF download 429s, then the gated-repo wall; a token-less **build** silently skips the weight bake and ships a weightless image that re-downloads 17GB every cold start | `google/gemma-2-9b-it` is gated; the Dockerfile's preload was best-effort | Bake weights with a BuildKit secret (`--secret id=hf_token`) and make download failure **fail the build**; inject `HF_TOKEN` at runtime as belt-and-braces |

Failure 4 deserves emphasis because it's the quiet one: the engine *starts*
loading, compiles cleanly, then refuses at KV-cache allocation — and with the
UDF claim/lease loop running alongside, items get claimed by a worker whose
engine is in a death loop, so the queue shows `processing` forever and
`rate/min: 0` with **zero failed items**. If the queue isn't draining and
nothing is failing, read the EngineCore logs, not the queue.

## The checklist for the next model UDF

Image (see `deploy/Dockerfile.gpu`, `classifier` stage):

- [ ] `gcc libc6-dev python3-dev` on top of the CUDA **runtime** base
- [ ] `VLLM_USE_FLASHINFER_SAMPLER=0` unless the image has `nvcc`
- [ ] Weights baked at build time, behind a BuildKit secret if gated; the
      preload must *fail the build* on failure, never skip
- [ ] Weights mirrored to S3 (`s3://hevlayer-models-186219257916-us-east-1/
      <org>/<model>/<revision>/`, `LATEST` pointer alongside) — in-region, no
      HF dependency or rate limits at runtime. `_ensure_weights()` in
      `functions/classify_events.py` restores the hub layout from the mirror
      when the baked cache is absent; HF is only touched at mirror time.
      Caveat: the runtime restore path needs S3 read on the models bucket —
      grant it to the GPU node role (or IRSA on the Function's service
      account) before relying on it; the baked image needs no IAM
- [ ] Expect a fat image (ours: 25GB) — first pull per node is ~10–15 min on
      default EBS throughput; baked weights beat a 17GB HF download on every
      cold start anyway

Engine construction (see `functions/classify_events.py:_engine`):

- [ ] `max_model_len` sized to your rows, not the model card (rows here are
      ~512-token chunks; 4096 is generous). This is what makes a 9B model fit
      a 24GB card at all
- [ ] `gpu_memory_utilization=0.95` on a dedicated node (nothing else wants
      the VRAM)
- [ ] `enforce_eager=True` — CUDA-graph capture memory buys little for
      batch-heavy guided decoding
- [ ] Truncate input text to fit `max_model_len` (we cap at ~12k chars)
- [ ] Guided decoding: mark every field you need `required` — optional schema
      fields get silently omitted by the model and your parser defaults fill
      the facet with `other`

Worker loop (see `run_batched_worker`):

- [ ] One `generate()` per claim (continuous batching), not per row — the
      client's `run_udf_worker` is per-row today (RFC 0068 §1 is the fix)
- [ ] One multi-row `patch_columns` per claim for writeback
- [ ] Lease sized to a whole batched pass (`HEVLAYER_UDF_LEASE_SECONDS=600`),
      not the 120s default
- [ ] Per-row parse failures degrade to an empty result; never fail the batch

Cluster (all found here first, all reported):

- [ ] AWS G/VT-instance vCPU quota: default 4 vCPUs = **one** `g5.xlarge` per
      account. Request the bump before the demo day, not during
- [ ] hev/layer#148: any Function spec change (image tag!) bricks gateway
      re-registration; every roll currently needs `DELETE /v2/udfs/{id}` +
      operator re-create + resume + re-discover
- [ ] hev/layer#150: on kind=search namespaces, re-discovery after the first
      completions 400s (virtual `_hevlayer_*_stale_after` leaks into the
      engine filter) — the backfill can only be restarted by a Layer fix.
      (kind=search only; `chart-notes` is on Turbopuffer today, so this does
      not currently bind chart)
- [ ] hev/layer#149: a poison document in *another* tenant's pipeline can hold
      the shared GPU with zero progress; queue depth is not progress
