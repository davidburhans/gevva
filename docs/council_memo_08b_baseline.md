# Council Memo: openjev-0.8B Baseline for the SOTA-Gate Phase

**Date**: 2026-09-20 (overnight) · **Status**: CONVERGED (pass cap 2, no material disputes remaining)

## Question & scope

openjev-0.8B/2B weights are absent from HF (verified, G3). The pre-registered SOTA
gate (EVALUATION_PROTOCOL §6.2) triggers a Qwen3.5-0.8B vs openjev-0.8B same-size
comparison — how is the openjev-0.8B baseline produced defensibly? Scope: evaluation
methodology + GPU budget only. Non-goals honored: no relitigating audit findings, no
changes to frozen gates, no reordering of tonight's queue.

## Recommendation (converged, both advisors)

**Option C+ — self-train openjev-0.8B from their UNMODIFIED public `train.py`
(Qwen/Qwen3.5-0.8B, AllNLI 200K, bs 32, lr 2e-5, 1 epoch, max_len 256, bf16, seed
logged) as the PRIMARY baseline; quoted 0.869/0.874 become labeled secondary context
(§5 permits labeled table presence, excludes them from ratio/significance claims —
the gate itself uses the repro unconditionally, since McNemar requires per-item
outputs only a local run produces).**

Adopted refinements:
1. **Reproduction-fidelity rule (oracle, revised in Pass 2)**: single-seed shortfall
   vs quote ≠ non-reproducibility. Run repro seed 1 with their defaults; if < quote−2pp,
   run ONE second seed; only after two seeds both miss, publish "not reproducible from
   public recipe (Δ X pp across seeds)" — and the gate STILL uses the repro.
2. **Volume-matched decomposition arm (oracle, Pass 2)**: `train.py --n-train 39494`
   (~1 GPU-hour) produces their recipe at OUR data volume, isolating method-vs-AllNLI
   at matched volume. Headline claim language: "our end-to-end pipeline at its
   operating point vs their published recipe at theirs", with the 5× data difference
   in every table caption. Mechanism claims only from volume-matched cells.
3. **Pre-registered abort criteria (oracle, Pass 2)**: allocator peak >28 GB at step 10;
   no loss decrease by step 200 vs step 50; projected wall-clock >8 h at step 250;
   NaN/inf loss anytime. Abort records the check that fired, in the run manifest.
4. **Due diligence first (oracle, Pass 1)**: search HF/web for mirrored openjev-0.8B
   weights before committing GPU (~30 min, zero GPU).

Accepted reviewer finding: **harness gap** — `_prefetch`/adapter assume HF subfolders;
a local-checkpoint path + a foreign-checkpoint per-item evaluator are REQUIRED before
the 0.8B phase (their train.py persists aggregates only; our gate needs
`test_items.jsonl`-style pairing). Scheduled as implementation work when the phase
triggers, not tonight.

## Owner decisions (ratify at morning review)

1. Ratify C+ + the volume-matched decomposition arm as an additive amendment to
   protocol §6.2 (registered before any results exist).
2. Approve ~4–7 GPU-hours (repro seed 1 [+ seed 2 if fidelity rule fires] +
   decomposition run) when the SOTA gate fires.
3. Confirm quoted openjev rows may remain in tables as labeled context (per §5) —
   or direct full removal.

## Evidence & run ids

- G3 HF tree verification (openjev subfolders) — PROGRESS §9 grounding addendum
- `research/openjev/train.py` recipe; `summary.md` quoted numbers; no run manifest
  upstream (their provenance is weaker by construction — audit pattern A9 analogue)
- Council run ids: Pass 1 oracle `644dcc3f`, reviewer `ac5f5eaf`; Pass 2 oracle
  `3dee7679`, reviewer `259797e9` (workflow `69723120`, `a981f230`)

## Confidence & what would change it

**High** for the recommendation. Would change it: discovery of mirrored openjev-0.8B
weights (due-diligence step makes this cheap); a material cost overrun repro
(abort criteria bound the loss); owner refusal to amend §6.2 (falls back to
quoted-caveated comparison, gate weakened to non-McNemar evidence).

## Roster, passes, fallbacks

`oracle` (fallback for absent council-* profiles; **forked, context-aware**) and
`reviewer` (fresh context, read-only). Passes: 1 (independent) + 2 (curated
cross-exam). Pass 3 not needed — converged with no unsettled material disputes.
