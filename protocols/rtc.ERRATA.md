# RTC preregistration: clarifications (recorded before the pool or any input was built)

1. **pxK ruler sentence (2026-09-28).** The rules name K = 27.50 / 27.06 *and* "the ruler arm of the
   paper". The paper's ruler arm (results/courtdyn/seq_*/qa_dyn_v1_ruler.json) states the rounded value:
   "In these frames, one meter on the floor spans about 27 pixels." for both overhead clips. The pxK arm
   uses that exact published sentence. Scoring is unchanged: the exact factor for R is the median itemwise
   K_i = pixel reference / metre reference (Q2_top 27.50, Q1_top 27.08), so the stated 27 is within 2% of
   the factor and far inside the [0.5 f, 2 f] conversion band.

2. **Conversion band tightened (amendment, 2026-09-28, before any training or evaluation of this
   line; `state.json` shows no stage run).** A synthetic end-to-end test of `tools/analyze_rtc.py`
   (no model output involved) showed that for exact factors near 3 (km/h x3.6, ft x3.281,
   ft/s x3.281) the registered cell band [0.5 f, 2 f] = [1.64, 6.56] for f = 3.281 accepts a model
   that does not convert but whose unit-arm answers sit about twice its metre answers. The cell
   rule is amended to **f / 1.5 <= R <= 1.5 f** (all other cell conditions unchanged: parse rate
   >= 0.5, > 2 distinct answers, Spearman >= 0.30). Counts needed for H1 (>= 6 of 8) are unchanged.
   The analysis reports every cell under both bands (`converts` = amended, `converts_as_registered`
   = original) and the per-cell R/f, and the paper will state that the band was tightened before
   any run and why.

3. **Replication: pre-run specification (2026-09-28 15:21 UTC, before any
   replication stage is built or run; the seed-42 RTC queue is still running and no seed-42 RTC
   result has been read).** The REPLICATION PLAN leaves the following open; they are fixed now.
   a. Gate: the replication runs only if H1_unseen_units_rtc = "holds" for seed 42 (after the rerun that
      adds the M1-fine comparator). Otherwise nothing below is run.
   b. Seeds 43 and 44 (Qwen3.5-4B): pool results/courtdyn/rtc/rtc_v3_pool.json unchanged (the M1 seeds
      43/44 used the same mixunit pool, so the content-hash cm half is identical); recipe as seed 42
      with --seed 43/44; adapters models/courtdyn-rtc-v3-s43, -s44.
      Matched comparators: v3 s43/s44 = models/courtdyn-v3-s43, -s44 (seed sweep); M1 s43/s44 =
      models/courtdyn-mixunit-v3-s43, -s44 (M1 three-seed sweep). Same arms and formats as seed 42
      (plain U1, U2, pxK; RTC-format m, cm, U1, U2, pxK). M1-fine exists only at seed 42: no M1-fine arm.
      Plain metre cells for R: published cells with byte-identical metre inputs (M1 s43/s44 both clips;
      v3 s43/s44 Q2 from results/a100/seeds). v3 s43/s44 have no published Q1 metre cell: it is
      evaluated with the plain_m input (byte-identical to the published m_height input, sha 81e6016c).
      H4 per seed uses the same-seed v3 plain metre rho on Q2.
   c. InternVL3-2B: RTC pool trained through train/train_qlora_internvl.py (one 448x448 tile per frame)
      on the round-2 schedule (4-epoch cosine, save every 70 steps) stopped at step 140, i.e. the
      recorded k* = 2 of results/courtdyn/backbone3_internvl/internvl3_2b/stopping.json. No new dev
      stopping rule is applied to the RTC adapter. Adapter models/courtdyn-internvl3_2b-rtc-r2ep4-s42,
      checkpoint-140. Comparators: InternVL v3 and mixed-unit adapters at checkpoint-140 (k*); their
      published metre cells (backbone3_internvl eval, byte-identical m inputs). H4 uses InternVL v3.
      All evaluation through tools/internvl_launch.py.
   d. Rules H1-H6 exactly as registered, with the amended conversion band (item 2), applied per seed and
      per backbone; no pooling across seeds. Summary wording fixed now: "replicates across seeds" only if
      H1 holds in all three seeds (42, 43, 44); "in 2 of 3 seeds" otherwise reported as such;
      "replicates on InternVL3-2B" only if H1 holds there. H6 and H4 are reported per seed likewise.
   e. The 392-item text probe is run for each new RTC adapter (descriptive, as at seed 42).
