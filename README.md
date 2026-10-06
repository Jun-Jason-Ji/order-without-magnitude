# CourtDyn — Order without Magnitude (release 2.0.0)

Concept DOI (all versions): https://doi.org/10.5281/zenodo.22667999 · previous version: https://doi.org/10.5281/zenodo.22731710
Associated manuscript: *Order without Magnitude: Single-Unit Fine-Tuning Can Make Vision-Language Models Rank Motion but Ignore the Requested Unit* (Neurocomputing, submitted 2026).

## What this version provides

Code, preregistered decision rules and aggregate results for every number in the manuscript and
its supplement. `results/courtdyn/paper2_v2_numbers.json` is the single table of quoted numbers;
`tools/audit_paper2_v2.py` and `tools/audit_paper2_supp.py` recompute each one from the files under
`results/` and fail if the text disagrees. `protocols/INDEX.json` lists every preregistration with
the SHA-256 of its rules and the ERRATA recorded during the runs.

### Records
- `results/courtdyn/backbone2/`
- `results/courtdyn/backbone2_r2/`
- `results/courtdyn/backbone3_internvl/`
- `results/courtdyn/backbone4_a100/`
- `results/courtdyn/backbone4_ext/`
- `results/courtdyn/backbone_seeds3/`
- `results/courtdyn/backbone_llama32/`
- `results/courtdyn/rtc/`
- `results/courtdyn/rtc_rep/`
- `results/courtdyn/repr_probe/`
- `results/courtdyn/t36/`
- `results/courtdyn/m1_mixunit/`
- `results/courtdyn/m1_fine/`
- `results/courtdyn/soccer_venue/`
- `results/courtdyn/unit_probe_v2_runs/`
- `results/courtdyn/unitdyn_posthoc/`
- `results/courtdyn/revision_execution_20260912/`
- `results/a100/seeds/` (primary backbone, seeds 43/44)

### Code (v2 additions; the v1.1.0 generator, scorers and queues are carried forward unchanged)
- **frozen evaluation and training**: `eval/run_bench.py`, `eval/evaluate.py`, `eval/llm_extract.py`, `eval/paired_relational_scorer.py`, `eval/prompt_reference.py`, `train/train_qlora.py`, `train/model_family.py`, `train/train_qlora_epochs.py`, `train/train_qlora_family.py`, `train/train_qlora_idefics3.py`, `train/train_qlora_internvl.py`, `train/train_durable.py`
- **A100 harness (families, image policy, durable runner, receipts)**: `a100/__init__.py`, `a100/config.py`, `a100/eval_controls.py`, `a100/model_families.py`, `a100/family_policy.py`, `a100/runner.py`, `a100/durable.py`, `a100/durable_runner.py`, `a100/backbone4.py`, `a100/seeds3_runner.py`, `a100/seeds3_local_runner.py`, `a100/ext_runner.py`, `a100/exp_backbones.py`, `a100/exp_seeds.py`, `a100/analyze.py`, `a100/progress.py`, `a100/selftest.py`, `a100/preflight.py`, `a100/check_all.py`
- **second backbones (rounds 1-6)**: `tools/run_backbone_replication.py`, `tools/run_backbone_round2.py`, `tools/run_backbone_internvl.py`, `tools/run_backbone_a100.py`, `tools/run_a100_ext.py`, `tools/run_backbone_seeds3_local.py`, `tools/build_backbone_dev.py`, `tools/family_launch.py`, `tools/internvl_launch.py`, `tools/analyze_backbone_replication.py`, `tools/aggregate_backbone_seeds.py`, `tools/decide_paper2_title.py`
- **mixed-unit control, text probe, two-stage routing**: `tools/build_mixunit_pool.py`, `tools/run_m1_mixunit.py`, `tools/analyze_m1_mixunit.py`, `tools/run_m1_fine.py`, `tools/build_unit_probe.py`, `tools/run_unit_probe.py`, `tools/analyze_unit_probe.py`, `tools/run_two_stage_units.py`, `tools/analyze_two_stage_units.py`, `tools/build_paper2_arithmetic_dev.py`, `tools/run_paper2_arithmetic_batch.py`, `tools/paper2_arithmetic_model_probe.py`, `tools/mixunit_pool_stats.py`
- **read-then-convert**: `tools/run_rtc.py`, `tools/run_rtc_replication.py`, `tools/analyze_rtc.py`, `tools/analyze_rtc_replication.py`, `tools/analyze_rtc_a100.py`
- **hidden-state probe**: `tools/repr_probe_extract.py`, `tools/repr_probe_analyze.py`, `tools/repr_probe_stability.py`, `tools/repr_probe_nested.py`, `tools/repr_probe_nested_table.py`, `tools/mixunit_item_ratios.py`
- **seeds, bootstrap, soccer venue, zero-shot sweep**: `tools/run_seeds_queue.py`, `tools/analyze_seed_sweep.py`, `tools/bootstrap_paper2_ci.py`, `tools/build_soccer_venue.py`, `tools/run_soccer_venue.py`, `tools/analyze_soccer_venue.py`, `tools/a100_t36.py`, `tools/analyze_t36.py`, `tools/unitdyn_prop2_posthoc.py`
- **manuscript tables, figure and audits**: `tools/make_paper2_v2_tables.py`, `tools/make_paper2_extra_tables.py`, `tools/make_paper2_backbone_figure.py`, `tools/audit_paper2_v2.py`, `tools/audit_paper2_supp.py`, `tools/rebuild_paper2_statistics.py`

## Data and model access

Code-and-aggregate release. Source imagery and annotations, the generated question pools, homography
matrices, per-item prediction rows with references, hidden-state features and adapter weights are
excluded; `manifests/qa_pools.sha256.json` lets a licensed user verify a rebuild byte-for-byte. The
backbones are public checkpoints pinned by revision in the protocols. See NOTICE.md.

## Validation

`VALIDATION.md` records the content checks run on this bundle (forbidden file types, source-derived
keys, local paths). Full GPU reproduction is not certified; the receipts in each cell's
`run_config.json` record the exact loader, image policy and hashes of the frozen scripts used.
