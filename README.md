# CourtDyn — rank agreement and unit responsiveness

Version **1.1.0**, P1/P2 revision. Version DOI: https://doi.org/10.5281/zenodo.22731710
Associated manuscript: *Rank Agreement and Unit Responsiveness in a Fine-Tuned Vision–Language Model: A CourtDyn Case Study*.

## What this version provides

Corrected source-code import layout, input builders, deterministic scorers,
training/evaluation commands, and aggregate evidence for the revised manuscript.
P1 crosses units and height prompts and includes text-only conversion controls.
P2 contains two matched 1,120-example, 70-step training runs at seed 42,
with v1 versus v3 supervision, two held-out events from one game, and full/static4
evaluation. The completed execution comprises 6,160 visual responses and 36
arithmetic responses. Aggregate reports include exact and rounded dual-reference
rescoring, MAE, parsing denominators, and undefined-correlation handling.

Key files: `manifests/revision_execution.json`,
`results/courtdyn/revision_controls_aggregate.json`, and the paired-reference
summaries listed in `manifests/bundle_inventory.json`.
The box-height, four-frame sampling, cross-question, and no-prior audit code and
aggregate results are included. See [REBUILD.md](REBUILD.md) for staged commands.
Five manuscript authors are recorded in `CITATION.cff`; original code copyright
is preserved separately in `LICENSE`.

## Data and model access

This is a **code-and-aggregate** release, not a self-contained dataset or model
release. Source imagery/annotations, derived QA/training pools, per-item
references, raw predictions, homography matrices and model weights are excluded.
Obtain TeamTrack and Human-M3 inputs under their applicable terms and reconstruct
and align them before inference or scoring. The two newly trained P2 adapters
are not distributed in this release. No third-party dataset rights are granted.

Historical v1.0.0 remains at https://doi.org/10.5281/zenodo.22668000 and contains
the original native adapter and sanitized historical outputs. Those files do
not contain the P1/P2 experiments and are not silently relabeled as new results.
The old source snapshot also lacks corrections supplied here.

## Validation and limits

```bash
python tools/smoke_paper2_release.py --bundle . --report ../smoke.json
```

Synthetic CPU smoke tests check entry points, imports, scoring, and trajectory
fixtures. Author-side restricted-input checks and execution hashes are provenance,
not independent public reconstruction. Full GPU reproduction, cross-platform
environment equivalence, multiple seeds, and cross-game generalization are not
certified. `requirements-*-observed.txt` describe the observed audit environment,
not a frozen training lockfile. See [NOTICE.md](NOTICE.md) and
[VALIDATION.md](VALIDATION.md). Historical aggregate prose is preserved as evidence;
the revised manuscript determines the scope of scientific claims.
