# Release boundaries and attribution

This is a derived evaluation resource, not a new source-image collection.

- Original code: MIT (LICENSE).
- Aggregate statistics and model outputs: supplied as research artifacts; no license to third-party datasets is granted. No blanket CC0 claim is made over third-party-derived content.
- CourtDyn-native LoRA: derived from Qwen/Qwen3.5-4B, Apache-2.0. Weight bytes unchanged. See models/courtdyn-native-sft/LICENSE in the artifacts archive. Processor/tokenizer must be acquired from the base-model publisher.

TeamTrack source data and derived QA pools are withheld because redistribution permission has not been established. Human-M3 source data and derived QA pools are withheld under the source release's access restrictions. Obtain authorization independently. No source imagery, annotations, per-item ground truth, questions, trajectory metadata or per-item scores are distributed in the public prediction files. Row-level quality-control verdicts and training auxiliary files are excluded as well.

Prediction files contain only model outputs, status flags, source-row ordering and opaque verification digests. Aggregate results retain statistical summaries, not per-item annotation tables. Complete scoring requires licensed local reconstruction; see README.md.

Attribution: Scott et al., TeamTrack: A Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos (CVPRW 2024), arXiv:2404.13868. Fan et al., Human-M3: A Multi-view Multi-modal Dataset for 3D Human Pose Estimation in Outdoor Scenes (2023), arXiv:2308.00628. Qwen Team, Qwen3.5-4B.
