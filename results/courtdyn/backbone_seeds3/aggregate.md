# Three-seed aggregation (seeds 42/43/44)

Rules sha256 `a259c9e07ecf2b09...` (backbone_seeds3). Per-seed verdicts; cells never pooled across seeds. SmolVLM2-2.2B is descriptive only (degenerate at the cap).

| backbone | rule | seed 42 | seed 43 | seed 44 | label |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B | R1 | intermediate | replicates | intermediate | not established |
| Qwen2.5-VL-3B | R2_E | intermediate | LIMITED | intermediate | not established |
| Qwen2.5-VL-3B | R2_T | intermediate | intermediate | intermediate | not established |
| Qwen2.5-VL-3B | R3 | replicates | replicates | replicates | holds at all three seeds |
| Qwen2.5-VL-3B | R4 | LIMITED | replicates | replicates | holds at two of three seeds |
| InternVL3-2B | R1 | intermediate | replicates | replicates | holds at two of three seeds |
| InternVL3-2B | R2_E | LIMITED | LIMITED | replicates | not established |
| InternVL3-2B | R2_T | replicates | LIMITED | replicates | holds at two of three seeds |
| InternVL3-2B | R3 | fails | fails | fails | fails across seeds |
| InternVL3-2B | R4 | replicates | replicates | replicates | holds at all three seeds |
| SmolVLM2-2.2B (descriptive) | R1 | intermediate | intermediate | LIMITED | not established |
| SmolVLM2-2.2B (descriptive) | R2_E | LIMITED | LIMITED | LIMITED | not established |
| SmolVLM2-2.2B (descriptive) | R2_T | intermediate | LIMITED | LIMITED | not established |
| SmolVLM2-2.2B (descriptive) | R3 | fails | fails | fails | fails across seeds |
| SmolVLM2-2.2B (descriptive) | R4 | intermediate | intermediate | intermediate | not established |
| Pixtral-12B | R1 | replicates | replicates | replicates | holds at all three seeds |
| Pixtral-12B | R2_E | intermediate | replicates | replicates | holds at two of three seeds |
| Pixtral-12B | R2_T | LIMITED | intermediate | LIMITED | not established |
| Pixtral-12B | R3 | replicates | fails | replicates | seed-dependent |
| Pixtral-12B | R4 | LIMITED | replicates | replicates | holds at two of three seeds |
| Idefics3-8B | R1 | LIMITED | replicates | intermediate | not established |
| Idefics3-8B | R2_E | intermediate | intermediate | LIMITED | not established |
| Idefics3-8B | R2_T | intermediate | intermediate | intermediate | not established |
| Idefics3-8B | R3 | intermediate | replicates | replicates | holds at two of three seeds |
| Idefics3-8B | R4 | LIMITED | replicates | intermediate | not established |
| Gemma3-12B | R1 | fails | intermediate | replicates | seed-dependent |
| Gemma3-12B | R2_E | LIMITED | LIMITED | LIMITED | not established |
| Gemma3-12B | R2_T | intermediate | intermediate | intermediate | not established |
| Gemma3-12B | R3 | fails | fails | fails | fails across seeds |
| Gemma3-12B | R4 | replicates | replicates | replicates | holds at all three seeds |
