# CourtDyn — Order Without Magnitude

Version 1.0.0. DOI: https://doi.org/10.5281/zenodo.22668000. Reproducibility materials for **Order Without Magnitude: Fine-tuned Vision–Language Models Read Image-Plane Motion in Court Dynamics Question Answering**.

## Release contents

Original `engine/`, `tools/`, `eval/` and `train/` paths are preserved. `results/courtdyn/` contains frozen aggregate findings and tables. `manifests/` contains pool digests and a release inventory. The separate artifacts archive contains 219 sanitized model-output files (85,400 rows) and the CourtDyn-native LoRA adapter.

## Data boundaries and reproducibility limits

Source video, images, annotations and derived QA pools are NOT included. Obtain TeamTrack and Human-M3 under their respective terms before reconstructing pools or rescoring. This release does not grant rights to either dataset. See NOTICE.md.

Published prediction rows contain only model-generated raw/extracted answers, parse/presence flags, row index and a SHA-256 of the canonical original result row. No question, ground-truth answer, trajectory metadata, frame identifier or per-item score is included. The digest is a verification checksum, not a substitute for licensed inputs. Rows retain source-file order. Exact rescoring requires locally rebuilt licensed QA pools and alignment checks; the sanitized outputs alone are NOT a self-contained scoring dataset. Do not pass them directly to the original analysis scripts expecting full predictions.json records.

## Reproduction

Install the dependencies required by the scripts in an isolated environment (PyTorch, transformers, PEFT, bitsandbytes, accelerate, datasets, Pillow, NumPy, SciPy, OpenCV, matplotlib and huggingface_hub; Kaggle access is needed for relevant data acquisition). This list is not a frozen tested GPU environment.

From the repository root, after obtaining dataset access:

```sh
python tools/build_courtdyn_qa.py --help
python tools/build_courtdyn_zoom2.py --help
```

The first builder uses its configured default source sequence; inspect each tool's `--help` before supplying parameters. Store licensed datasets under the paths specified in the scripts. Compare rebuilt pools with `manifests/qa_pools.sha256.json`; byte identity also depends on the same source data and software environment.

Queues T26–T34 are archived experiment orchestration, not an unattended installation command. Review paths, predecessor-state waits and download stages before running. Set QWEN35_4B and QWEN25_VL_7B to your local base-model directories. Historical source-SFT/GRPO adapters from Paper1 are not included here and must be obtained or retrained separately. Only the CourtDyn-native adapter is included in the artifacts ZIP. Load its processor/tokenizer from Qwen/Qwen3.5-4B and its adapter with PEFT.

Four diagnostic protocols cover constant baselines, first-frame repetition, zoom/declutter and visual minimal pairs. Inferences should remain limited to the models, clips and interventions reported in the paper. The second-venue test is inconclusive, not an independent confirmation.

## Citation and licenses

Use the version DOI in CITATION.cff once assigned for an immutable snapshot. Cite TeamTrack and Human-M3 separately. MIT applies ONLY to original code. The adapter is derived from Qwen/Qwen3.5-4B and carries its Apache-2.0 license, supplied alongside the weights. Original weights are unchanged; adapter_config.json replaces a machine-local base path with the public model identifier. No third-party dataset rights are conveyed.
