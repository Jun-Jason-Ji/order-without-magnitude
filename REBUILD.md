# Rebuilding in stages

Commands below run from the release root. No command here invokes the historical
Windows experiment queues, a publisher, an unattended download, or a service.

## 1. Code-only smoke and aggregate figure

Use a Python environment with NumPy, SciPy, OpenCV, Pillow and matplotlib. Observed
versions are supplied as provenance, not a cross-platform tested lockfile.

```bash
python tools/smoke_paper2_release.py --bundle . --report ../candidate-smoke.json
python -c "from pathlib import Path; Path('paper2/figs').mkdir(parents=True, exist_ok=True)"
python tools/make_paper2_native_figure.py
```

The figure reads frozen aggregate findings, not source data or fresh model outputs.
Rerunning it verifies presentation of those summaries, not independent inference.

## 2. Independently authorized data

`manifests/source_inputs.json` lists source sequences, video / MOT-label digests,
and original main-pool parameters. Place licensed inputs under:

```
data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/<family>/<split>/<sequence>/
  seqinfo.ini
  img1.mp4
  gt/gt.txt
```

For a Q1 example (use each sequence's recorded parameters for other pools):

```bash
python tools/build_courtdyn_qa.py --seq_dir data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train/Q1_top_0-30 --out_dir results/courtdyn/seq_Q1_top_0-30 --frame_dir data/courtdyn/frames_Q1_top_0-30 --win_s 2 --hop_s 1 --n_frames 4 --max_per_family 140
python tools/calibrate_court_homography.py --seq_dir data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train/Q1_top_0-30 --out_dir results/courtdyn/homography --n_frames 9
python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant ruler
python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant pxunit
```

Q4_side_480-510 uses `results/courtdyn/` and `data/courtdyn/frames` directly;
other sequences use `seq_<name>` and `frames_<name>`. Calibration is automatic
nine-frame court detection followed by visual quality checks, not supplied manual
ground truth. Compare numerical H separately from timestamped JSON byte hashes.
Original QA serialization used CRLF; LF-only output can have identical values and
a different digest. Do not normalize unknown contents merely to force a hash match.
`REPRODUCIBILITY.md` retains the prior detailed reconstruction notes and limited
author-input check history.

## 3. Native and paired aggregate reconstruction

Prepare an authorized workspace in the same layout with reconstructed and verified
QA pools, H matrices, MOT labels, `teamtrack_probe_hits.json`, ruler manifests, and
aligned original-format predictions. Native controls use the T33 full and T34
static4 predictions for Q1_top and Q2_top. All paired cells additionally require
the original full/ruler/pixel predictions of the compared checkpoints. The code
does not import another paper's results merely to run native controls.

```bash
python tools/rebuild_paper2_statistics.py --workspace /path/to/authorized-workspace --out ../rebuilt-native --check-reference paper2/audit/independence_native_controls_20260912.json
python tools/rebuild_paper2_statistics.py --workspace /path/to/authorized-workspace --out ../rebuilt-paired --paired --bootstrap-reps 2000 --check-reference paper2/audit/independence_native_controls_20260912.json
```

Both output directories must be new. The script reads inputs without changing
them and exports aggregates only. The bootstrap uses seed 20260912 and the original
5/10-second block definition. These are exploratory within-clip intervals, with
only 3–6 clusters per clip; they do not establish independent-match generalization
or equivalence. A lower replicate count is a code check, not the reported interval.

The public sanitized outputs cannot be passed directly to this tool: their source
question-key alignment is withheld. The exact reconstruction/alignment has not
been certified from the public archive in a fresh environment.

## 4. Corrected training split; no new checkpoint is implied

After rebuilding all seven original pools and frames:

```bash
python tools/build_courtdyn_native_sft.py --split event-holdout --dry-run
python tools/build_courtdyn_native_sft.py --split event-holdout
python train/train_qlora.py --model_path /path/to/Qwen3.5-4B --qa_json results/courtdyn/courtdyn_native_sft_train_event_holdout.json --img_root data/courtdyn --output_dir models/courtdyn-event-holdout-new-s42 --num_samples 0 --allow_base_init --budget_slice diagnostic --num_train_epochs 1 --max_steps -1 --seed 42 --lr 0.0001 --grad_accum 16 --lora_r 16 --max_pixels 200704 --save_steps 500 --validate_only
```

The explicit last command validates inputs only. Removing `--validate_only` starts
GPU training and requires a suitable CUDA/4-bit environment; no such run is implied
by assembling this release. Use a new output directory because the trainer can
resume checkpoints in an existing one. The builder excludes Q1_side's opening
event and should produce 1120 training items. The original historical checkpoint used 1400 items; the separate completed P2 pair uses 1120 items each. `--split legacy-cross-view` writes a separate
legacy reconstruction name and marks its event overlap; it does not overwrite the
old frozen training pool.

## 5. Coordinate, unit and height controls

After the original pools and event-holdout list have been independently rebuilt:

```bash
python tools/prepare_courtdyn_revision_experiments.py
python tools/run_courtdyn_revision_controls.py --seq Q2_top_480-510 --arms m_height cm_height px_height m_noheight cm_noheight px_noheight --arithmetic
```

The first command creates restricted local derived pools and records their source
digests. The second validates them on CPU and does not run a model. To execute it,
add `--run --model-path /path/to/Qwen3.5-4B --adapter /path/to/verified-adapter
--output /path/to/new-output`. Every real run requires a new output directory,
an idle supported CUDA device, local model weights and authorized input frames.
`--resume` appends only to an intact prefix with identical configuration and
source hashes; it does not repair a corrupted or changed experiment silently.

Six visual conditions cross m/cm/px with the presence or absence of the same
height sentence. Every condition specifies the original 960×540 canvas. Physical
units describe floor-plane motion; pixels describe projected image-plane motion.
Only m/cm is an exact unit conversion of the same quantity. The numerical control
contains 12 text-only questions with explicit conversion factors. These stimuli
are different from visual measurement questions.

The new runner reports strict finite-number parsing, all-item T-MRA with invalid
answers scored zero, and scores conditional on parsing. It uses exact references;
historical summaries used rounded references and conditional parsing. Changing
both the prompt and scoring version prevents interpreting a difference from an
old headline score as the isolated effect of one prompt edit.

## 6. Matched v1/v3 training and evaluation

The prepared `training/event_holdout_v1.json` and `event_holdout_v3.json` under
`results/courtdyn/revision_controls_20260912/` have identical inputs and order.
They differ in supervised labels. For each convention, use the following
configuration, replacing `<v1-or-v3>` and `<new-adapter-directory>`:

```bash
python train/train_qlora.py --model_path /path/to/Qwen3.5-4B --qa_json results/courtdyn/revision_controls_20260912/training/event_holdout_<v1-or-v3>.json --img_root data/courtdyn --output_dir <new-adapter-directory> --num_samples 0 --allow_base_init --budget_slice diagnostic --num_train_epochs 1 --max_steps -1 --seed 42 --lr 0.0001 --grad_accum 16 --lora_r 16 --max_pixels 200704 --save_steps 35 --validate_only
```

Remove `--validate_only` to train. The completed author-side budget was 1,120 examples and 70
optimizer steps for each convention, with a checkpoint at step 35. Confirm the
actual completion receipt and final adapter hash; a protocol file alone is not
proof of training. The original 1,400-example, 88-step adapter is a different
experiment and is not one member of this matched pair.

For each new adapter, run both `Q2_top_480-510` and `Q1_top_0-30` with
`--arms m_height cm_height px_height --frame-mode full`. For each sequence also
run `--arms m_height --frame-mode static4` in a separate new directory. Add
`--arithmetic` only to the Q2 full run. Use the same generation settings for all
comparisons. The static4 condition repeats the first frame four times while
retaining the question. Both events are held out for these new training pools,
but they are still from one game and one venue. This is one paired seed.

## 7. Analysis and verification of actual outputs

`tools/analyze_courtdyn_revision_results.py --self-test` performs synthetic CPU
checks. With authorized input pools and real result directories, provide repeated
`--run LABEL=/path/to/run` arguments and a new `--output-dir`; use
`--compare-models eventholdout_v1_s42 eventholdout_v3_s42` for the matched pair.
An execution plan supplied with `--plan` defines the expected run coverage and
training receipts. A missing or unfinished run produces PARTIAL rather than a
claim of successful reproduction. Do not use sanitized aggregate exports as
substitutes for the withheld item-level predictions and references.

The host's manually launched experiment pipeline and ASD process handoff are
machine-specific operations, not portable reproduction instructions. They are
not prerequisites for running the study on an independent machine. Presence of
a protocol or script in this release does not establish a completed run;
completion must be supported by the associated result and verification receipts.
