# Reproduction scope and inputs

The historical local release moved scripts into role folders while imports and
relative paths assumed the original repository layout. It also omitted calibration
and support modules. This bundle fixes the code layout and includes those modules.
Neither the historical release nor this repair alone establishes exact rebuilding
of every reported artifact.

## Authorized source data and pool construction

`manifests/source_inputs.json` gives the TeamTrack family/split/sequence mapping,
source video and MOT-label SHA-256, and the generation parameters recorded with
each original main pool. Place independently obtained data under:

```
data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/<family>/<split>/<sequence>/
  seqinfo.ini
  img1.mp4
  gt/gt.txt
```

For example, from this bundle's root:

```bash
python tools/build_courtdyn_qa.py --seq_dir data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train/Q1_top_0-30 --out_dir results/courtdyn/seq_Q1_top_0-30 --frame_dir data/courtdyn/frames_Q1_top_0-30 --win_s 2 --hop_s 1 --n_frames 4 --max_per_family 140
python tools/calibrate_court_homography.py --seq_dir data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train/Q1_top_0-30 --out_dir results/courtdyn/homography --n_frames 9
python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant ruler
python tools/build_courtdyn_ruler.py --seq Q1_top_0-30 --variant pxunit
```

The main generator takes `--seq_dir`, not `--seq`. `Q4_side_480-510` uses
`results/courtdyn/` and `data/courtdyn/frames` directly; other clips use
`results/courtdyn/seq_<name>` and `data/courtdyn/frames_<name>`.
Check the per-source recorded parameters rather than assuming all pools use defaults.
The generators' deterministic cap selection, chronological four-frame windows,
trajectory smoothing, and integration stride are retained in `engine/dynamics_qa.py`.
Random shuffling generally changes a time-connected path; only full reversal
necessarily preserves the summed path length.

## Homography provenance

The calibration implementation is automatic, followed by visual quality checks:
sample 9 video frames, locate court color/white lines with fixed-seed RANSAC,
take temporal median corners, and map to an assumed 28 by 15 m court. Frame sampling
is `round(linspace(5, frame_count - 6, 9))` in OpenCV's zero-based index, recorded
as index plus one in the H JSON. The seven local original H files include corners,
per-frame calibration diagnostics and a build timestamp; they are not supplied here.
Hash records identify the original files. Missing calibration source code was an
export omission, not evidence that the original H files were manually annotated.

Recalibration requires exact source videos and compatible OpenCV/NumPy/SciPy behavior.
Inspect the generated corner/rectification images. A `built` timestamp changes the
whole-file hash even if H is numerically identical. Pool manifests likewise contain
timestamps; a digest is provenance, not proof that regeneration is byte identical.
Compare QA payload hashes where applicable and compare H numerically separately.
The original Windows QA files use CRLF newlines. The hash manifest records the
newline convention; an otherwise equal JSON serialization with LF newlines has a
different byte digest. Match the recorded serialization before comparing SHA-256.

## Remaining gaps

- Source data, all released-model outputs with locally reconstructed alignment,
  and appropriate model weights are still needed to rescore real results. Public
  sanitized predictions cannot be fed directly to the original scorer without
  reconstructing and verifying the withheld question/answer alignment.
- No original training dependency lockfile is established by this repair. Observed
  versions document the audit host only; GPU training, inference, seeds and processor
  tensor sizes require separate verification.
- Human-M3 reconstruction needs its separately obtained data and camera calibration;
  this repair has not validated the full Human-M3 flow.
- Frozen aggregate JSON/Markdown files do not replace per-item evaluation inputs.
  The synthetic smoke test proves that the exported code starts and handles a small
  independent example, not that all published numerical claims have been reproduced.
- The online archive was not read or smoke-tested in this repair. Any public update
  must begin with the actual sanitized public version and undergo its own audit.

## Local audit performed on 2026-09-12

The exported code passed 17 smoke checks from a clean working directory using
Python `-I` and the installed CPU dependencies. Separately, the author-side audit
used the locally obtained Q1_top_0-30 source data and unredacted predictions:

- Both video and tracking-file hashes matched the recorded source manifest.
- All 681 main QA records rebuilt with the original CRLF serialization and matched
  the original pool SHA-256.
- One actual four-frame sample was decoded and rendered; all four output frame
  SHA-256 digests matched the original frames.
- Automatic nine-frame recalibration recovered the original H matrix with maximum
  absolute element difference 0.0 and the same recorded frame numbers. The temporary
  H/overlay outputs were discarded after the audit.
- Six CourtDyn-native cells (full/ruler/pxunit, speed/path) were rescored from the
  locally available predictions and reconstructed ground truth; all reported cell
  fields matched the frozen findings within 1e-10.

The author-side check can be run with `tools/audit_paper2_release_local_inputs.py
--workspace <authorized-local-workspace> --bundle <this-bundle> --report <report.json>
--calibrate`. It reads original local inputs that are deliberately absent here.
These checks are limited to one sequence and the audit environment. They do not
certify the separately sanitized online version or resolve the remaining gaps above.

The updated native-SFT builder defaults to event-group exclusion and writes a newly
named training pool. Frozen aggregate results and existing adapter weights still
refer to the original training run; preparing a corrected split does not retrain
that adapter or validate its event-independent performance. No generated training
pool is included in this bundle.
