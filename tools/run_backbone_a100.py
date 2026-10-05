#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fourth-backbone replication on the A100 (backbones and order: a100/backbone4.py).

Rules: results/courtdyn/backbone4_a100/PREREGISTRATION.json (written before any run).
The round-2 protocol, run exactly as tools/run_backbone_internvl.py runs the third backbone,
reusing the round-2 stage builders; per backbone:

  base cells (explicit arms, both test clips) + base text probe
  -> dev-set stopping rule (v3 adapter, epochs 1..4 of the 4-epoch schedule) -> stopping.json
  -> mixed-unit adapter to k* -> test cells (explicit, first-frame x4, in-template) + probes

What differs from the round-2/third-backbone runners is machine-level only:
  * model paths come from $COURTDYN_MODELS (a100/download_models.py fills it at pinned
    revisions); nothing Windows-specific;
  * training runs through train/train_durable.py (checkpoint every 10 steps, fsync, validation
    and quarantine of torn checkpoints, automatic resume); a training stage counts as done only
    when its epoch checkpoint carries the DURABLE_OK marker and an intact adapter file;
  * evaluation/probes run through tools/family_launch.py (image policy pinned per family,
    atomic json, fsync per prediction, torn-tail repair);
  * state.json and stopping.json are written atomically and fsynced; PROGRESS.md is refreshed.

Rerun after any interruption: finished stages are skipped, interrupted ones resume.

Usage
  python tools/run_backbone_a100.py --backbone pixtral_12b --dry-run
  python tools/run_backbone_a100.py --backbone pixtral_12b --run
  python tools/run_backbone_a100.py --backbone pixtral_12b --smoke   # check_all.py uses this
Exit codes: 0 done, 1 some stages failed, 3 conditional arm not run (no weights/access).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from a100 import backbone4 as B4                         # noqa: E402
from a100 import config as C                             # noqa: E402
from a100 import durable as D                            # noqa: E402
from a100.durable_runner import DurableRunner            # noqa: E402
from a100.runner import Stage, python_cmd                # noqa: E402
import run_backbone_replication as R1                    # noqa: E402
import run_backbone_round2 as R2                         # noqa: E402

OUT = ROOT / "results" / "courtdyn" / "backbone4_a100"
PREREG = OUT / "PREREGISTRATION.json"

MODELS = B4.MODELS                   # key -> dict(repo, revision, wrapper, gated, conditional, ...)
SAVE_EVERY = 10                      # optimizer steps between resumable checkpoints
LAUNCHER = ROOT / "tools" / "family_launch.py"
DURABLE = ROOT / "train" / "train_durable.py"


def model_dir(key: str) -> Path:
    repo, rev = MODELS[key]["repo"], MODELS[key]["revision"]
    return C.MODEL_ROOT / "hf" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev


def launched_cmd(script, *args):
    """python_cmd, with evaluation/probe scripts entered through the family launcher."""
    if Path(str(script)).name in ("run_backbone_replication.py", "run_unit_probe.py"):
        return python_cmd(LAUNCHER, script, *args)
    return python_cmd(script, *args)


def checkpoint_done(ckpt: Path) -> bool:
    """Complete and intact: marker written after fsync, sizes as recorded, adapter readable."""
    marker = ckpt / "DURABLE_OK.json"
    adapter = ckpt / "adapter_model.safetensors"
    if not (marker.is_file() and adapter.is_file()):
        return False
    try:
        files = json.loads(marker.read_text(encoding="utf-8"))["files"]
        if any((ckpt / n).stat().st_size != s for n, s in files.items()):
            return False
        sys.path.insert(0, str(ROOT / "train"))
        from train_durable import _safetensors_ok
        return _safetensors_ok(adapter)
    except Exception:                                    # noqa: BLE001
        return False


def train_stage(key, pool, k):
    """R2.train_stage, entered through the durable wrapper with a checkpoint every 10 steps."""
    a = C.TRAIN_ARGS
    ckpt = R2.checkpoint(key, pool, k)
    cmd = python_cmd(
        DURABLE, "--family", MODELS[key]["wrapper"], "--keep-every", R2.STEPS_PER_EPOCH,
        "--model_path", R1.BACKBONES[key], "--qa_json", R1.POOLS[pool],
        "--img_root", C.DATA, "--output_dir", R2.adapter_root(key, pool),
        "--num_samples", a["num_samples"], "--allow_base_init",
        "--budget_slice", a["budget_slice"], "--num_train_epochs", R2.MAX_EPOCHS,
        "--max_steps", -1, "--seed", R2.SEED, "--lr", a["lr"], "--grad_accum", a["grad_accum"],
        "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
        "--save_steps", SAVE_EVERY, "--stop_at_step", R2.STEPS_PER_EPOCH * k)
    return Stage(name=f"train_{key}_{pool}_epoch{k}", cmd=cmd,
                 verify=lambda: checkpoint_done(ckpt),
                 log_name=f"train_{key}_{pool}.log",
                 needs=[R1.POOLS[pool], Path(R1.BACKBONES[key]) / "config.json"],
                 stall_minutes=60, max_restarts=6,
                 note=f"4-epoch schedule, stop at step {R2.STEPS_PER_EPOCH * k}")


def configure(key):
    """Point the round-2 builders at this backbone and this output directory."""
    R1.BACKBONES[key] = str(model_dir(key))
    R2.OUT = OUT
    R2.python_cmd = launched_cmd
    R2.train_stage = train_stage


def base_stages(key):
    model = R1.BACKBONES[key]
    stages, ev = [], OUT / key / "eval"
    for seq in R1.SEQS:
        out = ev / f"base_{seq}_full"
        stages.append(Stage(
            name=f"eval_{key}_{out.name}",
            cmd=launched_cmd(ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                             "--model", model, "--label", "base", "--seq", seq,
                             "--arms", "m_height", "cm_height", "px_height",
                             "--frame-mode", "full", "--output", out),
            done_file=out / "summary.json", log_name=f"eval_{key}_{out.name}.log",
            needs=[Path(model) / "config.json"]))
    runs = OUT / key / "unit_probe"
    stages.append(Stage(
        name=f"probe_{key}_base",
        cmd=launched_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", "base",
                         "--adapter", "none", "--model-path", model, "--out", runs),
        done_file=runs / "base" / "summary.json", log_name=f"probe_{key}_base.log",
        needs=[Path(model) / "config.json"]))
    return stages


def choose_k(key, runner) -> int:
    """R2.choose_k with the decision written atomically and fsynced (same rule, same record)."""
    decision_path = OUT / key / "stopping.json"
    if decision_path.is_file():
        d = json.loads(decision_path.read_text(encoding="utf-8"))
        runner.log(f"{key}: stopping decision already recorded: k* = {d['k_star']}")
        return d["k_star"]
    trail = []
    for k in range(1, R2.MAX_EPOCHS + 1):
        if runner.run([train_stage(key, "v3", k)]) != 0:
            raise SystemExit(f"{key}: training to epoch {k} failed")
        out = OUT / key / "dev" / f"v3_epoch{k}"
        if runner.run([R2.eval_stage(key, f"v3_epoch{k}", R2.checkpoint(key, "v3", k),
                                     R2.DEV_SEQ, ["legacy_m"], "full", out,
                                     R2.DEV_INPUTS)]) != 0:
            raise SystemExit(f"{key}: dev evaluation at epoch {k} failed")
        ok, fams = R2.dev_passes(out / "summary.json")
        trail.append({"epoch": k, "passes": ok, "families": fams})
        runner.log(f"{key}: dev epoch {k}: {'PASS' if ok else 'fail'} {fams}")
        if ok:
            break
    k_star = next((t["epoch"] for t in trail if t["passes"]), R2.MAX_EPOCHS)
    decision = {"backbone": key, "k_star": k_star,
                "degenerate_at_cap": not any(t["passes"] for t in trail),
                "rule": f"parse >= {R2.MIN_PARSE} and >= {R2.MIN_DISTINCT} distinct per family on dev",
                "trail": trail, "decided_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_cells_evaluated_before_decision": False}
    D.atomic_write_json(decision_path, decision)
    return k_star


def run_backbone(key, dry_run=False) -> int:
    configure(key)
    runner = DurableRunner(f"backbone4-{key}", OUT / key / "state.json", OUT / key / "logs")
    if dry_run:
        return runner.run(base_stages(key) + [train_stage(key, "v3", k) for k in (1, 2, 3, 4)]
                          + [train_stage(key, "mixunit", 4)] + R2.test_stages(key, 4),
                          dry_run=True)
    failed = runner.run(base_stages(key)) != 0
    if failed:
        runner.log(f"{key}: some base stages failed; see state.json")
    k = choose_k(key, runner)
    if runner.run([train_stage(key, "mixunit", k)]) != 0:
        raise SystemExit(f"{key}: mixed-unit training failed")
    if runner.run(R2.test_stages(key, k)) != 0:
        runner.log(f"{key}: some test stages failed; see state.json")
        failed = True
    runner.log(f"{key}: complete" + (" (with failures)" if failed else ""))
    return 1 if failed else 0


# ------------------------------------------------------------------ smoke
def smoke(key) -> int:
    """Disclosed smoke test (PREREGISTRATION (e)); writes only under $A100_WORK/smoke/<key>.

    1. a 2-step training run of the v3 pool into a scratch directory, interrupted after step 1
       and resumed (exercises the durable checkpoint/resume path on this GPU);
    2. greedy answers of the bare backbone to 4 dev items (a 4-item copy of the dev cell);
    3. one text-probe item answered with the 2-step scratch adapter (exercises adapter loading).
    Checks: rc 0, a resume from step 1 was logged, run_config.json parses, answers parse as
    numbers.  Nothing here is a dev statistic or a test cell.
    """
    configure(key)
    work = Path(os.environ.get("A100_WORK") or ROOT) / "smoke" / key
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    import subprocess

    def run(cmd, log):
        with open(work / log, "w", encoding="utf-8") as handle:
            print("$ " + " ".join(str(c) for c in cmd), file=handle, flush=True)
            rc = subprocess.call([str(c) for c in cmd], stdout=handle, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), env=dict(os.environ, PYTHONUNBUFFERED="1"))
        return rc

    results, a = {}, C.TRAIN_ARGS
    train_out = work / "train"
    base = [DURABLE, "--family", MODELS[key]["wrapper"], "--keep-every", 1000,
            "--model_path", R1.BACKBONES[key], "--qa_json", R1.POOLS["v3"],
            "--img_root", C.DATA, "--output_dir", train_out,
            "--num_samples", 32, "--allow_base_init", "--budget_slice", a["budget_slice"],
            "--num_train_epochs", 1, "--max_steps", 2, "--seed", R2.SEED, "--lr", a["lr"],
            "--grad_accum", 2, "--lora_r", a["lora_r"], "--max_pixels", a["max_pixels"],
            "--save_steps", 1]
    rc1 = run(python_cmd(*base, "--stop_at_step", 1), "train_part1.log")
    rc2 = run(python_cmd(*base), "train_part2.log")
    resumes = [json.loads(l) for l in (train_out / "durable_resume_log.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()] if (train_out / "durable_resume_log.jsonl").is_file() else []
    results["train"] = dict(ok=rc1 == 0 and rc2 == 0
                            and (train_out / "training_completion.json").is_file()
                            and any(r.get("resume_from_step") == 1 for r in resumes),
                            rc=[rc1, rc2], resumes=resumes)

    # 4-item copy of the dev cell, hashes recomputed (the dev set itself is untouched).
    inputs = work / "dev4"
    manifest = json.loads((R2.DEV_INPUTS / "manifest.json").read_text(encoding="utf-8"))
    cell = dict(manifest["cells"][0])
    rows = json.loads((R2.DEV_INPUTS / cell["file"]).read_text(encoding="utf-8"))[:4]
    (inputs / Path(cell["file"]).parent).mkdir(parents=True)
    body = json.dumps(rows, ensure_ascii=False, indent=1).encode("utf-8")
    (inputs / cell["file"]).write_bytes(body)
    import hashlib
    cell.update(items=len(rows), sha256=hashlib.sha256(body).hexdigest())
    manifest["cells"] = [cell]
    (inputs / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ev = work / "eval_base_dev4"
    rc = run(launched_cmd(ROOT / "tools" / "run_backbone_replication.py", "eval-cell",
                          "--model", R1.BACKBONES[key], "--label", "smoke_base",
                          "--seq", cell["seq"], "--arms", cell["arm"], "--frame-mode", "full",
                          "--output", ev, "--inputs", inputs), "eval_base_dev4.log")
    answers = []
    if (ev / f"{cell['arm']}.jsonl").is_file():
        answers = [json.loads(l) for l in (ev / f"{cell['arm']}.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
    try:
        json.loads((ev / "run_config.json").read_text(encoding="utf-8"))
        config_ok = True
    except Exception:                                     # noqa: BLE001
        config_ok = False
    parsed = [r["prediction"] for r in answers if r.get("prediction") is not None]
    results["eval_base_dev4"] = dict(ok=rc == 0 and config_ok and len(answers) == 4
                                     and len(parsed) >= 3,
                                     rc=rc, answers=[(r.get("raw_answer") or "")[:40]
                                                     for r in answers],
                                     references=[r.get("reference_value") for r in answers])
    probe_out = work / "probe"
    rc = run(launched_cmd(ROOT / "tools" / "run_unit_probe.py", "--label", "smoke_adapter",
                          "--adapter", train_out, "--model-path", R1.BACKBONES[key],
                          "--out", probe_out, "--limit", 1), "probe_adapter.log")
    rows = []
    path = probe_out / "smoke_adapter" / "results.jsonl"
    if path.is_file():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    results["probe_adapter"] = dict(ok=rc == 0 and len(rows) == 1, rc=rc,
                                    answers=[(r.get("raw_answer") or "")[:40] for r in rows])
    ok = all(v["ok"] for v in results.values())
    D.atomic_write_json(work / "smoke_result.json", dict(backbone=key, ok=ok, **results))
    print(json.dumps(dict(backbone=key, ok=ok, **results), indent=1, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=sorted(MODELS))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not PREREG.is_file():
        raise SystemExit("no PREREGISTRATION.json: record the rules before running")
    if not (R2.DEV_INPUTS / "manifest.json").is_file():
        raise SystemExit(f"no dev inputs at {R2.DEV_INPUTS}")
    m = MODELS[args.backbone]
    if m["conditional"] and not (model_dir(args.backbone) / "config.json").is_file():
        # Preregistered as conditional: no access (or not downloaded) -> recorded as not run.
        note = OUT / args.backbone / "NOT_RUN.json"
        if not args.dry_run:
            D.atomic_write_json(note, {
                "backbone": args.backbone, "repo": m["repo"], "revision": m["revision"],
                "reason": "conditional arm: weights not available (gated access not granted or "
                          "not downloaded) when the run reached it; no substitution",
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"[backbone4] {args.backbone}: conditional arm not run (no weights at "
              f"{model_dir(args.backbone)}); recorded in {note}", flush=True)
        return 3
    if args.smoke:
        return smoke(args.backbone)
    named = json.loads(PREREG.read_text(encoding="utf-8")).get("revisions", {})
    if list(named.get(args.backbone, []))[:2] != [m["repo"], m["revision"]]:
        raise SystemExit(f"{args.backbone} at {m['revision'][:8]} is not named in "
                         f"{PREREG}; preregister it before running")
    (OUT / args.backbone / "NOT_RUN.json").unlink(missing_ok=True)
    return run_backbone(args.backbone, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
