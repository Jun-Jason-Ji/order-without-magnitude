#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the text-only unit-conversion probe on one checkpoint (plan D6, step 1).

The probe carries no images, so this runs on the 8 GB local card in minutes --
it does not wait for the A100.  Everything that defines the measurement is
taken from the frozen evaluation path rather than reimplemented here:

  loader   a100.eval_controls.build_infer -> eval.run_bench.load_model (nf4)
  suffix   eval.run_bench.GENERATION_SUFFIX
  parse    a100.eval_controls.parse  (strict whole-answer numeric)
  decode   greedy, do_sample=False, MAX_NEW_TOKENS

so a number produced here is produced the same way as every published cell.
The one thing that is deliberately *not* inherited is image handling: passing
an empty image list is what makes this a text-only probe.

Correctness keeps the published absolute criterion (|error| < 0.05).  The
diagnostic that actually carries the argument is log10(prediction / target),
recorded per item and classified by analyze_unit_probe.py.

Resumable: results stream to results.jsonl and a rerun continues after the last
complete line, refusing to continue if the configuration changed.

Usage
  python tools/run_unit_probe.py --label base
  python tools/run_unit_probe.py --label native --adapter models/courtdyn-native-sft
  python tools/run_unit_probe.py --all          # base + the three seed-42 adapters
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROBE = ROOT / "results" / "courtdyn" / "unit_probe_v2.json"
DEFAULT_OUT = ROOT / "results" / "courtdyn" / "unit_probe_v2_runs"

# label -> adapter directory relative to the repository root (None = base model)
ARMS = {
    "base": None,
    "native": "models/courtdyn-native-sft",
    "v1": "models/courtdyn-revision-v1-eventholdout-s42",
    "v3": "models/courtdyn-revision-v3-eventholdout-s42",
    # M1 mixed-unit control (tools/run_m1_mixunit.py): v3 pool, half cm targets.
    "mixunit": "models/courtdyn-mixunit-v3-s42",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_base_model() -> str:
    """The reproduction backbone. Same environment variable as the A100 suite."""
    for variable in ("QWEN35_4B", "COURTDYN_BASE_MODEL"):
        value = os.environ.get(variable)
        if value:
            return value
    from a100 import config as C
    spec = C.MODELS.get("qwen35-4b") if hasattr(C, "MODELS") else None
    if isinstance(spec, dict) and spec.get("hf"):
        return spec["hf"]
    raise SystemExit(
        "set QWEN35_4B to the Qwen3.5-4B weights (the reproduction backbone)")


def load_done(path: Path):
    """Return the item_ids already written, tolerating a torn final line."""
    if not path.is_file():
        return [], 0
    rows, complete_bytes = [], 0
    with open(path, "rb") as handle:
        for raw in handle:
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except Exception:                              # noqa: BLE001
                break                                      # torn tail
            complete_bytes += len(raw)
    return rows, complete_bytes


def run_arm(label, adapter, probe_path, out_root, base_model, limit=None,
            published_only=False):
    from a100.eval_controls import build_infer, parse
    from a100 import config as C

    items = json.loads(Path(probe_path).read_text(encoding="utf-8"))
    if published_only:
        items = [row for row in items if row["meta"]["published_subset"]]
    if limit:
        items = items[:limit]

    output = Path(out_root) / label
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    config_path = output / "run_config.json"

    adapter_path = str((ROOT / adapter).resolve()) if adapter else None
    config = {
        "protocol": "courtdyn-unit-probe-v2",
        "label": label,
        "probe": str(Path(probe_path).resolve()),
        "probe_sha256": sha256_file(Path(probe_path)),
        "n_items": len(items),
        "published_only": bool(published_only),
        "model_path": str(base_model),
        "adapter": adapter_path,
        "precision": "nf4",
        "decoding": "greedy",
        "max_new_tokens": C.MAX_NEW_TOKENS,
        "text_only": True,
        "correct_abs_tolerance": 0.05,
        "code_sha256": {
            "tools/run_unit_probe.py": sha256_file(Path(__file__)),
            "tools/build_unit_probe.py": sha256_file(ROOT / "tools" / "build_unit_probe.py"),
            "a100/eval_controls.py": sha256_file(ROOT / "a100" / "eval_controls.py"),
        },
    }
    if adapter_path:
        config["adapter_sha256"] = sha256_file(
            Path(adapter_path) / "adapter_model.safetensors")

    done_rows, complete_bytes = load_done(results_path)
    done_ids = {row["item_id"] for row in done_rows}
    if done_rows and all(row["item_id"] in done_ids for row in items):
        # Nothing to append, so nothing is being resumed: the configuration guard
        # below protects appends, not finished arms.  The run_config.json on disk
        # still records the code and inputs these rows were produced with.
        print(f"[{label}] complete ({len(done_rows)} rows), not reloading", flush=True)
        return summarize(label, output, done_rows)
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        varying = ("started_at", "finished_at", "runtime", "gpu", "loader",
                   "generation_suffix")
        a = {k: v for k, v in previous.items() if k not in varying}
        b = {k: v for k, v in config.items() if k not in varying}
        if a != b:
            differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            raise SystemExit(
                f"{output}: configuration changed since the partial run "
                f"({', '.join(differing)}); the partial results are not resumable")
        if done_rows:
            # Drop a torn final line rather than appending after it.
            with open(results_path, "r+b") as handle:
                handle.truncate(complete_bytes)
    todo = [row for row in items if row["item_id"] not in done_ids]
    print(f"[{label}] {len(done_ids)} done, {len(todo)} to run", flush=True)
    if not todo:
        return summarize(label, output, done_rows)

    started = time.time()
    infer, receipt = build_infer(base_model, "nf4", adapter=adapter_path)
    config["loader"] = receipt.get("loader")
    config["generation_suffix"] = receipt.get("generation_suffix")
    config["started_at"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    rows = list(done_rows)
    with open(results_path, "a", encoding="utf-8") as handle:
        for index, item in enumerate(todo, 1):
            raw = infer([], item["question"])          # [] -> text only
            prediction = parse((raw or "").strip())
            target = float(item["meta"]["target"])
            row = {
                "item_id": item["item_id"],
                "question": item["question"],
                "target": target,
                "raw_answer": raw,
                "prediction": prediction,
                "correct": prediction is not None and abs(prediction - target) < 0.05,
                "meta": item["meta"],
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(todo):
                elapsed = time.time() - started
                print(f"[{label}] {index}/{len(todo)}  "
                      f"{elapsed:.0f}s  {elapsed / index:.2f}s/item", flush=True)
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return summarize(label, output, rows)


def summarize(label, output, rows):
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    unparsed = sum(1 for r in rows if r["prediction"] is None)
    summary = {"label": label, "n": n, "correct": correct,
               "accuracy": correct / n if n else 0.0, "unparsed": unparsed}
    (Path(output) / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{label}] {correct}/{n} correct ({100 * correct / max(n, 1):.1f}%), "
          f"{unparsed} unparsed", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=str(DEFAULT_PROBE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--label",
                    help=f"a known arm ({', '.join(sorted(ARMS))}) or, with --adapter, "
                         "any new label")
    ap.add_argument("--adapter", help="adapter for --label; 'none' for the bare backbone")
    ap.add_argument("--model-path",
                    help="backbone weights (default: $QWEN35_4B); used for the "
                         "second-backbone replication")
    ap.add_argument("--all", action="store_true", help="run every arm in turn")
    ap.add_argument("--limit", type=int, help="first N items only (smoke)")
    ap.add_argument("--published-only", action="store_true",
                    help="only the 12 published items: the regression gate")
    args = ap.parse_args()

    if not args.all and not args.label:
        ap.error("pass --label or --all")
    if args.label and args.label not in ARMS and not args.adapter:
        ap.error(f"unknown label {args.label!r}: pass --adapter (or --adapter none)")
    if args.all and args.model_path:
        ap.error("--all runs the Qwen3.5-4B arms; use --label/--adapter for another backbone")

    base_model = args.model_path or resolve_base_model()
    print(f"base model: {base_model}")
    if args.all:
        # One process per arm.  Loading the next checkpoint in the same process
        # leaves the previous 4-bit model resident, 8 GB overflows, accelerate
        # dispatches modules to the CPU and bitsandbytes refuses to load -- which
        # is exactly how the first --all run died after two arms.  A fresh
        # process per arm releases the card completely.  Finished arms resume
        # instantly without loading anything.
        import subprocess
        failed = []
        for label in sorted(ARMS):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--label", label,
                   "--probe", args.probe, "--out", args.out]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.published_only:
                cmd.append("--published-only")
            if subprocess.run(cmd).returncode != 0:
                failed.append(label)
        if failed:
            print(f"FAILED arms: {failed}")
            return 1
        return 0
    labels = [args.label]
    summaries = []
    for label in labels:
        adapter = args.adapter if (args.adapter and label == args.label) else ARMS[label]
        if adapter and adapter.lower() == "none":
            adapter = None
        out = args.out
        if args.published_only:
            out = str(Path(args.out).with_name(Path(args.out).name + "_published_gate"))
        summaries.append(run_arm(label, adapter, args.probe, out,
                                 base_model, limit=args.limit,
                                 published_only=args.published_only))
    print()
    for summary in summaries:
        print(f"  {summary['label']:>8}  {summary['correct']:>4}/{summary['n']}  "
              f"{100 * summary['accuracy']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
