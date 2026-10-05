#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the CourtDyn unit-control arms for one (model, adapter, clip) cell.

This is a portable generalisation of tools/run_courtdyn_revision_controls.py.
That script is referenced by frozen execution plans through its file hash and
is hard-wired to one adapter, NF4 loading and the Windows paths baked into the
prepared manifest, so it is left untouched; this module keeps its protocol
(same prompts, same strict parsing, same scale-aware T-MRA, same greedy
decoding and token budget) while adding what the A100 sweeps need:

  * optional adapter, so public checkpoints can run the identical arms
  * NF4 or bf16, chosen explicitly and recorded, because they are not the
    same arm -- NF4 reproduces every published CourtDyn cell
  * families beyond Qwen-VL (a100/model_families.py)
  * frame roots resolved on this machine instead of the manifest's absolute path

Examples
  python -m a100.eval_controls --list
  python -m a100.eval_controls --model qwen35-4b --adapter-dir models/... \\
      --seq Q2_top_480-510 --arms m_height cm_height px_height \\
      --output results/a100/... --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                     # noqa: E402
from a100 import model_families as MF            # noqa: E402

NUMBER = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


# --------------------------------------------------------------------------
# scoring -- identical definitions to the frozen protocol
# --------------------------------------------------------------------------
def parse(raw):
    """Strict whole-answer numeric parse; anything else is an invalid answer."""
    raw = (raw or "").strip()
    return float(raw) if NUMBER.fullmatch(raw) and math.isfinite(float(raw)) else None


def scalar_score(prediction, target, tolerance, floor):
    relative = (abs(prediction - target) - tolerance) / max(abs(target), floor)
    return 10.0 * sum(relative < 1 - (0.5 + 0.05 * i) for i in range(10))


def spearman(xs, ys):
    """Spearman rho with average ranks; None when either side is constant."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None                     # constant output: undefined, not zero
    return sxy / math.sqrt(sxx * syy)


def score_rows(rows):
    """Per-family summary. Invalid parses score zero in the all-item score."""
    result = {}
    for category in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == category]
        valid = [r for r in group if r["prediction"] is not None]
        scores = [scalar_score(r["prediction"], r["reference_value"],
                               r["score_tolerance"], r["score_floor"])
                  for r in valid]
        preds = [r["prediction"] for r in valid]
        refs = [r["reference_value"] for r in valid]
        result[category] = dict(
            n=len(group), parsed=len(valid),
            parse_rate=len(valid) / len(group) if group else 0.0,
            tmra_all_items=sum(scores) / len(group) if group else None,
            tmra_parsed=sum(scores) / len(valid) if valid else None,
            spearman=spearman(preds, refs) if len(valid) >= 3 else None,
            n_distinct=len(set(preds)),
            prediction_median=_median(preds), reference_median=_median(refs),
            # Red line from the 2026-09 analysis: a ratio of ~1 only means
            # "no conversion happened" when the model's answers actually vary.
            # Two or fewer distinct answers is a degenerate cell, flagged here
            # so no downstream table can quietly treat it as evidence.
            degenerate=len(set(preds)) <= 2,
        )
    return result


def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_cells(seq, arms, inputs=None):
    """(manifest cell, items) per arm, with the frame root rebound locally."""
    inputs = Path(inputs or C.INPUTS)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf8"))
    root = C.image_root(seq)
    out = []
    for arm in arms:
        cell = next((c for c in manifest["cells"]
                     if c["seq"] == seq and c["arm"] == arm), None)
        if cell is None:
            raise SystemExit(f"prepared manifest has no cell {seq}/{arm}")
        path = inputs / cell["file"].replace("\\", "/")
        if sha256(path) != cell["sha256"]:
            raise SystemExit(f"prepared input hash does not match manifest: {path}")
        rows = json.loads(path.read_text(encoding="utf8"))
        cell = dict(cell, manifest_image_root=cell["image_root"],
                    image_root=str(root))
        out.append((cell, rows))
    return manifest, out


def verify_frames(cells):
    missing = []
    for cell, rows in cells:
        root = Path(cell["image_root"])
        for image_id in sorted({i for row in rows for i in row["image_ids"]}):
            if not (root / image_id).is_file():
                missing.append(str(root / image_id))
                if len(missing) > 5:
                    return missing
    return missing


def validate_saved_rows(path, items):
    """A resume may append only after an intact prefix of these exact inputs."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf8")
    if content and not content.endswith("\n"):
        raise SystemExit(f"resume requires newline-terminated records: {path}")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(rows) > len(items):
        raise SystemExit(f"too many saved predictions: {path}")
    for index, row in enumerate(rows):
        expected = dict(index=index, category=items[index]["category"],
                        **items[index]["meta"])
        if any(row.get(k) != v for k, v in expected.items()):
            raise SystemExit(f"resume input identity differs at {path}:{index + 1}")
        if row.get("prediction") != parse(row["raw_answer"]):
            raise SystemExit(f"saved parsing differs at {path}:{index + 1}")
    return rows


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def gpu_snapshot():
    try:
        q = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"]
        line = subprocess.run(q, capture_output=True, text=True,
                              timeout=30, check=True).stdout.strip()
        name, used, total = [p.strip() for p in line.split(",")[:3]]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                               "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.split()
        return dict(name=name, memory_used_mib=int(used),
                    memory_total_mib=int(total), compute_processes=len(apps))
    except Exception as error:                            # noqa: BLE001
        return dict(error=repr(error))


def build_infer(model_path, precision, adapter=None, max_pixels=C.MAX_PIXELS,
                max_new_tokens=C.MAX_NEW_TOKENS):
    """Return (infer, receipt). Qwen/SmolVLM2 delegate to the frozen loader."""
    import torch
    from PIL import Image
    from eval.run_bench import GENERATION_SUFFIX

    spec, _ = MF.inspect(model_path)
    receipt = dict(model_family=spec.name, precision=precision,
                   generation_suffix=GENERATION_SUFFIX,
                   max_new_tokens=max_new_tokens, do_sample=False)

    if spec.name in MF.BASE_FAMILY_NAMES and precision == "nf4":
        # Byte-identical path to every published CourtDyn cell.
        from eval.run_bench import load_model
        receipt["loader"] = "eval.run_bench.load_model"
        receipt["processor_policy"] = MF.processor_kwargs(spec, max_pixels)
        infer = load_model(str(model_path), load_4bit=True, adapter=adapter,
                           max_pixels=max_pixels, max_new_tokens=max_new_tokens)
        return infer, receipt

    from transformers import AutoProcessor, AutoModelForImageTextToText
    receipt["loader"] = "a100.eval_controls.build_infer"
    proc_kwargs = dict(trust_remote_code=spec.trust_remote_code)
    proc_kwargs.update(MF.processor_kwargs(spec, max_pixels))
    processor = AutoProcessor.from_pretrained(str(model_path), **proc_kwargs)
    receipt["processor_policy"] = MF.processor_kwargs(spec, max_pixels)
    receipt["processor_observed"] = MF.observed_policy(processor)

    kwargs = dict(dtype=torch.bfloat16, device_map="auto",
                  trust_remote_code=spec.trust_remote_code)
    if precision == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif precision != "bf16":
        raise SystemExit(f"unknown precision {precision!r}")
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), **kwargs)
    if spec.name == "smolvlm2":
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        if not any("lora_" in n for n, _ in model.named_parameters()):
            raise SystemExit(f"adapter {adapter} attached no LoRA tensors")
    model.eval()
    receipt["dtype"] = str(next(model.parameters()).dtype)

    def infer(image_paths, question):
        paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
        images = [Image.open(p).convert("RGB") for p in paths]
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question + GENERATION_SUFFIX})
        msgs = [{"role": "user", "content": content}]
        tmpl = dict(add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
        try:
            enc = processor.apply_chat_template(msgs, **tmpl).to(model.device)
        except Exception:                                  # noqa: BLE001
            rendered = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            enc = processor(text=rendered, images=images or None,
                            return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        gen = out[0][enc["input_ids"].shape[1]:]
        return processor.decode(gen, skip_special_tokens=True).strip()

    return infer, receipt


def extract_answer(raw):
    """Last non-empty line, 'Answer:' prefix removed -- as in eval/run_bench."""
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return re.sub(r"(?i)^(final\s+)?answer\s*[:：]\s*", "", lines[-1]).strip().strip("*").strip()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def run_cell(model_path, precision, seq, arms, output, adapter=None,
             frame_mode="full", arithmetic=False, resume=False, inputs=None,
             label=None, progress_every=20, infer_factory=None):
    """Run one cell. `infer_factory` exists only for a100/selftest.py, which
    exercises the whole pipeline on a CPU box with a deterministic stub."""
    output = Path(output)
    manifest, cells = load_cells(seq, arms, inputs)
    missing = verify_frames(cells)
    if missing:
        raise SystemExit("missing frames, first few:\n  " + "\n  ".join(missing[:5]))
    for _, rows in cells:                    # the references must score 100
        for row in rows:
            m = row["meta"]
            assert scalar_score(float(row["answer"]), m["reference_value"],
                                m["score_tolerance"], m["score_floor"]) == 100

    inputs_dir = Path(inputs or C.INPUTS)
    arithmetic_items = (json.loads((inputs_dir / "arithmetic.json").read_text(encoding="utf8"))
                        if arithmetic else [])
    config = dict(
        protocol=manifest["summary"]["protocol"] + "+a100-v1",
        label=label, seq=seq, arms=list(arms), frame_mode=frame_mode,
        model_path=str(Path(model_path).resolve()),
        adapter=str(Path(adapter).resolve()) if adapter else None,
        precision=precision, max_pixels=C.MAX_PIXELS,
        max_new_tokens=C.MAX_NEW_TOKENS, decoding="greedy", strict_parse=True,
        input_manifest_sha256=sha256(inputs_dir / "manifest.json"),
        input_cells=[dict(arm=c["arm"], file=c["file"], sha256=c["sha256"],
                          manifest_image_root=c["manifest_image_root"],
                          resolved_image_root=c["image_root"]) for c, _ in cells],
        arithmetic=bool(arithmetic),
        arithmetic_sha256=sha256(inputs_dir / "arithmetic.json") if arithmetic_items else None,
        score="per-item scale-aware T-MRA; invalid parses score zero over all items",
        code_sha256={p: sha256(C.ROOT / p) for p in
                     ("a100/eval_controls.py", "a100/model_families.py",
                      "a100/config.py", "eval/run_bench.py")},
    )
    if adapter:
        config["adapter_sha256"] = sha256(Path(adapter) / "adapter_model.safetensors")

    # The sweeps always pass --resume so that an interrupted cell continues.  On a
    # cell that never started there is no run_config.json to resume from; that is
    # a fresh run, not an error.  (Treating it as an error made every new cell in
    # the 2026-09-25 local seed sweep fail.)  A directory that holds anything but
    # nothing-to-resume is still refused.
    if resume and not (output / "run_config.json").is_file():
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"{output} has files but no run_config.json; "
                             "refusing to resume or overwrite")
        resume = False
    elif output.exists() and not resume and any(output.iterdir()):
        raise SystemExit(f"{output} exists; pass --resume or choose a new directory")
    if resume:
        previous = json.loads((output / "run_config.json").read_text(encoding="utf8"))
        # These vary between machines and attempts; everything that defines the
        # experiment -- prompts, hashes, model, adapter, precision, arms -- must
        # match exactly or the partial run is not resumable.
        drop = ("started_at", "runtime", "gpu", "loader")
        if {k: v for k, v in previous.items() if k not in drop} != config:
            raise SystemExit("resume configuration differs; use a new output directory")
        if (output / "summary.json").exists():
            raise SystemExit("this run already has a summary; verify it instead of rerunning")
    else:
        output.mkdir(parents=True)

    factory = infer_factory or build_infer
    infer, receipt = factory(model_path, precision, adapter=adapter)
    config = dict(config, runtime=_runtime_versions(), gpu=gpu_snapshot(),
                  loader=receipt, started_at=datetime.now(timezone.utc).isoformat())
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf8")

    summaries = {}
    for cell, items in cells:
        result_path = output / (cell["arm"] + ".jsonl")
        rows = validate_saved_rows(result_path, items) if resume else []
        mode = "a" if rows else "w"
        root = Path(cell["image_root"])
        with result_path.open(mode, encoding="utf8") as handle:
            for index, item in enumerate(items[len(rows):], start=len(rows)):
                paths = [str(root / i) for i in item["image_ids"]]
                if frame_mode == "static4":
                    paths = [paths[0]] * len(paths)
                raw = infer(paths, item["question"])
                answer = raw.strip()
                prediction = parse(answer)
                if prediction is None:
                    # The frozen protocol parses the whole answer. Keep that as
                    # the scored value, but also record what the lenient
                    # last-line rule would have given, so a parse-rate argument
                    # can be answered with data instead of a rerun.
                    lenient = parse(extract_answer(raw))
                else:
                    lenient = prediction
                row = dict(index=index, category=item["category"], **item["meta"],
                           prediction=prediction, lenient_prediction=lenient,
                           raw_answer=raw)
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % progress_every == 0 or index + 1 == len(items):
                    print(datetime.now(timezone.utc).isoformat(), label or "", seq,
                          cell["arm"], f"{index + 1}/{len(items)}", flush=True)
        summaries[cell["arm"]] = score_rows(rows)

    if arithmetic_items:
        path = output / "arithmetic_results.json"
        if resume and path.exists():
            observations = json.loads(path.read_text(encoding="utf8"))
        else:
            observations = []
            for item in arithmetic_items:
                raw = infer([], item["question"])
                pred = parse(raw.strip())
                target = float(item["answer"])
                observations.append(dict(
                    question=item["question"], target=target, raw_answer=raw,
                    prediction=pred,
                    correct=pred is not None and abs(pred - target) < 0.05))
            path.write_text(json.dumps(observations, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf8")
        summaries["arithmetic"] = dict(n=len(observations),
                                       correct=sum(r["correct"] for r in observations))

    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return summaries


def _runtime_versions():
    import importlib.metadata as md
    out = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            out[package] = md.version(package)
        except Exception:                                  # noqa: BLE001
            out[package] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--model", help="registry name from a100/config.py")
    ap.add_argument("--model-path", help="explicit weights directory, overrides --model")
    ap.add_argument("--precision", choices=["nf4", "bf16"])
    ap.add_argument("--adapter-dir")
    ap.add_argument("--seq", default="Q2_top_480-510")
    ap.add_argument("--arms", nargs="+", default=C.ARMS_THREE)
    ap.add_argument("--frame-mode", choices=["full", "static4"], default="full")
    ap.add_argument("--arithmetic", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--output")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="without this the call only validates inputs on CPU")
    args = ap.parse_args()

    if args.list:
        print(json.dumps(dict(paths=C.describe(), backbones={
            name: dict(spec, resolved=str(C.backbone_path(name) or ""))
            for name, spec in C.BACKBONES.items()}), indent=2, ensure_ascii=False))
        return 0

    if args.model_path:
        model_path, precision = Path(args.model_path), args.precision or "bf16"
    elif args.model:
        spec = C.BACKBONES[args.model]
        resolved = C.backbone_path(args.model)
        model_path = resolved or spec["hf"]
        precision = args.precision or spec["precision"]
    else:
        ap.error("pass --model or --model-path")

    if len(args.arms) != len(set(args.arms)):
        ap.error("--arms must not repeat an arm; outputs would collide")

    if not args.run:
        manifest, cells = load_cells(args.seq, args.arms)
        missing = verify_frames(cells)
        print(json.dumps(dict(
            status="VALIDATED_NOT_RUN", model=str(model_path), precision=precision,
            adapter=args.adapter_dir, seq=args.seq, arms=args.arms,
            frame_mode=args.frame_mode,
            visual_items=sum(len(rows) for _, rows in cells),
            missing_frames=missing[:5], gpu=gpu_snapshot()),
            indent=2, ensure_ascii=False))
        return 1 if missing else 0

    if not args.output:
        ap.error("--run requires --output")
    summaries = run_cell(model_path, precision, args.seq, args.arms, args.output,
                         adapter=args.adapter_dir, frame_mode=args.frame_mode,
                         arithmetic=args.arithmetic, resume=args.resume,
                         label=args.label)
    print(json.dumps(dict(status="COMPLETE", output=args.output,
                          summary=summaries), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
