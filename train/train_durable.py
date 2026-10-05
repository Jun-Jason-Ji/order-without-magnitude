#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crash-safe training entry point for machines that can be restarted at any moment.

Wraps a family entry point (train_qlora_epochs.py, train_qlora_internvl.py or
train_qlora_family.py -- all of which end in the byte-frozen train/train_qlora.py) and adds
durability only; the optimisation is unchanged.

Before training
  * every checkpoint-N directory in --output_dir is checked, newest first: the adapter
    safetensors header and size, optimizer/scheduler/rng torch.load on CPU, trainer_state.json.
    An unreadable newest checkpoint is moved to _quarantine/checkpoint-N.corrupt-<time> and the
    next one is tried, so Trainer.resume_from_checkpoint picks the newest *valid* checkpoint.
  * an unreadable training_protocol.json (torn by a crash) is moved aside; train_qlora.py
    rewrites the identical receipt.
  * the resume (from which step, what was quarantined) is appended to
    durable_resume_log.jsonl in the adapter directory.
During training (a Trainer callback, on every save)
  * the new checkpoint's files are fsynced, then DURABLE_OK.json is written atomically;
  * intermediate checkpoints are pruned: every multiple of --keep-every (the epoch
    checkpoints, protocol artefacts) is kept forever, of the rest only the newest two.
After training
  * the adapter directory's top-level files (final adapter, training_completion.json) are fsynced.

Resume equivalence: Trainer restores optimizer, LR scheduler, gradient scaler, RNG states and
the data position (it skips the batches already consumed in the current epoch, with the same
seeded sampler), and the schedule is still the full --num_train_epochs one, so a resumed run
follows the same 280-step cosine schedule as an uninterrupted one.  GPU kernels are not bitwise
deterministic, so "equivalent", not "bitwise identical".

Usage: python train/train_durable.py --family {epochs,internvl,family,idefics3,llama32} [--keep-every 70]
       <train_qlora_epochs.py arguments...>
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a100 import durable as D                      # noqa: E402

# epochs: Qwen-VL / SmolVLM2 (frozen families); internvl: the third-backbone wrapper;
# family: any family registered in a100/family_policy.py (Gemma3, DeepSeek-VL hybrid, Pixtral)
# idefics3: Idefics3-8B through the frozen SmolVLM2 family (connector-name alias)
WRAPPERS = {"epochs": "train_qlora_epochs.py", "internvl": "train_qlora_internvl.py",
            "family": "train_qlora_family.py", "idefics3": "train_qlora_idefics3.py",
            "llama32": "train_qlora_llama32.py"}   # Llama 3.2 Vision: family + a100/family_policy_llama32
CKPT = re.compile(r"^checkpoint-(\d+)$")
MARKER = "DURABLE_OK.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pop(argv, flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def _safetensors_ok(path: Path) -> bool:
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(8)
        if len(head) != 8:
            return False
        n = int.from_bytes(head, "little")
        if n <= 0 or 8 + n > size:
            return False
        header = json.loads(handle.read(n).decode("utf-8"))
    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"),
              default=0)
    return 8 + n + end == size


def checkpoint_problem(ckpt: Path):
    """None if the checkpoint can be resumed from, else a short reason."""
    try:
        state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
        if int(state.get("global_step", -1)) != int(CKPT.match(ckpt.name).group(1)):
            return "trainer_state global_step does not match the directory name"
        adapter = ckpt / "adapter_model.safetensors"
        if not adapter.is_file() or not _safetensors_ok(adapter):
            return "adapter_model.safetensors missing or truncated"
        json.loads((ckpt / "adapter_config.json").read_text(encoding="utf-8"))
        import torch
        for name in ("optimizer.pt", "scheduler.pt"):
            if not (ckpt / name).is_file():
                return f"{name} missing"
            torch.load(ckpt / name, map_location="cpu", weights_only=False)
        rng = sorted(ckpt.glob("rng_state*.pth"))
        if not rng:
            return "rng_state missing"
        for path in rng:
            torch.load(path, map_location="cpu", weights_only=False)
        marker = ckpt / MARKER
        if marker.is_file():
            files = json.loads(marker.read_text(encoding="utf-8"))["files"]
            for name, size in files.items():
                if (ckpt / name).stat().st_size != size:
                    return f"{name} size differs from {MARKER}"
    except Exception as error:                              # noqa: BLE001
        return f"unreadable: {error!r}"[:300]
    return None


def checkpoints(out: Path):
    found = [(int(CKPT.match(p.name).group(1)), p) for p in out.iterdir()
             if p.is_dir() and CKPT.match(p.name)] if out.is_dir() else []
    return sorted(found)


def quarantine(out: Path, ckpt: Path, reason: str):
    q = out / "_quarantine"
    q.mkdir(exist_ok=True)
    target = q / f"{ckpt.name}.corrupt-{int(time.time())}"
    os.replace(ckpt, target)
    D.fsync_dir(out)
    D.atomic_write_json(target / "QUARANTINE_REASON.json", {"reason": reason, "at": now()})
    return str(target)


def prepare(out: Path):
    """Quarantine unreadable newest checkpoints; return (resume step or None, quarantined)."""
    quarantined = []
    receipt = out / "training_protocol.json"
    if receipt.is_file():
        try:
            json.loads(receipt.read_text(encoding="utf-8"))
        except ValueError:
            target = receipt.with_name(f"training_protocol.json.corrupt-{int(time.time())}")
            os.replace(receipt, target)
            quarantined.append(str(target))
    for step, ckpt in reversed(checkpoints(out)):
        problem = checkpoint_problem(ckpt)
        if problem is None:
            if not (ckpt / MARKER).is_file():
                write_marker(ckpt)
            return step, quarantined
        print(f"[durable] {ckpt.name}: {problem}; quarantined", flush=True)
        quarantined.append(quarantine(out, ckpt, problem))
    return None, quarantined


def write_marker(ckpt: Path):
    D.fsync_tree(ckpt)
    files = {p.relative_to(ckpt).as_posix(): p.stat().st_size
             for p in sorted(ckpt.rglob("*")) if p.is_file() and p.name != MARKER}
    D.atomic_write_json(ckpt / MARKER, {"files": files, "at": now()})


def prune(out: Path, keep_every: int):
    others = [(s, p) for s, p in checkpoints(out) if s % keep_every]
    for _, path in others[:-2]:
        shutil.rmtree(path, ignore_errors=True)
    D.fsync_dir(out)


def install_callback(out: Path, keep_every: int):
    from transformers import Trainer, TrainerCallback

    class DurableCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt.is_dir():
                write_marker(ckpt)
                prune(Path(args.output_dir), keep_every)
                print(f"[durable] {ckpt.name} synced", flush=True)
            return control

    original = Trainer.__init__

    def init(self, *args, callbacks=None, **kwargs):
        callbacks = list(callbacks or []) + [DurableCheckpoints()]
        original(self, *args, callbacks=callbacks, **kwargs)

    Trainer.__init__ = init


def main():
    argv = sys.argv[1:]
    family = _pop(argv, "--family", "epochs")
    keep_every = int(_pop(argv, "--keep-every", "70"))
    if family not in WRAPPERS:
        raise SystemExit(f"--family must be one of {sorted(WRAPPERS)}")
    out = Path(argv[argv.index("--output_dir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    step, quarantined = prepare(out)
    with open(out / "durable_resume_log.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps({"at": now(), "resume_from_step": step, "quarantined": quarantined,
                              "stop_at_step": argv[argv.index("--stop_at_step") + 1]
                              if "--stop_at_step" in argv else None,
                              "host": os.uname().nodename if hasattr(os, "uname") else None})
                  + "\n")
        log.flush()
        os.fsync(log.fileno())
    print(f"[durable] start: resume from step {step}; quarantined {quarantined}", flush=True)
    install_callback(out, keep_every)
    target = Path(__file__).resolve().parent / WRAPPERS[family]
    sys.argv = [str(target)] + argv
    import runpy
    runpy.run_path(str(target), run_name="__main__")
    for path in out.iterdir():
        if path.is_file():
            try:
                fd = os.open(str(path), os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass
    D.fsync_dir(out)


if __name__ == "__main__":
    main()
