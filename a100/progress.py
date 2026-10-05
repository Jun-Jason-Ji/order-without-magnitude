#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write PROGRESS.md: where the A100 run stands, readable after any restart.

Reads every runner state.json of the A100 run (backbone 4 per model, T36), the durable
checkpoints of the backbone-4 adapters and the phase marker run_all.sh keeps, and writes
<repo>/PROGRESS.md atomically.  Called by the runner after state changes and by run_all.sh
between phases; `python -m a100.progress` prints it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from a100 import config as C                 # noqa: E402
from a100 import backbone4 as BB             # noqa: E402
from a100 import durable as D                # noqa: E402

B4 = C.ROOT / "results" / "courtdyn" / "backbone4_a100"
# Rough A100-40GB hours per backbone (base cells, 4 dev epochs, mixed-unit, test cells; from
# a100/backbone4.py) and for T36, scaled from the local round-2 timings (3B: ~4.2 h per 280-step
# pool on an 8 GB laptop GPU).
ETA_HOURS = {**{k: v["hours"] for k, v in BB.MODELS.items()}, "t36": 7}
EXPECTED_STAGES = 30          # per backbone, when k* = 4 (fewer if the rule stops earlier)


def work_dir() -> Path:
    return Path(os.environ.get("A100_WORK") or C.ROOT)


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def counts(state):
    stages = (state or {}).get("stages", {})
    c = {"done": 0, "running": 0, "failed": 0, "other": 0}
    running = []
    for name, s in stages.items():
        st = s.get("status")
        if st == "done":
            c["done"] += 1
        elif st in ("running", "queued_gpuq", "waiting_gpu"):
            c["running"] += 1
            running.append(f"{name} (since {s.get('started', '?')}, attempt {s.get('restarts', 0) + 1})")
        elif st == "failed":
            c["failed"] += 1
        else:
            c["other"] += 1
    return c, running


def last_checkpoints(key):
    out = []
    for pool in ("v3", "mixunit"):
        root = C.ROOT / "models" / f"courtdyn-{key}-{pool}-r2ep4-s42"
        steps = [int(m.group(1)) for p in (root.iterdir() if root.is_dir() else [])
                 if (m := re.match(r"^checkpoint-(\d+)$", p.name))
                 and (p / "DURABLE_OK.json").is_file()]
        if steps:
            out.append(f"{pool}: step {max(steps)}/280")
    return ", ".join(out) or "none yet"


def render() -> str:
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    phase = load(work_dir() / ".phase.json") or {}
    lines = [f"# A100 run progress", "",
             f"Updated {t} UTC.  Phase: **{phase.get('phase', '?')}** "
             f"(since {phase.get('since', '?')}).", "",
             "| part | done | running | failed | k* | last durable checkpoint | est. hours |",
             "|---|---|---|---|---|---|---|"]
    remaining = 0.0
    running_all = []
    for key in BB.active():
        state = load(B4 / key / "state.json")
        c, running = counts(state)
        stop = load(B4 / key / "stopping.json")
        if (B4 / key / "NOT_RUN.json").is_file() and not state:
            lines.append(f"| {key} | - | - | - | - | not run: conditional arm, no access yet "
                         f"(rerun run_all.sh once granted) | {ETA_HOURS[key]} |")
            continue
        kstar = "-" if not stop else f"{stop['k_star']}{' (degenerate at cap)' if stop['degenerate_at_cap'] else ''}"
        finished = bool(state and state.get("finished")) and c["running"] == 0 and \
            (B4 / key / "unit_probe" / "mixunit" / "summary.json").is_file()
        frac = 1.0 if finished else min(c["done"] / EXPECTED_STAGES, 0.95)
        remaining += ETA_HOURS[key] * (1 - frac)
        lines.append(f"| {key} | {c['done']} | {c['running']} | {c['failed']} | {kstar} | "
                     f"{last_checkpoints(key)} | {ETA_HOURS[key]} |")
        running_all += running
    t36 = load(C.OUT / "backbones" / "state.json")
    c, running = counts(t36)
    plan = len((t36 or {}).get("plan", [])) or 36
    remaining += ETA_HOURS["t36"] * (1 - min(c["done"] / plan, 1.0))
    lines.append(f"| T36 zero-shot | {c['done']}/{plan} | {c['running']} | {c['failed']} | - | - | "
                 f"{ETA_HOURS['t36']} |")
    running_all += running
    lines += ["", "Running now: " + ("; ".join(running_all) if running_all else "nothing"),
              "", f"Rough time left: ~{remaining:.0f} GPU-hours (estimate, scaled from local runs).",
              "", "After a restart run `bash a100/resume.sh` (or run_all.sh again): finished "
              "stages are skipped, training resumes from its last durable checkpoint, "
              "evaluation cells from their last saved prediction."]
    return "\n".join(lines) + "\n"


def write():
    D.atomic_write_text(C.ROOT / "PROGRESS.md", render())


def set_phase(name):
    D.atomic_write_json(work_dir() / ".phase.json",
                        {"phase": name, "since": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--phase":
        set_phase(sys.argv[2])
    else:
        write()
    print(render())
