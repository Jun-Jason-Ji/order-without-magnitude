#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runner for the A100 extension run (results/courtdyn/backbone4_ext).

Same durable state writes as a100/durable_runner.py, but its progress page is PROGRESS_EXT.md
and it is rewritten on EVERY state change, without the 15-second throttle.  The throttle is why
the round-4 PROGRESS.md almost always said "Running now: nothing": a stage's "running" state was
saved seconds after the previous stage's "done" and the refresh was skipped.  Here the stage
start is written as it happens.  a100/durable_runner.py and a100/progress.py are unchanged.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from a100 import config as C
from a100 import durable as D
from a100.runner import Runner, now

EXT = C.ROOT / "results" / "courtdyn" / "backbone4_ext"
# part -> (state file, rough A100-40GB hours from the round-4 stage timings)
PARTS = {
    "seed43_gemma3_12b": (EXT / "gemma3_12b" / "state.json", 5.5),
    "seed43_pixtral_12b": (EXT / "pixtral_12b" / "state.json", 4.5),
    "rtc_pixtral_12b": (EXT / "rtc" / "pixtral_12b_state.json", 6.0),
    "rtc_gemma3_12b": (EXT / "rtc" / "gemma3_12b_state.json", 7.0),
}


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render() -> str:
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# A100 extension run progress", "",
             f"Updated {t} UTC.  Order: seed-43 Gemma3-12B -> seed-43 Pixtral-12B -> "
             "RTC Pixtral-12B -> RTC Gemma3-12B.", "",
             "| part | done | running | failed | planned | est. hours |", "|---|---|---|---|---|---|"]
    running_all, left = [], 0.0
    for part, (path, hours) in PARTS.items():
        st = _load(path) or {}
        stages = st.get("stages", {})
        plan = len(st.get("plan", [])) or 0
        done = sum(s.get("status") == "done" for s in stages.values())
        failed = sum(s.get("status") == "failed" for s in stages.values())
        running = [f"{n} (since {s.get('started', '?')}, attempt {s.get('restarts', 0) + 1})"
                   for n, s in stages.items() if s.get("status") in ("running", "waiting_gpu", "queued_gpuq")]
        running_all += running
        frac = 1.0 if st.get("finished") and not running else (done / plan if plan else 0.0)
        left += hours * (1 - min(frac, 1.0))
        lines.append(f"| {part} | {done} | {len(running)} | {failed} | {plan or '-'} | {hours} |")
    lines += ["", "Running now: " + ("; ".join(running_all) if running_all else "nothing (between stages or not started)"),
              "", f"Rough time left: ~{left:.1f} GPU-hours (from round-4 A100 stage timings).", "",
              "After a restart: `bash a100/resume_ext.sh` (finished stages are skipped; training resumes "
              "from its last durable checkpoint).  Results: `bash a100/pack_ext.sh`."]
    return "\n".join(lines) + "\n"


def write():
    D.atomic_write_text(C.ROOT / "PROGRESS_EXT.md", render())


class ExtRunner(Runner):
    def __init__(self, *args, **kwargs):
        if "require_idle_gpu" not in kwargs:
            kwargs["require_idle_gpu"] = os.environ.get("A100_REQUIRE_IDLE_GPU", "1") != "0"
        super().__init__(*args, **kwargs)

    def save(self):
        self.state["updated"] = now()
        D.atomic_write_text(self.state_path,
                            json.dumps(self.state, indent=1, ensure_ascii=False) + "\n")
        try:
            write()
        except Exception as error:                     # noqa: BLE001
            print(f"[progress-ext] not written: {error!r}", flush=True)


if __name__ == "__main__":
    write()
    print(render())
