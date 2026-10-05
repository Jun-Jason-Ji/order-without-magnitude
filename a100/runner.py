#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A small, portable stage runner for the single-A100 experiment sweeps.

tools/queue_guard.py cannot be reused here: it reads Windows commit memory
through GlobalMemoryStatusEx and kills process trees with taskkill. This runner
keeps the properties that mattered on the 8 GB box -- skip a stage whose
artefact already exists, retry a crash, kill a stage whose log has gone quiet,
persist state so an interrupted sweep resumes -- with nothing platform
specific, and it never waits on an unrelated queue.

It is a library; a100/exp_seeds.py and a100/exp_backbones.py build the stage
lists. Run either with --dry-run first: that prints the plan and validates
every input without touching the GPU.
"""
from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# Machine-wide GPU queue (~/gpuq/gpuq.py, or GPUQ_HOME): when present, every GPU stage
# takes an exclusive, crash-safe slot there before it starts, so jobs from other projects
# never share the card with it.  Optional: on a machine without gpuq (the A100) the runner
# falls back to its own idle-GPU polling.
_GPUQ_HOME = os.environ.get("GPUQ_HOME") or os.path.join(os.path.expanduser("~"), "gpuq")
_gpuq = None
if os.path.isfile(os.path.join(_GPUQ_HOME, "gpuq.py")):
    try:
        sys.path.insert(0, _GPUQ_HOME)
        import gpuq as _gpuq  # noqa: E402
    except Exception:                                  # noqa: BLE001
        _gpuq = None


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Stage:
    name: str
    cmd: Sequence[str]
    done_file: Path | None = None
    log_name: str | None = None
    needs: Sequence[Path] = field(default_factory=tuple)
    kind: str = "gpu"
    stall_minutes: int = 45
    max_restarts: int = 4
    # A stage may declare its own completion test when an artefact's presence
    # is not enough (a training run must also match its protocol receipt).
    verify: Callable[[], bool] | None = None
    note: str = ""


class Runner:
    def __init__(self, name, state_path, log_dir, require_idle_gpu=False):
        self.name = name
        self.state_path = Path(state_path)
        self.log_dir = Path(log_dir)
        self.require_idle_gpu = require_idle_gpu
        self.state = {"sweep": name, "started": now(), "finished": None,
                      "host": os.environ.get("HOSTNAME") or os.uname().nodename
                      if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
                      "stages": {}}
        if self.state_path.is_file():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf8"))
                if previous.get("sweep") == name:
                    self.state["stages"] = previous.get("stages", {})
                    self.state["resumed_from"] = previous.get("started")
            except (OSError, ValueError):
                pass

    # ---------------------------------------------------------------- state
    def log(self, message):
        print(f"[{self.name} {now()}] {message}", flush=True)

    def save(self):
        self.state["updated"] = now()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf8")
        os.replace(tmp, self.state_path)

    def set(self, stage, **fields):
        self.state["stages"].setdefault(stage, {}).update(fields)
        self.save()

    # ------------------------------------------------------------------ gpu
    @staticmethod
    def gpu_state():
        try:
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30, check=True).stdout.split()
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True).stdout.split()
            return dict(memory_used_mib=int(used[0]), compute_processes=len(apps),
                        idle=(not apps and int(used[0]) < 500))
        except Exception as error:                        # noqa: BLE001
            # No driver, or nvidia-smi absent. Treat as "unknown", not "idle":
            # on the 8 GB box a missing driver once let every stage launch into
            # a CUDA-init failure loop.
            return dict(error=repr(error), idle=False)

    def wait_for_gpu(self, stage, poll=30, announce_every=300):
        if not self.require_idle_gpu:
            return
        announced = 0
        while True:
            snapshot = self.gpu_state()
            if snapshot.get("idle"):
                return
            if time.monotonic() - announced >= announce_every:
                self.log(f"{stage}: waiting for an idle GPU {snapshot}")
                announced = time.monotonic()
            self.set(stage, status="waiting_gpu", gpu=snapshot)
            time.sleep(poll)

    # ------------------------------------------------------------- stages
    @staticmethod
    def _kill(process):
        try:
            if os.name == "nt":
                subprocess.call(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:                                  # noqa: BLE001
            process.kill()
        try:
            process.wait(timeout=60)
        except Exception:                                  # noqa: BLE001
            pass

    def complete(self, stage: Stage) -> bool:
        if stage.verify is not None:
            try:
                return bool(stage.verify())
            except Exception:                              # noqa: BLE001
                return False
        return bool(stage.done_file and Path(stage.done_file).exists())

    def run_stage(self, stage: Stage) -> bool:
        if self.complete(stage):
            self.log(f"{stage.name}: artefact present, skipping")
            self.set(stage.name, status="done", note="already complete")
            return True
        for need in stage.needs:
            if not Path(need).exists():
                self.log(f"{stage.name}: missing prerequisite {need}")
                self.set(stage.name, status="skipped", note=f"missing {need}")
                return False

        log_path = self.log_dir / (stage.log_name or f"{stage.name}.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1",
                   TOKENIZERS_PARALLELISM="false")
        restarts = 0
        while True:
            slot = contextlib.ExitStack()
            if stage.kind == "gpu":
                if _gpuq is not None:
                    self.set(stage.name, status="queued_gpuq")
                    if getattr(self, "_gpuq_batch", None) is None:
                        # reserve the card for this whole sweep: other projects' jobs wait
                        # until this process exits, instead of slipping in between stages
                        self._gpuq_batch = contextlib.ExitStack()
                        self._gpuq_batch.enter_context(_gpuq.batch(self.name, log=self.log))
                    slot.enter_context(_gpuq.slot(
                        f"{self.name}:{stage.name}", " ".join(str(c) for c in stage.cmd),
                        log=self.log, group=self.name))
                else:
                    self.wait_for_gpu(stage.name)
            started = time.time()
            self.set(stage.name, status="running", restarts=restarts,
                     started=now(), log=str(log_path), cmd=[str(c) for c in stage.cmd])
            self.log(f"{stage.name}: start (attempt {restarts + 1})")
            popen_kwargs = {}
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True
            with log_path.open("a" if restarts else "w", encoding="utf8") as handle:
                handle.write(f"\n===== {stage.name} attempt {restarts + 1} {now()} =====\n")
                handle.write(" ".join(str(c) for c in stage.cmd) + "\n\n")
                handle.flush()
                process = subprocess.Popen([str(c) for c in stage.cmd],
                                           stdout=handle, stderr=subprocess.STDOUT,
                                           cwd=str(Path(__file__).resolve().parent.parent),
                                           env=env, **popen_kwargs)
                stalled = False
                # "quiet" is counted in awake polling time, not wall-clock: a laptop that sleeps or
                # hibernates mid-stage (2026-09-30: a 6 h hibernation killed a healthy training run
                # as "quiet for 379 min") adds no polls, so it no longer counts as a stall.
                last_mtime, quiet = None, 0.0
                while process.poll() is None:
                    time.sleep(30)
                    if stage.stall_minutes:
                        try:
                            mtime = log_path.stat().st_mtime
                        except OSError:
                            mtime = None
                        if mtime != last_mtime:
                            last_mtime, quiet = mtime, 0.0
                        else:
                            quiet += 0.5
                        if quiet > stage.stall_minutes:
                            self.log(f"{stage.name}: log quiet for {quiet:.0f} min, killing")
                            self._kill(process)
                            stalled = True
                            break
            slot.close()                       # release the GPU before any retry wait
            elapsed = round(time.time() - started, 1)
            if not stalled and process.returncode == 0 and self.complete(stage):
                self.set(stage.name, status="done", rc=0, ended=now(),
                         seconds=elapsed)
                self.log(f"{stage.name}: done in {elapsed / 60:.1f} min")
                return True
            restarts += 1
            reason = "stalled" if stalled else f"rc={process.returncode}"
            if restarts > stage.max_restarts:
                self.set(stage.name, status="failed", rc=process.returncode,
                         ended=now(), note=f"{reason}; gave up after {restarts} attempts")
                self.log(f"{stage.name}: FAILED ({reason}); see {log_path}")
                return False
            self.log(f"{stage.name}: {reason}, retry {restarts}/{stage.max_restarts} in 30 s")
            time.sleep(30)

    def run(self, stages: Sequence[Stage], dry_run=False):
        self.state["plan"] = [dict(name=s.name, kind=s.kind, note=s.note,
                                   cmd=[str(c) for c in s.cmd]) for s in stages]
        pending = [s for s in stages if not self.complete(s)]
        self.log(f"{len(stages)} stages, {len(pending)} pending, "
                 f"{len(stages) - len(pending)} already complete")
        self.save()
        if dry_run:
            for stage in stages:
                mark = "done" if self.complete(stage) else "todo"
                missing = [str(n) for n in stage.needs if not Path(n).exists()]
                self.log(f"  [{mark}] {stage.name}"
                         + (f"  MISSING: {missing}" if missing else ""))
            self.log("dry run: nothing was executed")
            return 0
        failures = []
        for stage in stages:
            try:
                if not self.run_stage(stage):
                    failures.append(stage.name)
            except KeyboardInterrupt:
                self.set(stage.name, status="interrupted", ended=now())
                self.log("interrupted; state saved, rerun to resume")
                raise
            except Exception as error:                     # noqa: BLE001
                self.set(stage.name, status="failed", note=repr(error))
                self.log(f"{stage.name}: runner-level error {error!r}")
                failures.append(stage.name)
        self.state["finished"] = now()
        self.state["failed_stages"] = failures
        self.save()
        self.log("finished; failures: " + (", ".join(failures) or "none"))
        return 1 if failures else 0


def python_cmd(*args):
    """A command that re-enters this repository's Python with -m or a script."""
    return [sys.executable, "-u", *[str(a) for a in args]]
