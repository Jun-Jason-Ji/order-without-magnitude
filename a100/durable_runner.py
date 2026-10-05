#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""a100.runner.Runner with durable state and a progress page.

Runner.save() replaces state.json with tmp + os.replace, but without fsync (so after a power cut
the rename can survive while the data does not).  Here every save is an atomic, fsynced write,
and PROGRESS.md at the repository root is regenerated after each state change (throttled), so
after a restart one file says what is done, running and pending.  Runner.py itself is unchanged
(local queues import it).
"""
from __future__ import annotations

import json
import os
import time

from a100 import durable as D
from a100.runner import Runner, now


class DurableRunner(Runner):
    def __init__(self, *args, **kwargs):
        if "require_idle_gpu" not in kwargs:
            kwargs["require_idle_gpu"] = os.environ.get("A100_REQUIRE_IDLE_GPU", "1") != "0"
        super().__init__(*args, **kwargs)
        self._progress_at = 0.0

    def save(self):
        self.state["updated"] = now()
        D.atomic_write_text(self.state_path,
                            json.dumps(self.state, indent=1, ensure_ascii=False) + "\n")
        if time.time() - self._progress_at > 15:
            self._progress_at = time.time()
            try:
                from a100 import progress
                progress.write()
            except Exception as error:                     # noqa: BLE001
                print(f"[progress] not written: {error!r}", flush=True)
