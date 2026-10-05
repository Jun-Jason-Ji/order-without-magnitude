#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run train/train_qlora.py unchanged, but keep every checkpoint and optionally stop early.

train_qlora.py is byte-frozen (its sha256 is recorded in the 2026-09-12 execution plan and
audits), so the two changes the round-2 second-backbone run needs are applied from outside:

  * every checkpoint is kept (TrainingArguments.save_total_limit -> None), so the step-70k
    checkpoint of a 4-epoch schedule exists for each epoch k;
  * --stop_at_step <n> (consumed here, not passed on) ends training once global_step reaches n.  The learning-rate
    schedule is still the full num_train_epochs one, so stopping at step 70k yields exactly the
    epoch-k checkpoint of the 4-epoch run.  Rerunning with a larger STOP_AT_STEP resumes from the
    last checkpoint (train_qlora.py resumes automatically), again on the same schedule.

Usage: train/train_qlora.py arguments, plus optionally --stop_at_step <n>.
"""
import runpy
import sys
from pathlib import Path

import transformers
# The original classes are patched in place, not subclassed: Trainer pickles the
# TrainingArguments object into every checkpoint (training_args.bin), and a subclass
# defined here would not be importable under runpy's temporary __main__.
from transformers import Trainer, TrainerCallback, TrainingArguments

argv = sys.argv[1:]
STOP = 0
if "--stop_at_step" in argv:
    i = argv.index("--stop_at_step")
    STOP = int(argv[i + 1])
    del argv[i:i + 2]

_post_init = TrainingArguments.__post_init__


def _keep_all_checkpoints(self):
    self.save_total_limit = None
    _post_init(self)


TrainingArguments.__post_init__ = _keep_all_checkpoints


class StopAtStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if STOP and state.global_step >= STOP:
            control.should_save = True
            control.should_training_stop = True
        return control


_trainer_init = Trainer.__init__


def _init_with_stop(self, *args, callbacks=None, **kwargs):
    callbacks = list(callbacks or [])
    if STOP:
        callbacks.append(StopAtStep())
    _trainer_init(self, *args, callbacks=callbacks, **kwargs)


Trainer.__init__ = _init_with_stop

TARGET = Path(__file__).resolve().parent / "train_qlora.py"
sys.argv = [str(TARGET)] + argv
runpy.run_path(str(TARGET), run_name="__main__")
