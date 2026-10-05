#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crash-safe file primitives for runs on a machine that can be restarted at any moment.

Why this exists: on 2026-09-27 an unclean shutdown left a training resume file full-size but
unreadable, because it had been written with tmp + os.replace and no fsync -- the rename was
durable, the data was not.  Every file a resumable run depends on (checkpoints, state,
decisions, receipts, summaries) therefore goes through the helpers here:

  atomic_write_*   write tmp -> flush -> fsync(file) -> os.replace -> fsync(directory)
  fsync_tree       fsync every file of a directory tree, then the directories
  repair_jsonl     drop a torn final record (and any NUL padding) from an append-only log,
                   keeping the cut bytes next to it, so a resume starts from an intact prefix
  jsonable         convert library objects (transformers' SizeDict, dataclasses) for json

Standard library only; importable without torch.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path


def fsync_dir(directory) -> None:
    """Make a rename or a new entry in `directory` durable (no-op where unsupported)."""
    if os.name == "nt":
        return                      # Windows cannot open a directory for fsync
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path, obj, indent=1) -> None:
    atomic_write_text(path, json.dumps(jsonable(obj), indent=indent, ensure_ascii=False) + "\n")


def fsync_tree(root) -> None:
    root = Path(root)
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(os.path.join(directory, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)
        fsync_dir(directory)
    fsync_dir(root.parent)


def repair_jsonl(path) -> int:
    """Truncate `path` to its longest prefix of complete JSON lines.

    Returns the number of bytes removed.  The removed tail is kept as
    <name>.torn-<unix time> so nothing is silently lost.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    data = path.read_bytes()
    keep = 0
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        body = line.strip()
        if body:
            if b"\x00" in body:
                break
            try:
                json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                break
        keep += len(line)
    removed = len(data) - keep
    if removed:
        atomic_write_bytes(path.with_name(f"{path.name}.torn-{int(time.time())}"), data[keep:])
        with open(path, "r+b") as handle:
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())
    return removed


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {k: jsonable(v) for k, v in dataclasses.asdict(o).items()}
    if hasattr(o, "to_dict"):
        try:
            return jsonable(o.to_dict())
        except Exception:                                  # noqa: BLE001
            pass
    if hasattr(o, "keys") and hasattr(o, "__getitem__"):
        try:
            return {str(k): jsonable(o[k]) for k in o.keys()}
        except Exception:                                  # noqa: BLE001
            pass
    return repr(o)


class FsyncOnFlush:
    """Wrap a text/binary file so every flush() is also an fsync.

    Used for append-only prediction logs: a record the writer has flushed survives a power cut.
    """

    def __init__(self, handle):
        self._h = handle

    def flush(self):
        self._h.flush()
        try:
            os.fsync(self._h.fileno())
        except OSError:
            pass

    def close(self):
        try:
            self.flush()
        finally:
            self._h.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self):
        return iter(self._h)

    def __getattr__(self, name):
        return getattr(self._h, name)
