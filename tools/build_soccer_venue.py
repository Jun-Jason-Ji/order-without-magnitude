#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a second-venue CourtDyn pool from SoccerNet-GSR (reviewer item 4).

Every CourtDyn result comes from one basketball game.  SoccerNet-GSR valid holds
58 broadcast sequences from three different matches, each frame labelled with the
players' image boxes *and* their metric pitch positions (bbox_pitch, centre-origin
105 x 68 m).  That is enough to pose exactly the CourtDyn motion questions on a
second sport, three new matches and a new camera regime, with metric ground truth
that does not come from a height heuristic.

What is built (all under a private directory -- see "Licence" below)
  <PRIVATE>/data/frames_<SEQ>/f{frame:04d}_r{track}.jpg   rendered 960x540 frames,
                                                          target in a red box
  <PRIVATE>/inputs/<SEQ>/{m,cm,px}_height.json            question pools
  <PRIVATE>/inputs/manifest.json                          a100.eval_controls manifest
  results/courtdyn/soccer_venue/build_receipt.json        aggregate receipt only

Design, fixed before any model sees these items
  sequences  one per match (game ids 2, 3, 5), the visible-action sequence with the
             least camera motion.  Camera motion is measured, not assumed: a
             per-frame image->pitch homography is fitted (RANSAC) to the labelled
             players' foot points, static pitch points in view at frame t are
             carried to frame t+50, and their median image displacement on the
             960x540 canvas is the window's camera motion.  Broadcast cameras pan
             and zoom, which CourtDyn's fixed cameras never do; the receipt reports
             the residual camera motion of every chosen sequence.
  window     2.0 s at 25 fps = frames t..t+50; the four shown frames are
             t, t+17, t+33, t+50 (CourtDyn: 60 frames at 29.97 fps, 20-frame steps).
  items      player/goalkeeper tracks present and fully inside the image in all 51
             frames, box height >= 40 px at 1080p.  140 path and 140 speed items per
             sequence, sampled separately with a sequence-seeded RNG, like CourtDyn.
  truth      path: length of the pitch-plane foot trajectory over all 51 frames,
             positions smoothed with a centred 5-frame moving average; speed:
             path / 2.0 s.  Pixel truth: the same on the rendered 960x540 canvas.
             K (px per m) is the per-sequence median of pixel/metre path ratios,
             as in CourtDyn.
  prompts    the exact CourtDyn explicit-coordinate template; only the sport words
             change ("soccer clip", "on this pitch is 1.80 m tall").
  scoring    CourtDyn's T and epsilon per family; x100 for cm, x K for px.

Licence
  SoccerNet-GSR is used under an NDA that forbids redistribution; the project's own
  soccer tooling never writes source frames or derivatives into the repository.
  Frames and pools therefore go to E:/datasets/SoccerNet/derived_private, and only
  aggregate statistics and hashes are written here.

Usage
  python tools/build_soccer_venue.py --scan        # camera-motion scan only
  python tools/build_soccer_venue.py               # scan, select, render, write pools
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GSR = Path(r"E:\datasets\SoccerNet\raw_private\SoccerNetGS\valid")
PRIVATE = Path(r"E:\datasets\SoccerNet\derived_private\courtdyn_venue")
RESULTS = ROOT / "results" / "courtdyn" / "soccer_venue"

FPS = 25.0
SPAN = 50                          # frames in a 2.0 s window
SHOWN = (0, 17, 33, 50)            # offsets of the four frames shown
HOP = 25                           # 1 s between window starts
SMOOTH = 5                         # centred moving average, frames
MIN_BOX_H = 40                     # px at 1080p
MARGIN = 10                        # px at 1080p
CANVAS = (960, 540)
ITEMS_PER_FAMILY = 140
HEIGHT_PRIOR_M = 1.80

TEMPLATE = ("The 4 frames are consecutive samples from one soccer clip spanning 2.0 "
            "seconds, in chronological order. Assume a typical player on this pitch "
            f"is {HEIGHT_PRIOR_M:.2f} m tall. Each supplied frame is 960 pixels wide and "
            "540 pixels high. Pixel answers refer to this original canvas before any "
            "resizing. Physical units describe floor-plane motion; pixels describe "
            "projected image-plane motion. ")
ASK = {
    "dynamics_path_player": "What is the path length of the player marked with the red "
                            "box over the whole clip, in {unit}? Output only the number, "
                            "one decimal place.",
    "dynamics_speed_player": "What is the average speed of the player marked with the "
                             "red box, in {unit} per second? Output only the number, one "
                             "decimal place.",
}
UNIT_WORD = {"m": "meters", "cm": "centimeters", "px": "pixels"}
TOL = {"dynamics_path_player": (0.5, 2.0), "dynamics_speed_player": (0.3, 1.0)}


# ------------------------------------------------------------------ load labels

def load_sequence(name):
    labels = json.loads((GSR / name / "Labels-GameState.json").read_text(encoding="utf-8"))
    frame_of = {str(img["image_id"]): int(Path(img["file_name"]).stem)
                for img in labels["images"]}
    tracks = defaultdict(dict)          # track -> frame -> record
    frames = defaultdict(list)          # frame -> [(foot_img, pitch)] for homography
    for a in labels["annotations"]:
        if a.get("category_id") not in (1, 2):          # player, goalkeeper
            continue
        bi, bp = a.get("bbox_image"), a.get("bbox_pitch")
        if not bi or not bp:
            continue
        f = frame_of[str(a["image_id"])]
        foot = (float(bi["x_center"]), float(bi["y"]) + float(bi["h"]))
        pitch = (float(bp["x_bottom_middle"]), float(bp["y_bottom_middle"]))
        box = (float(bi["x"]), float(bi["y"]), float(bi["w"]), float(bi["h"]))
        tracks[int(a["track_id"])][f] = {"foot": foot, "pitch": pitch, "box": box}
        frames[f].append((foot, pitch))
    return labels["info"], tracks, frames


def homographies(frames):
    """Per-frame image->pitch homography from the labelled foot points."""
    out = {}
    for f, pairs in frames.items():
        if len(pairs) < 6:
            continue
        img = np.float32([p[0] for p in pairs])
        pit = np.float32([p[1] for p in pairs])
        H, mask = cv2.findHomography(img, pit, cv2.RANSAC, 1.0)
        if H is not None and mask is not None and mask.sum() >= 6:
            out[f] = H
    return out


def camera_motion(H, t):
    """Median canvas-pixel displacement of static pitch points from t to t+SPAN."""
    if t not in H or t + SPAN not in H:
        return None
    xs, ys = np.meshgrid(np.linspace(0.3, 0.7, 3) * 1920, np.linspace(0.45, 0.8, 3) * 1080)
    grid = np.float32(np.stack([xs.ravel(), ys.ravel()], 1)).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(grid, H[t])
    try:
        back = cv2.perspectiveTransform(world, np.linalg.inv(H[t + SPAN]))
    except np.linalg.LinAlgError:
        return None
    d = np.linalg.norm((back - grid).reshape(-1, 2), axis=1) * (CANVAS[0] / 1920)
    return float(np.median(d))


def scan():
    info = json.loads((GSR / "sequences_info.json").read_text(encoding="utf-8"))
    rows = []
    for s in info["validation"]:
        meta, tracks, frames = load_sequence(s["name"])
        H = homographies(frames)
        motion = [m for t in range(1, 751 - SPAN, HOP)
                  if (m := camera_motion(H, t)) is not None]
        rows.append({"name": s["name"], "game_id": meta["game_id"],
                     "visibility": meta.get("visibility"),
                     "action": meta.get("action_class"),
                     "camera_motion_px_median": st.median(motion) if motion else None,
                     "windows_measured": len(motion), "frames_with_H": len(H)})
        print(f"{s['name']} game {meta['game_id']} {meta.get('visibility'):10s} "
              f"camera motion {rows[-1]['camera_motion_px_median']}", flush=True)
    return rows


def choose(rows):
    chosen = []
    for game in sorted({r["game_id"] for r in rows}):
        pool = [r for r in rows if r["game_id"] == game and r["visibility"] == "visible"
                and r["camera_motion_px_median"] is not None and r["windows_measured"] >= 20]
        pool.sort(key=lambda r: (r["camera_motion_px_median"], r["name"]))
        if pool:
            chosen.append(pool[0])
    return chosen


# ------------------------------------------------------------------ items

def smoothed_path(points):
    arr = np.asarray(points, dtype=float)
    k = SMOOTH // 2
    sm = np.array([arr[max(0, i - k):i + k + 1].mean(axis=0) for i in range(len(arr))])
    return float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())


def candidates(tracks):
    out = []
    for t in range(1, 751 - SPAN, HOP):
        span = range(t, t + SPAN + 1)
        for track, rec in tracks.items():
            if not all(f in rec for f in span):
                continue
            ok = True
            for f in span:
                x, y, w, h = rec[f]["box"]
                if h < MIN_BOX_H or x < MARGIN or y < MARGIN or \
                        x + w > 1920 - MARGIN or y + h > 1080 - MARGIN:
                    ok = False
                    break
            if not ok:
                continue
            path_m = smoothed_path([rec[f]["pitch"] for f in span])
            path_px = smoothed_path([(rec[f]["foot"][0] * CANVAS[0] / 1920,
                                      rec[f]["foot"][1] * CANVAS[1] / 1080) for f in span])
            out.append({"t": t, "track": track, "path_m": path_m, "path_px": path_px})
    return out


def render(name, item, tracks, out_dir):
    ids = []
    for off in SHOWN:
        f = item["t"] + off
        fname = f"f{f:04d}_r{item['track']}.jpg"
        ids.append(fname)
        target = out_dir / fname
        if target.is_file():
            continue
        img = cv2.imread(str(GSR / name / "img1" / f"{f:06d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"{name} frame {f}")
        x, y, w, h = tracks[item["track"]][f]["box"]
        cv2.rectangle(img, (int(round(x)), int(round(y))),
                      (int(round(x + w)), int(round(y + h))), (0, 0, 255), 4)
        img = cv2.resize(img, CANVAS, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return ids


def build_pools(name, K, items_by_family, tracks):
    frame_dir = PRIVATE / "data" / f"frames_{name}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    pools = {arm: [] for arm in ("m_height", "cm_height", "px_height")}
    for family, items in items_by_family.items():
        T, eps = TOL[family]
        for it in items:
            ids = render(name, it, tracks, frame_dir)
            metre = it["path_m"] if family == "dynamics_path_player" else it["path_m"] / 2.0
            pixel = it["path_px"] if family == "dynamics_path_player" else it["path_px"] / 2.0
            for arm, unit, value, scale, conv in (
                    ("m_height", "m", metre, 1.0, "GSR bbox_pitch"),
                    ("cm_height", "cm", metre * 100, 100.0, "GSR bbox_pitch"),
                    ("px_height", "px", pixel, K, "rendered pixels")):
                pools[arm].append({
                    "category": family,
                    "question": TEMPLATE + ASK[family].format(unit=UNIT_WORD[unit]),
                    "answer": f"{value:.1f}",
                    "image_ids": ids, "image_id": ids[0],
                    "meta": {"seq": name, "track": it["track"],
                             "window": [it["t"], it["t"] + SPAN], "fps": FPS,
                             "camera_motion_px": it.get("camera_motion_px"),
                             "arm": arm, "unit": unit,
                             "score_tolerance": round(T * scale, 6),
                             "score_floor": round(eps * scale, 6),
                             "reference_value": value,
                             "reference_convention": conv},
                })
    return pools


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="camera-motion scan only")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = scan()
    (RESULTS / "camera_motion_scan.json").write_text(json.dumps(rows, indent=1) + "\n",
                                                     encoding="utf-8")
    chosen = choose(rows)
    print("chosen:", [(r["name"], r["game_id"], round(r["camera_motion_px_median"], 1))
                      for r in chosen])
    if args.scan:
        return 0
    if len(chosen) != 3:
        print(f"ERROR: expected one sequence per match (3), got {len(chosen)}")
        return 1

    from a100.eval_controls import scalar_score        # noqa: E402
    manifest_cells, receipt_seqs = [], []
    for r in chosen:
        name = r["name"]
        _, tracks, frames = load_sequence(name)
        H = homographies(frames)
        cands = [c for c in candidates(tracks) if c["path_m"] > 0]
        for c in cands:
            c["camera_motion_px"] = camera_motion(H, c["t"])
        ratios = [c["path_px"] / c["path_m"] for c in cands if c["path_m"] > 0.5]
        K = st.median(ratios)
        rng = random.Random(int(sha(f"soccer-venue|{name}".encode())[:16], 16))
        items = {}
        for family in ("dynamics_path_player", "dynamics_speed_player"):
            pool = list(cands)
            rng.shuffle(pool)
            items[family] = sorted(pool[:ITEMS_PER_FAMILY], key=lambda c: (c["t"], c["track"]))
            if len(items[family]) < ITEMS_PER_FAMILY:
                print(f"ERROR: {name} has only {len(cands)} candidates")
                return 1
        pools = build_pools(name, K, items, tracks)
        for arm, rows_ in pools.items():
            for row in rows_:                           # references must score 100
                m = row["meta"]
                assert scalar_score(float(row["answer"]), m["reference_value"],
                                    m["score_tolerance"], m["score_floor"]) == 100, row
            data = (json.dumps(rows_, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
            rel = f"{name}/{arm}.json"
            path = PRIVATE / "inputs" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest_cells.append({"seq": name, "arm": arm, "items": len(rows_), "file": rel,
                                   "image_root": str(PRIVATE / "data" / f"frames_{name}"),
                                   "sha256": sha(data), "event_overlap": False})
        m_all = [c["path_m"] for c in cands]
        receipt_seqs.append({
            "name": name, "game_id": r["game_id"], "action": r["action"],
            "camera_motion_px_median": r["camera_motion_px_median"],
            "candidates": len(cands), "K_px_per_m": K,
            "path_m_median": st.median(m_all),
            "items_per_family": ITEMS_PER_FAMILY,
            "tracks_used": len({c["track"] for fam in items.values() for c in fam}),
        })
        print(f"{name}: {len(cands)} candidates, K {K:.2f} px/m, "
              f"camera motion {r['camera_motion_px_median']:.1f} px", flush=True)

    manifest = {"summary": {"protocol": "courtdyn-soccer-venue-v1",
                            "note": "SoccerNet-GSR valid; private; not redistributable"},
                "cells": manifest_cells, "arithmetic_file": None}
    (PRIVATE / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                                      encoding="utf-8")
    receipt = {"protocol": "courtdyn-soccer-venue-v1", "sequences": receipt_seqs,
               "cells": {c["file"]: c["sha256"] for c in manifest_cells},
               "private_root": str(PRIVATE), "template_prefix": TEMPLATE,
               "note": "Frames and pools are private (SoccerNet NDA); only this receipt "
                       "and aggregate results live in the repository."}
    (RESULTS / "build_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n",
                                                encoding="utf-8")
    print(f"-> {PRIVATE}\n-> {RESULTS / 'build_receipt.json'}")
    return 0


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.exit(main())
