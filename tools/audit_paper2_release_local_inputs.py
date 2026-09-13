#!/usr/bin/env python
"""Check exported code against locally obtained inputs without putting them in a release.

Loads real data read-only; generated frames/calibration stay in a temporary directory.
This is an author-side audit, not a self-contained public reproduction workflow.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--calibrate", action="store_true", help="also rerun automatic video calibration")
    args = ap.parse_args()
    root, bundle = args.workspace.resolve(), args.bundle.resolve()
    sys.path.insert(0, str(bundle))
    from engine import dynamics_qa as dq
    from engine.court_homography import CourtPlane, recompute_answer
    from tools import build_courtdyn_qa as builder
    from tools import summarize_courtdyn_t28 as scorer
    for module in (dq, builder, scorer):
        assert Path(module.__file__).is_relative_to(bundle), module.__file__
    seq = "Q1_top_0-30"
    cd = root / "results/courtdyn"
    source = root / "data/external_validation/teamtrack/teamtrack-mot/teamtrack-mot/basketball_top/train" / seq
    original = cd / f"seq_{seq}/qa_dyn_v1.json"
    manifest = json.loads(original.with_name("qa_dyn_v1.manifest.json").read_text(encoding="utf-8"))
    source_matches = {
        "gt": digest(source / "gt/gt.txt") == manifest["source"]["gt_sha256"],
        "video": digest(source / "img1.mp4") == manifest["source"]["video_sha256"],
    }
    assert all(source_matches.values()), source_matches
    params = manifest["params"]
    items, _ = dq.generate(str(source), **{k: params[k] for k in
        ("win_s", "hop_s", "n_frames", "max_per_family")})
    # Frame basenames are deterministic; this checks all QA payloads independently
    # of video decoding. One real four-frame sample is actually rendered below.
    for item in items:
        item["image_ids"] = [f"f{f:04d}_{builder.hl_key(item['highlight'])}.jpg"
                             for f in item["window"]["frames"]]
        item["image_id"] = item["image_ids"][0]
    rows = [builder.strip(item) for item in items]
    newline = "CRLF" if b"\r\n" in original.read_bytes() else "LF"
    payload = json.dumps(rows, ensure_ascii=False, indent=1)
    if newline == "CRLF":
        payload = payload.replace("\n", "\r\n")
    rebuilt_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    pool_check = {"n": len(rows), "original_sha256": digest(original),
                  "rebuilt_sha256": rebuilt_sha, "line_endings": newline,
                  "match": rebuilt_sha == digest(original)}
    assert pool_check["match"], pool_check
    tracks, info = dq.load_tracks(str(source)), dq.load_seqinfo(str(source))
    h_original = cd / "homography" / f"H_{seq}.json"
    plane = CourtPlane.load(seq, h_dir=str(h_original.parent))
    v3, px, fam_of = {}, {}, {}
    scale = 960 / info["width"]
    for row in rows:
        fam = dict(scorer.FAMS).get(row["category"])
        if not fam:
            continue
        key = scorer.ikey(row)
        fam_of[key] = fam
        v3[key] = float(recompute_answer(plane, row, tracks, info["fps"])[0])
        m = row["meta"]
        f0, f1 = m["window"]
        d = dq.path_length_px(tracks[m["track"]], f0, f1) * scale
        px[key] = d if fam == "path" else d / ((f1 - f0) / info["fps"])
    frozen = json.loads((cd / "courtdyn_t28_findings.json").read_text(encoding="utf-8"))["seqs"][seq]
    k = frozen["k_px_per_m"]
    cells = []
    for arm, pool in [("full", "main"), ("ruler", "ruler"), ("pxunit", "pxunit")]:
        predictions = scorer.preds(str(cd / f"t33_{seq}_{pool}_full_cdnative_parsed"))
        for fam in ("speed", "path"):
            result = scorer.cell_row(predictions, px if arm == "pxunit" else v3, fam_of, fam,
                                     k=k if arm == "pxunit" else 1.0)
            expected = frozen["cells"][f"cdnative@{'pxunit' if arm == 'pxunit' else 'ruler'}@{fam}"][arm]
            match = all(abs(result[key] - expected[key]) < 1e-10 for key in expected)
            cells.append({"arm": arm, "family": fam, "match": match, "recomputed": result})
            assert match, cells[-1]
    calibration = {"status": "not rerun; frozen local H used for rescoring"}
    with tempfile.TemporaryDirectory(prefix="courtdyn-real-audit-") as scratch:
        temp = Path(scratch)
        sample = copy.deepcopy(next(i for i in items if i["category"] == "dynamics_speed_player"))
        base = builder.extract_base_frames(str(source / "img1.mp4"), sample["window"]["frames"],
                                           str(temp / "base"), scale)
        builder.render_overlays([sample], tracks, base, scale, str(temp / "overlays"))
        frame_checks = []
        for name in sample["image_ids"]:
            reference = root / "data/courtdyn" / f"frames_{seq}" / name
            frame_checks.append({"reference_available": reference.exists(),
                                 "match": reference.exists() and digest(reference) == digest(temp / "overlays" / name)})
        if args.calibrate:
            import numpy as np
            from tools import calibrate_court_homography as calibration_code
            rebuilt = calibration_code.calibrate(str(source), str(temp / "calibration"), n_frames=9)
            old = json.loads(h_original.read_text(encoding="utf-8"))
            err = float(np.max(np.abs(np.array(rebuilt["H_img2court_m"]) - np.array(old["H_img2court_m"]))))
            calibration = {"status": "rerun", "H_max_absolute_difference": err,
                           "H_match_at_1e-10": err < 1e-10,
                           "sampled_frame_numbers_match": [r["frame"] for r in old["per_frame"]] ==
                               [r["frame"] for r in rebuilt["per_frame"]],
                           "whole_file_byte_equality_not_expected": "build timestamp changes"}
    report = {"scope": "author-side, exported code + read-only local source data and unredacted predictions",
              "sequence": seq, "source_hashes": source_matches, "qa_payload": pool_check,
              "core_cells": cells, "rendered_sample_frame_hashes": frame_checks, "calibration": calibration,
              "limitations": ["only one sequence", "no retraining or inference", "not an online archive audit",
                              "not every withheld pool checked", "audit environment differs from training environment"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
