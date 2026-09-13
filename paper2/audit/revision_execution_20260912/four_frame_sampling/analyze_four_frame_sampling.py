"""CPU-only audit of annotation-trajectory subsampling in frozen CourtDyn QA.

This is not an image reader or a model oracle. Both dense and four-point paths
use the original annotation smoothing, including annotation frames not shown to
the model. No source QA, predictions, model state, or pipeline code is modified.
The optional PRIVATE output must not be copied to a public release candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import scipy
from scipy.stats import spearmanr


class MethodMismatch(ValueError):
    """Stop rather than silently invent a replacement reference convention."""


def summarize(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "median": None, "iqr": None, "min_max": None}
    return {"n": int(len(a)), "median": float(np.median(a)),
            "iqr": np.quantile(a, [.25, .75]).tolist(),
            "min_max": [float(a.min()), float(a.max())]}


def rank(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def polyline(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def close(a, b, what, atol=1e-9):
    if not math.isclose(a, b, rel_tol=1e-11, abs_tol=atol):
        raise MethodMismatch(f"Reference convention mismatch: {what}")


def synthetic_checks(DQ, CourtPlane):
    """Two geometries check the nontrivial subsampling direction and units."""
    f0, f1 = 16, 76
    dense = list(range(f0, f1 + 1, DQ.STRIDE))
    four = [16, 36, 56, 76]
    plane = CourtPlane(np.diag([2., 2., 1.]))
    straight = {f: (f * .7, f * .2, 10., 30.) for f in range(1, 121)}
    curved = {f: (30 * math.cos(f / 10), 30 * math.sin(f / 10), 10., 30.)
              for f in range(1, 121)}
    results = []
    for name, boxes in (("straight", straight), ("curved", curved)):
        dp = polyline(DQ.foot_series(boxes, dense))
        sp = polyline(DQ.foot_series(boxes, four))
        dg = polyline(plane.foot_series_m(boxes, dense))
        sg = polyline(plane.foot_series_m(boxes, four))
        close(dg, 2 * dp, f"synthetic {name} metric scale")
        close(sg, 2 * sp, f"synthetic {name} sampled metric scale")
        if name == "straight":
            close(dp, sp, "straight trajectory must have no sampling deficit")
        elif not 0 < sp < dp:
            raise MethodMismatch("Curved-trajectory sampling check failed")
        results.append({"case": name, "four_over_dense": sp / dp,
                        "affine_ground_scale_check": "PASS"})
    return results


def write_json_new(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def analyze(workspace):
    sys.path.insert(0, str(workspace))
    from engine import dynamics_qa as DQ
    from engine.court_homography import CourtPlane, recompute_answer
    from tools import summarize_courtdyn_t28 as T
    from tools.build_courtdyn_ruler import tt_split
    import eval.evaluate as E

    if DQ.SMOOTH_HALF != 5 or DQ.STRIDE != 5 or T.RENDER_W != 960:
        raise MethodMismatch("Frozen smoothing/stride/render convention changed")
    synthetic = synthetic_checks(DQ, CourtPlane)
    source_sha = {}

    def record(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Authorized local reconstruction input missing: {path}")
        rel = path.relative_to(workspace).as_posix()
        source_sha[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path):
        return json.loads(record(path).read_text(encoding="utf-8"))

    for rel in ("engine/dynamics_qa.py", "engine/court_homography.py",
                "eval/evaluate.py", "tools/summarize_courtdyn_t28.py",
                "tools/summarize_courtdyn_v2.py", "tools/courtdyn_pixel_reader.py",
                "tools/rescore_courtdyn_homography.py", "tools/summarize_courtdyn_t26.py",
                "tools/summarize_courtdyn_seqs.py",
                "tools/build_courtdyn_ruler.py", "tools/build_courtdyn_qa.py",
                "results/courtdyn/teamtrack_probe_hits.json"):
        record(workspace / rel)

    cd = workspace / "results/courtdyn"
    families = dict(T.FAMS)
    records, cells, sequences = [], [], []
    for seq in ("Q2_top_480-510", "Q1_top_0-30"):
        selected = [x for x in read(cd / f"seq_{seq}/qa_dyn_v1.json")
                    if x["category"] in families]
        counts = Counter(x["category"] for x in selected)
        if counts != Counter({cat: 140 for cat in families}):
            raise MethodMismatch(f"Frozen two-family cohort changed: {seq}")
        items = {T.ikey(x): x for x in selected}
        if len(items) != len(selected):
            raise MethodMismatch(f"Duplicate item key: {seq}")
        px_pool = {T.ikey(x): x for x in read(cd / f"seq_{seq}/qa_dyn_v1_pxunit.json")}
        read(cd / f"seq_{seq}/qa_dyn_v1_ruler.manifest.json")
        record(cd / f"homography/H_{seq}.json")
        tt = Path(tt_split(seq))
        record(tt / "seqinfo.ini")
        record(tt / "gt/gt.txt")
        info = DQ.load_seqinfo(tt)
        tracks = DQ.load_tracks(tt)
        fps, scale = float(info["fps"]), T.RENDER_W / float(info["width"])
        plane = CourtPlane.load(seq)
        ctx = T.seq_ctx(seq)
        native_raw = read(Path(T.full_parsed(seq, "cdnative")) / "predictions.json")
        native_rows = native_raw if isinstance(native_raw, list) else native_raw["predictions"]
        native = {T.ikey(x): x for x in native_rows if x["category"] in families}
        sampling_offsets, dense_counts, durations, smooth_counts = [], [], [], []
        missing = Counter()
        source_agreement = Counter()
        seq_rows = []

        for key, item in items.items():
            m = item["meta"]
            frame_numbers = list(m["frames"])
            f0, f1 = m["window"]
            if len(frame_numbers) != 4 or sorted(set(frame_numbers)) != frame_numbers:
                raise MethodMismatch(f"Expected four unique ordered source moments: {seq}")
            if frame_numbers[0] != f0 or frame_numbers[-1] != f1:
                raise MethodMismatch(f"Sampling does not share dense-window endpoints: {seq}")
            dense_frames = list(range(f0, f1 + 1, DQ.STRIDE))
            if dense_frames[-1] != f1:
                dense_frames.append(f1)
            if not set(frame_numbers).issubset(dense_frames):
                raise MethodMismatch(f"Four sample moments are not a dense-grid subset: {seq}")
            if key not in native or key not in px_pool:
                raise MethodMismatch(f"Frozen native/pixel cohort does not cover selected items: {seq}")
            nrow = native[key]
            if (nrow.get("inference_image_ids") or nrow["image_ids"]) != item["image_ids"]:
                raise MethodMismatch(f"Native actual input images differ from selected QA: {seq}")
            for field in ("frames", "window", "track"):
                if nrow["meta"][field] != m[field] or px_pool[key]["meta"][field] != m[field]:
                    raise MethodMismatch(f"Frozen arm metadata mismatch: {seq}/{field}")
            if px_pool[key]["image_ids"] != item["image_ids"]:
                raise MethodMismatch(f"Pixel-arm source images differ: {seq}")
            encoded_frames = []
            for image_id in item["image_ids"]:
                match = re.match(r"f(\d+)_", Path(image_id).name)
                if not match:
                    raise MethodMismatch(f"Cannot verify frame filename convention: {seq}")
                encoded_frames.append(int(match.group(1)))
            if encoded_frames != frame_numbers:
                raise MethodMismatch(f"Image filenames and four sampling moments disagree: {seq}")
            close(float(m["fps"]), fps, "item fps")
            duration = (f1 - f0) / fps
            sampling_offsets.append(tuple(f - f0 for f in frame_numbers))
            dense_counts.append(len(dense_frames))
            durations.append(duration)
            boxes = tracks.get(m["track"])
            if boxes is None:
                missing["track_missing"] += 1
                continue
            if not all(f in boxes for f in frame_numbers):
                missing["annotation_missing_at_displayed_moment"] += 1
                continue
            for f in dense_frames:
                smooth_counts.append(sum(g in boxes for g in range(f - DQ.SMOOTH_HALF,
                                                                   f + DQ.SMOOTH_HALF + 1)))
            dense_px = DQ.foot_series(boxes, dense_frames)
            sampled_px = DQ.foot_series(boxes, frame_numbers)
            dense_ground = plane.foot_series_m(boxes, dense_frames)
            sampled_ground = plane.foot_series_m(boxes, frame_numbers)
            if any(x is None for x in (dense_px, sampled_px, dense_ground, sampled_ground)):
                missing["smoothing_neighborhood_unavailable"] += 1
                continue
            close(polyline(dense_px), DQ.path_length_px(boxes, f0, f1), "dense pixel function")
            close(polyline(dense_ground), plane.path_length_m(boxes, f0, f1), "dense ground function")
            # Verify the exact original averaging-before-projection order.
            for a, b in zip(dense_ground, [plane.to_court(x) for x in dense_px]):
                close(math.dist(a, b), 0., "smoothing before homography")
            source_agreement["dense_functions_and_projection_order"] += 1
            reconstructed = recompute_answer(plane, item, tracks, fps)
            close(float(reconstructed[0]), ctx["v3"][key], "current v3 rounded reference")
            close(float(px_pool[key]["answer"]), float(f"{ctx['px'][key]:.1f}"),
                  "stored one-decimal pixel reference")
            source_agreement["frozen_reference_conventions"] += 1

            for unit, dense_points, four_points, coordinate_scale in (
                    ("v3_ground", dense_ground, sampled_ground, 1.),
                    ("rendered_pixel", dense_px, sampled_px, scale)):
                dense_length = polyline(dense_points) * coordinate_scale
                sampled_length = polyline(four_points) * coordinate_scale
                if dense_length <= 0:
                    raise MethodMismatch(f"Nonpositive dense reference in selected cohort: {seq}")
                if sampled_length > dense_length + 1e-9:
                    raise MethodMismatch(f"Chord path exceeds identical-endpoint dense path: {seq}")
                for f, point in zip(frame_numbers, four_points):
                    close(math.dist(point, dense_points[dense_frames.index(f)]), 0.,
                          "same smoothed point at sampled moment")
                gaps = []
                for j, (a, b) in enumerate(zip(frame_numbers, frame_numbers[1:])):
                    segment = dense_points[dense_frames.index(a):dense_frames.index(b) + 1]
                    dense_gap = polyline(segment) * coordinate_scale
                    chord_gap = math.dist(four_points[j], four_points[j + 1]) * coordinate_scale
                    gaps.append({"sample_gap_index": j, "dense_length": dense_gap,
                                 "chord_length": chord_gap,
                                 "deficit": max(0., dense_gap - chord_gap)})
                close(sum(x["deficit"] for x in gaps), dense_length - sampled_length,
                      "three gap deficits sum to total path deficit")
                family = families[item["category"]]
                factor = 1 / duration if family == "speed" else 1.
                exact_ref = dense_length * factor
                sampled_value = sampled_length * factor
                target = ctx["v3" if unit == "v3_ground" else "px"][key]
                if unit == "v3_ground":
                    close(float(f"{exact_ref:.1f}"), target, "dense ground answer rounding")
                else:
                    close(exact_ref, target, "unrounded pixel scoring target")
                rounded_pred = float(f"{sampled_value:.1f}")
                rounded_ref = float(f"{target:.1f}")
                unit_scale = 1. if unit == "v3_ground" else float(ctx["K"])
                threshold, floor = (x * unit_scale for x in E.SCALAR_TMRA_SPEC[family])
                rel = (abs(rounded_pred - rounded_ref) - threshold) / max(abs(rounded_ref), floor)
                passes = [rel < 1 - theta for theta in E.T_MRA_THRESHOLDS]
                official = E.t_mra_scalar(f"{sampled_value:.1f}", f"{target:.1f}", threshold, floor)
                close(official, sum(passes) / len(passes), "threshold coverage and official scorer")
                row = {"sequence": seq, "family": family, "unit": unit,
                       "window_index": m["window_index"], "track": m["track"],
                       "window": m["window"], "frames": frame_numbers,
                       "duration_seconds": duration,
                       "dense_length_exact": dense_length,
                       "four_sample_length_exact": sampled_length,
                       "four_over_dense_exact": sampled_length / dense_length,
                       "relative_length_deficit": max(0., 1 - sampled_length / dense_length),
                       "dense_target_exact": exact_ref, "scoring_reference": target,
                       "four_sample_answer_exact": sampled_value,
                       "four_sample_answer_one_decimal": rounded_pred,
                       "tmra_0_to_100": 100 * official,
                       "within_absolute_T_after_rounding": abs(rounded_pred - rounded_ref) <= threshold,
                       "threshold_passes": passes, "gap_path_deficits": gaps}
                records.append(row)
                seq_rows.append(row)

        sequences.append({"sequence": seq,
                          "event_scope": "primary within-game temporal holdout" if seq.startswith("Q2") else
                                         "cross-view event overlap for original native checkpoint",
                          "questions": len(selected),
                          "unique_trajectory_window_pairs": len({k[1:] for k in items}),
                          "paired_speed_path_intersection": len(set(k[1:] for k in items if families[k[0]] == "speed") &
                                                                set(k[1:] for k in items if families[k[0]] == "path")),
                          "source_image_dimensions": [info["width"], info["height"]],
                          "rendered_image_dimensions": [960, int(round(info["height"] * scale))],
                          "fps": fps, "duration_seconds": summarize(durations),
                          "four_sample_offsets": sorted(set(sampling_offsets)),
                          "dense_point_counts": sorted(set(dense_counts)),
                          "smoothing_annotation_counts_per_point": summarize(smooth_counts),
                          "method_checks_passed_question_count": dict(source_agreement),
                          "unavailable_question_reasons": dict(missing)})

        for family in ("speed", "path"):
            for unit in ("v3_ground", "rendered_pixel"):
                rows = [x for x in seq_rows if x["family"] == family and x["unit"] == unit]
                unit_scale = 1. if unit == "v3_ground" else float(ctx["K"])
                threshold, floor = (float(x * unit_scale) for x in E.SCALAR_TMRA_SPEC[family])
                ratios = [x["four_over_dense_exact"] for x in rows]
                deficits = [x["relative_length_deficit"] for x in rows]
                official = T.tmra([(x["four_sample_answer_exact"], x["scoring_reference"]) for x in rows],
                                  family, unit_scale)
                dense_score = T.tmra([(x["dense_target_exact"], x["scoring_reference"]) for x in rows],
                                    family, unit_scale)
                close(official, float(np.mean([x["tmra_0_to_100"] for x in rows])), "summary T-MRA")
                close(dense_score, 100., "dense identity T-MRA")
                bucket_edges = [0., .01, .05, .10, .20, .50, float("inf")]
                buckets = []
                for left, right in zip(bucket_edges, bucket_edges[1:]):
                    count = sum(left <= x < right for x in deficits)
                    buckets.append({"lower_inclusive": left, "upper_exclusive": right if math.isfinite(right) else None,
                                    "count": count})
                if sum(x["count"] for x in buckets) != len(rows):
                    raise MethodMismatch("Descriptive loss bins do not cover the cohort")
                cells.append({"sequence": seq, "family": family, "unit": unit,
                              "cohort_questions": 140, "available_questions": len(rows),
                              "coverage_fraction": len(rows) / 140,
                              "positive_dense_denominators": sum(x["dense_target_exact"] > 0 for x in rows),
                              "T": threshold, "denominator_floor": floor,
                              "K_for_pixel_tolerance_only": float(ctx["K"]) if unit == "rendered_pixel" else None,
                              "ratio_four_to_dense_exact": summarize(ratios),
                              "relative_length_deficit": summarize(deficits),
                              "absolute_target_deficit": summarize([x["dense_target_exact"] - x["four_sample_answer_exact"] for x in rows]),
                              "smoothed_path_chord_deficit_positive_count": sum(x["dense_length_exact"] - x["four_sample_length_exact"] > 1e-9 for x in rows),
                              "deficit_exceeds_one_percent_count": sum(x > .01 for x in deficits),
                              "deficit_exceeds_five_percent_count": sum(x > .05 for x in deficits),
                              "deficit_exceeds_ten_percent_count": sum(x > .10 for x in deficits),
                              "deficit_exceeds_twenty_percent_count": sum(x > .20 for x in deficits),
                              "descriptive_relative_deficit_bins": buckets,
                              "gap_index_relative_deficit_medians": [float(np.median([x["gap_path_deficits"][j]["deficit"] / x["dense_length_exact"] for x in rows])) for j in range(3)],
                              "tmra": official, "dense_identity_tmra": dense_score,
                              "tmra_deficit_from_dense_identity": 100 - official,
                              "tmra_100_count": sum(x["tmra_0_to_100"] == 100 for x in rows),
                              "tmra_zero_count": sum(x["tmra_0_to_100"] == 0 for x in rows),
                              "within_absolute_T_count_after_rounding": sum(x["within_absolute_T_after_rounding"] for x in rows),
                              "coverage_by_original_tmra_threshold": [{"theta": float(theta), "count": sum(x["threshold_passes"][j] for x in rows), "total": len(rows)} for j, theta in enumerate(E.T_MRA_THRESHOLDS)],
                              "rho_one_decimal_answer_vs_original_scoring_reference": rank([x["four_sample_answer_one_decimal"] for x in rows], [x["scoring_reference"] for x in rows]),
                              "rho_unrounded_answer_vs_original_scoring_reference": rank([x["four_sample_answer_exact"] for x in rows], [x["scoring_reference"] for x in rows]),
                              "rho_unrounded_four_vs_unrounded_dense": rank([x["four_sample_answer_exact"] for x in rows], [x["dense_target_exact"] for x in rows]),
                              "rho_dense_one_decimal_vs_original_scoring_reference": rank([float(f"{x['dense_target_exact']:.1f}") for x in rows], [x["scoring_reference"] for x in rows]),
                              "reference_median_constant_tmra": ctx["const"][(family, "v3" if unit == "v3_ground" else "px")]
                              })

    for rel, before in source_sha.items():
        if hashlib.sha256((workspace / rel).read_bytes()).hexdigest() != before:
            raise MethodMismatch(f"Read-only source changed during analysis: {rel}")
    result = {"schema": "courtdyn-four-smoothed-samples-v1", "status": "PASS",
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "scope": "CPU annotation-trajectory subsampling audit only. No image coordinate detection and no model inference.",
              "method": {"smoothing": "Original image-plane footpoint moving average over available annotations within +/-5 frames, before homography.",
                         "dense_path": "Original STRIDE=5 inclusive sampling; endpoint appended if needed.",
                         "four_sample_path": "Original meta.frames, same smoothed coordinates and endpoints; three Euclidean chords.",
                         "speed": "Corresponding path divided by (f1-f0)/source_fps, not nominal 2 seconds.",
                         "ground": "Existing H_img2court_m and CourtPlane.foot_series_m, unchanged.",
                         "pixel": "Existing DQ.foot_series coordinates times 960/source_width; no processor-size claim.",
                         "reference": "Original T.seq_ctx: one-decimal v3, unrounded recomputed rendered-pixel target; verify stored pixel answer rounding.",
                         "tmra": "Original T.tmra/E.t_mra_scalar; format prediction and GT to one decimal; original T/floor, both multiplied by clip K only for pixel.",
                         "rho_primary": "Serialize the four-point answer to requested one decimal; compare against original scoring reference. Additional unrounded rank values are labeled separately.",
                         "ratio": "Four-point path divided by exact dense path before answer rounding; speed has the same per-trajectory ratio.",
                         "coverage": "Availability and each original T-MRA threshold; no model parse success claim.",
                         "loss_interpretation": "Chord deficits along smoothed annotations, not proof of actual physical bends rather than annotation noise."},
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
              "synthetic_geometry_checks": synthetic, "sequences": sequences, "cells": cells,
              "source_sha256": source_sha, "source_preservation_check": "PASS",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cautions": ["Four sampled points use hidden neighboring annotation frames through the fixed smoothing convention; not an image-based oracle or proof of coordinate readability.",
                           "Dense reference is the original 13-point smoothed annotation polyline, not continuous true physical motion.",
                           "Every selected question remains unchanged; sampling estimates are separate audit values, not replacement GT.",
                           "Q1 event overlap applies to the historical native model; this deterministic geometry check does not create new independent events.",
                           "Questions and speed/path pairs share trajectories and time. No independent-event confidence intervals are claimed.",
                           "The PRIVATE item diagnostics contain restricted derived values and identifiers and must not enter public candidates."]}
    private = {"distribution": "PRIVATE_LOCAL_ONLY_NOT_FOR_PUBLIC_RELEASE", "schema": result["schema"],
               "script_sha256": result["script_sha256"], "records": records}
    return result, private


def render_report(result):
    lines = ["# 四采样轨迹参照与采样损失：CPU 核对", "",
             "此分析使用已知标注轨迹，不从图像检测坐标，不执行模型推理。原始题目、真值、预测和训练流程均未修改。", "",
             "在 Q2 与 Q1 各选用现有 140 个 speed 与 140 个 path 问题，共 560 题、8 个事件/家族/坐标结果单元。", "",
             "## 方法与解释范围", "",
             "严格复用原足点平滑：在图像平面用 ±5 帧标注做滑动平均，再投影到地面或换算至 960×540 渲染像素。原密集路径以 5 帧为步长，包含 13 个点；四采样路径读取原 QA 的四个实际时刻，连接同样平滑后的四个点。两者首尾相同，四点是密集点的子集。", "",
             "实际采样偏移为 0、20、40、60 帧，时长 60 / 29.97002997 = 2.002 秒。speed 为对应路径除以该实际时长。四采样点本身仍借助周边未显示帧的标注平滑，因此这是固定参考约定下的理想化子采样检查；不能据此证明四张图像足以读取坐标，更不能证明模型能完成读取。", "",
             "下表比值使用未舍入的四点/密集路径，IQR 描述同一事件内的题目分布；ρ 的主口径将四点答案按题面要求保留一位小数，再与原评分参考比较。T-MRA 复用原评分器、参考、T 与 floor，像素臂对 T/floor 乘原片段 K。聚合 JSON 另列未舍入相关、原阈值逐级覆盖、绝对容差覆盖及常数参照。", "",
             "## 按事件与问题家族的结果", "",
             "| 片段 | 家族 | 坐标 | 可用 | 四点/密集 中位数 [IQR] | >10% 损失 | T-MRA | ρ（一位小数） |",
             "|---|---|---|---:|---|---:|---:|---:|"]
    for c in result["cells"]:
        s = c["ratio_four_to_dense_exact"]
        lines.append(f"| {c['sequence'].split('_')[0]} | {c['family']} | {c['unit']} | {c['available_questions']}/140 | "
                     f"{s['median']:.4f} [{s['iqr'][0]:.4f}, {s['iqr'][1]:.4f}] | "
                     f"{c['deficit_exceeds_ten_percent_count']} | {c['tmra']:.3f} | "
                     f"{c['rho_one_decimal_answer_vs_original_scoring_reference']:.4f} |")
    q2 = [c for c in result["cells"] if c["sequence"].startswith("Q2")]
    q1 = [c for c in result["cells"] if c["sequence"].startswith("Q1")]
    q2_ratio = [c["ratio_four_to_dense_exact"]["median"] for c in q2]
    q1_ratio = [c["ratio_four_to_dense_exact"]["median"] for c in q1]
    min_ratio = min(c["ratio_four_to_dense_exact"]["min_max"][0] for c in result["cells"])
    all_nonzero = all(c["smoothed_path_chord_deficit_positive_count"] == c["available_questions"]
                      for c in result["cells"])
    all_100 = all(abs(c["tmra"] - 100) < 1e-9 for c in result["cells"])
    score_range = [min(c["tmra"] for c in result["cells"]), max(c["tmra"] for c in result["cells"])]
    deficit_text = "所有题均有非零的几何弦长亏损" if all_nonzero else "各格的非零几何弦长亏损计数见聚合结果"
    score_text = "全部 T-MRA=100" if all_100 else f"T-MRA 范围为 {score_range[0]:.3f}–{score_range[1]:.3f}"
    lines += ["", "所有单元的密集参照自评分均为 100。采样损失为三个采样间隔内，密集折线路径长度减对应端点弦长之和；脚本逐项核验该和等于总损失，并核验四点路径不长于同首尾密集路径。", "",
              "## 可以写入论文的观察与不能外推的结论", "",
              f"Q2 四个单元的路径保留比中位数为 {min(q2_ratio):.4f}–{max(q2_ratio):.4f}，Q1 为 {min(q1_ratio):.4f}–{max(q1_ratio):.4f}。相同平滑条件下，{deficit_text}；最严重题目的保留比为 {min_ratio:.4f}，约损失 {(1-min_ratio)*100:.1f}% 的密集参考长度。{score_text}。原评分容差可能包容路径差异，高分不能表示四点路径与密集路径完全一致。", "",
              ("在这个使用标注坐标的固定参考计算中，单独把密集积分改为四采样折线，没有降低原 T-MRA。这个结果限制了纯粹由该子采样操作造成评分损失的解释；" if all_100 else "本次结果量化了将密集积分改为四采样折线时的参考损失；") + "它没有测试图像中的坐标可读性、遮挡、跟踪、标定错误或模型是否获得了这些平滑坐标。", "",
              f"建议英文表述：Using the same annotated footpoint smoothing and the four source sampling times, an annotation-based polyline reference retained {min(q2_ratio):.4f}–{max(q2_ratio):.4f} of the dense path at the median in the Q2 cells. Across all eight clip/family/coordinate cells, T-MRA ranged from {score_range[0]:.3f} to {score_range[1]:.3f} under the original tolerances. This is a trajectory-subsampling check, not an image-based oracle: the smoothing uses neighboring annotation frames that are not displayed to the model.", "",
              "## 保留的诊断与边界", "",
              "- `aggregate.json` 只含聚合值、方法与输入 SHA；不含逐项问题、轨迹编号、参考答案或图像编号。",
              "- `PRIVATE_item_sampling_diagnostics.json` 记录哪些题出现弦长亏损、三个间隔各自的亏损以及原窗口/轨迹定位。该文件是受限派生标注，仅供本地审计，禁止纳入公开候选。",
              "- 小于密集路径不必都来自真实球员转弯，也可能包括平滑后仍残留的标注噪声。此处只确认几何上的折线/弦长差。",
              "- 高 T-MRA 可以与非零路径损失同时出现，因为原评分存在绝对容差和分母下限。T-MRA 覆盖不能被解释成精确轨迹恢复。",
              "- 此检查没有新增模型、比赛、场地或对照，也不解决坐标读取、投影标定精度与训练先验问题。", "",
              "## 验证", "",
              f"输入 SHA 数量：{len(result['source_sha256'])}；分析前后文件 SHA 全部一致。原密集函数、投影顺序、题目与模型实际输入元数据、原像素题池舍入答案、原评分函数及几何关系均逐项核验。另通过直线无亏损、弯曲路径有亏损与 2× 仿射单位缩放检查。", "",
              "重跑时使用新输出目录：", "", "```powershell",
              "python -B paper2/audit/revision_execution_20260912/four_frame_sampling/analyze_four_frame_sampling.py --output-dir <new-local-audit-directory>",
              "```", "", "已有输出不被覆盖；`--no-private` 可仅生成聚合结果与报告。仍需作者有权访问的原始标注、题池与冻结原生模型元数据。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--no-private", action="store_true", help="Omit restricted item-level diagnostics")
    args = parser.parse_args()
    workspace = args.workspace or next((p for p in Path(__file__).resolve().parents
                                        if (p / "engine/dynamics_qa.py").is_file()), None)
    if workspace is None:
        parser.error("Use --workspace with authorized local CourtDyn reconstruction inputs")
    workspace = workspace.resolve()
    if not (workspace / "engine/dynamics_qa.py").is_file():
        parser.error("Workspace lacks engine/dynamics_qa.py; supply the authorized CourtDyn workspace")
    out = args.output_dir.resolve()
    names = ["aggregate.json", "report.md"]
    if not args.no_private:
        names.append("PRIVATE_item_sampling_diagnostics.json")
    if any((out / name).exists() for name in names):
        parser.error("Audit output exists; choose a new --output-dir. Existing files are never overwritten.")
    try:
        result, private = analyze(workspace)
    except (FileNotFoundError, MethodMismatch, KeyError, ModuleNotFoundError) as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    write_json_new(out / "aggregate.json", result)
    if not args.no_private:
        write_json_new(out / "PRIVATE_item_sampling_diagnostics.json", private)
    with (out / "report.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(render_report(result))
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]),
                      "questions": sum(x["questions"] for x in result["sequences"]),
                      "output_dir": str(out), "source_files_unchanged": len(result["source_sha256"]),
                      "private_item_output": not args.no_private}, ensure_ascii=False))


if __name__ == "__main__":
    main()
