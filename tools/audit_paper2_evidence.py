#!/usr/bin/env python
"""Post-review, CPU-only evidence audit; never writes frozen result artifacts.

Run with the existing research Python environment (numpy/scipy/cv2; optional
transformers/torch for --processor). Outputs belong to paper2/audit only.
The temporal block bootstrap is exploratory within-clip sensitivity analysis,
not preregistered inference and not evidence of independent event generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_courtdyn_t28 as T28
from tools import courtdyn_pixel_reader as PR
from tools import build_courtdyn_native_sft as NS
from engine import dynamics_qa as DQ
from engine.court_homography import CourtPlane, recompute_answer

OUT = ROOT / "paper2" / "audit"
CD = ROOT / "results" / "courtdyn"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    path = OUT / ("evidence_" + name + ".json")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return str(path.relative_to(ROOT))


def rho(a, b):
    a, b = rankdata(a), rankdata(b)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def interval(values):
    vals = [x for x in values if x is not None and np.isfinite(x)]
    return {"valid_replicates": len(vals), "percentile_95": np.quantile(vals, [.025, .975]).tolist() if vals else None}


def numeric(preds):
    return {k: v for k, v in preds.items() if math.isfinite(v)}


def native():
    train = read(CD / "courtdyn_native_sft_train.json")
    pools = {seq: {T28.ikey(r): r for r in read(NS.qa_path(seq))} for seq in NS.LEGACY_TRAIN_SEQS}
    matches = Counter()
    mismatches = []
    train_frames = set()
    for r in train:
        seq = r["meta"]["source_seq"]
        src = pools[seq][T28.ikey(r)]
        matched = r["answer"] == str(src["answer"]) and r["question"] == src["question"]
        matches[seq] += int(matched)
        if not matched:
            mismatches.append([seq, list(T28.ikey(r))])
        train_frames.update((seq, int(f)) for f in r["meta"]["frames"])
    held_frames = {(seq, int(f)) for seq in NS.HELD_OUT for r in read(Path(NS.qa_path(seq))) for f in r["meta"]["frames"]}
    return {"n_train": len(train), "n_per_seq": dict(Counter(r["meta"]["source_seq"] for r in train)),
            "all_questions_and_answers_match_v1": not mismatches, "v1_matches_per_seq": dict(matches),
            "mismatches": mismatches, "training_file_sha256": sha(CD / "courtdyn_native_sft_train.json"),
            "sequence_id_frame_intersection": len(train_frames & held_frames),
            "event_sync_status": "Q1 event overlap supported by source-video inspection and collective-trajectory alignment; see evidence_event_alignment.json. Exact camera clock offset/person identity not certified.",
            "source_video_info": {seq: DQ.load_seqinfo(next(p.parent for p in (ROOT / "data/external_validation/teamtrack").rglob("seqinfo.ini") if p.parent.name == seq)) for seq in ["Q1_side_0-30", "Q1_top_0-30"]}}


def human_m3():
    paper1 = read(ROOT / "results/external_validation/human_m3_metric_qa.json")
    p1_frames = {(r["group_key"].replace("/-/", "/"), int(r["frame"])) for r in paper1}
    hmroot = ROOT / "data/external_validation/human_m3_basketball_test/annotated/humanm3/test"
    selected = [CD / f"seq_hm3_basketball1_split1_camera_{i}" for i in range(4)]
    rows = []
    p2_all = set()
    for directory in selected:
        manifest = read(directory / "qa_dyn_v1.manifest.json")
        group = manifest["clip"] + "/" + manifest["camera"]
        pool = read(directory / "qa_dyn_v1.json")
        frames = {(group, int(f)) for r in pool for f in r["meta"]["frames"]}
        p2_all |= frames
        overlap = sorted(frames & p1_frames)
        poses = sorted(int(p.stem) for p in (hmroot / manifest["clip"] / "pose_calib").glob("*.json"))
        imgs = sorted((hmroot / manifest["clip"] / "images" / manifest["camera"]).glob("*.jp*g"))
        mapping = dict(zip(poses, imgs))
        identity_matches = []
        for g, f in overlap:
            source = mapping[f].resolve()
            p1_paths = {(ROOT / r["image_id"]).resolve() for r in paper1 if r["group_key"].replace("/-/", "/") == g and int(r["frame"]) == f}
            identity_matches.append(source in p1_paths)
        rows.append({"group": group, "paper2_qa": len(pool), "paper2_unique_source_frames": len(frames),
                     "paper1_unique_source_frames_same_group": sum(g == group for g, f in p1_frames),
                     "shared_camera_frame_count": len(overlap), "shared_frame_ids": [f for _, f in overlap],
                     "all_shared_frames_resolve_to_identical_original_image_paths": all(identity_matches),
                     "paper2_qa_with_at_least_one_shared_source_frame": sum(any((group, int(f)) in p1_frames for f in r["meta"]["frames"]) for r in pool)})
    return {"scope": "Paper 1 real-metric Human-M3 800-item pool vs Paper 2 reported four basketball1/split1 cameras; source frames, not identically rendered inputs or identical QA.",
            "paper1_items": len(paper1), "paper1_unique_camera_frames": len(p1_frames),
            "paper2_unique_camera_frames": len(p2_all), "shared_camera_frames": len(p1_frames & p2_all),
            "per_camera": rows, "paper1_sha256": sha(ROOT / "results/external_validation/human_m3_metric_qa.json")}


def event_alignment():
    """Descriptive check of matched group motion; no identity/sync inference claim."""
    from scipy.optimize import linear_sum_assignment
    seqs = ["Q1_side_0-30", "Q1_top_0-30"]
    context = {}
    for seq in seqs:
        d = next((ROOT / "data/external_validation/teamtrack").rglob(seq))
        tracks = DQ.load_tracks(d)
        players, _ = DQ.split_roles(tracks)
        context[seq] = (DQ.load_seqinfo(d), tracks, players, CourtPlane.load(seq))

    def positions(seq, t):
        info, tracks, players, plane = context[seq]
        f = round(t * info["fps"]) + 1
        xy = []
        for track in players:
            if f not in tracks[track]:
                continue
            x, y, w, h = tracks[track][f]
            pos = np.array(plane.to_court((x + w / 2, y + h)))
            if seq == seqs[1]:
                pos = np.array([28., 15.]) - pos
            xy.append(pos)
        return np.array(xy)

    rows = []
    for t in np.arange(.5, 29.01, .5):
        a, b = positions(seqs[0], t), positions(seqs[1], t)
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        ii, jj = linear_sum_assignment(distances)
        rows.append({"original_time_s": float(t), "n_side": len(a), "n_top": len(b),
                     "centroid_side_xy_m": a.mean(axis=0).tolist(),
                     "centroid_top_rotated_xy_m": b.mean(axis=0).tolist(),
                     "median_matched_distance_m": float(np.median(distances[ii, jj]))})
    a = np.array([r["centroid_side_xy_m"] for r in rows])
    b = np.array([r["centroid_top_rotated_xy_m"] for r in rows])
    return {"scope": "Descriptive cross-view event corroboration using existing per-view court homographies and MOT foot points; 180-degree top court rotation follows the visible court orientation. Not person-ID matching, certified clock synchronization, or a new metric calibration.",
            "visual_evidence": "Six original-time samples per view show the same opening centre-circle arrangement/tip-off, attack at the same physical basket at 10-20 s, and transition to the other half by 25 s.",
            "interpretation": "Visual event inspection plus matching collective trajectories supports overlapping opening game events across the two clips. Exact camera clock offset and individual player identity were not certified.",
            "time_basis": "Source frame index round(t * seqinfo fps) + 1 for MOT; contact images use zero-based index round(t * seqinfo fps). Packed MP4 fps is not used.",
            "n_time_samples": len(rows), "centroid_pearson_x": float(np.corrcoef(a[:, 0], b[:, 0])[0, 1]),
            "centroid_pearson_y": float(np.corrcoef(a[:, 1], b[:, 1])[0, 1]),
            "median_centroid_distance_m": float(np.median(np.linalg.norm(a-b, axis=1))),
            "median_of_timepoint_matched_distances_m": float(np.median([r["median_matched_distance_m"] for r in rows])),
            "rows": rows}


def processor(model):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from eval.run_bench import GENERATION_SUFFIX
    p = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True, max_pixels=200704)
    item = read(CD / "seq_Q1_top_0-30/qa_dyn_v1.json")[0]
    images = [Image.open(ROOT / "data/courtdyn/frames_Q1_top_0-30" / name).convert("RGB") for name in item["image_ids"]]
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": item["question"] + GENERATION_SUFFIX}]}]
    enc = p.apply_chat_template(msgs, enable_thinking=False, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    grid = enc["image_grid_thw"].tolist()
    patch = int(p.image_processor.patch_size)
    return {"scope": "Current installed environment CPU replay of run_bench preprocessing; historical grid tensors were not saved. No model weights or inference loaded.",
            "python": sys.version, "transformers": transformers.__version__, "torch": torch.__version__,
            "model_snapshot": str(model), "processor": type(p).__name__, "image_processor": type(p.image_processor).__name__,
            "image_processor_config": p.image_processor.to_dict(), "source_sizes_wh": [list(im.size) for im in images],
            "image_grid_thw": grid, "processed_sizes_wh": [[w * patch, h * patch] for _, h, w in grid],
            "pixel_values_shape": list(enc["pixel_values"].shape),
            "snapshot_preprocessor_sha256": sha(Path(model) / "preprocessor_config.json")}


def stats(n_boot):
    results = []
    rng = np.random.default_rng(20260912)
    for seq in T28.TOPS:
        ctx = T28.seq_ctx(seq)
        meta = {T28.ikey(r): r["meta"] for r in read(CD / f"seq_{seq}/qa_dyn_v1.json")}
        v1 = {T28.ikey(r): float(r["answer"]) for r in read(CD / f"seq_{seq}/qa_dyn_v1.json") if r["category"] in dict(T28.FAMS)}
        for model in ("sft", "grpo", "tsft", "cdnative", "base", "qwen25vl7b"):
            full = numeric(T28.preds(T28.full_parsed(seq, model)))
            for arm in ("ruler", "pxunit"):
                apred = numeric(T28.preds(T28.t28_parsed(seq, arm, "full", model)))
                agt = ctx["px"] if arm == "pxunit" else ctx["v3"]
                for _, fam in T28.FAMS:
                    keys = sorted(k for k in full.keys() & apred.keys() & ctx["v3"].keys() & agt.keys() if ctx["fam_of"][k] == fam)
                    if not keys:
                        continue
                    f = np.array([full[k] for k in keys]); a = np.array([apred[k] for k in keys])
                    g = np.array([ctx["v3"][k] for k in keys]); ag = np.array([agt[k] for k in keys])
                    ratio = a[f > 0] / f[f > 0]
                    rf, ra, ra_same = rho(f, g), rho(a, ag), rho(a, g)
                    scores_f = np.array([T28.tmra([(p, gt)], fam) for p, gt in zip(f, g)])
                    scores_a = np.array([T28.tmra([(p, gt)], fam, ctx["K"] if arm == "pxunit" else 1) for p, gt in zip(a, ag)])
                    row = {"sequence": seq, "model": model, "arm": arm, "family": fam, "n_pool": 140,
                           "n_valid_full": sum(ctx["fam_of"].get(k) == fam for k in full),
                           "n_valid_arm": sum(ctx["fam_of"].get(k) == fam for k in apred),
                           "n_common": len(keys), "n_positive_full_ratio": len(ratio),
                           "ratio_median": float(np.median(ratio)) if len(ratio) else None,
                           "ratio_q1_q3": np.quantile(ratio, [.25, .75]).tolist() if len(ratio) else None,
                           "ratio_of_medians": float(np.median(a) / np.median(f)) if np.median(f) > 0 else None,
                           "full_rho_v3": rf, "arm_rho_native_target": ra, "arm_rho_v3_same_target": ra_same,
                           "delta_rho_native_targets": ra - rf if ra is not None and rf is not None else None,
                           "delta_rho_v3_same_target": ra_same - rf if ra_same is not None and rf is not None else None,
                           "full_rho_v1_same_items": rho(f, [v1[k] for k in keys]),
                           "score_delta_native_targets": float(np.mean(scores_a - scores_f)), "temporal_blocks": {}}
                    # Cluster all tracks by contiguous bins of original window start;
                    # adjacent bins can still share boundary frames. Sensitivity only.
                    for seconds in (5, 10):
                        labels = np.array([int((meta[k]["window"][0] / meta[k]["fps"]) // seconds) for k in keys])
                        clusters = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
                        b_rho, b_same, b_ratio, b_score = [], [], [], []
                        for _ in range(n_boot):
                            idx = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), size=len(clusters))])
                            r0, r1, r2 = rho(f[idx], g[idx]), rho(a[idx], ag[idx]), rho(a[idx], g[idx])
                            b_rho.append(r1 - r0 if r0 is not None and r1 is not None else None)
                            b_same.append(r2 - r0 if r0 is not None and r2 is not None else None)
                            valid = idx[f[idx] > 0]
                            b_ratio.append(float(np.median(a[valid] / f[valid])) if len(valid) else None)
                            b_score.append(float(np.mean(scores_a[idx] - scores_f[idx])))
                        row["temporal_blocks"][str(seconds)] = {"n_clusters": len(clusters), "cluster_sizes": [len(c) for c in clusters],
                            "delta_rho_native_targets": interval(b_rho), "delta_rho_v3_same_target": interval(b_same),
                            "median_paired_ratio": interval(b_ratio), "score_delta_native_targets": interval(b_score)}
                    results.append(row)
    return {"status": "Exploratory post-review sensitivity, not preregistered; no equivalence margin was fixed.",
            "bootstrap": {"seed": 20260912, "replicates": n_boot, "method": "Paired non-overlapping 5-s and 10-s bins of window start, resampled as whole clusters including all tracks. Paired models/targets use exactly the same sampled rows; quantiles are NumPy linear quantiles; undefined constant-output correlations are excluded and counts retained.",
                          "limitations": "Only 3-6 clusters per 30-s clip; adjacent bins can share boundary frames, tracks span bins, no independent-match uncertainty. These intervals cannot establish equivalence or population generalization."},
            "cells": results}


def rank_versions():
    rows = []
    for seq, _ in PR.SEQS:
        sd = PR.seq_dir_of(seq)
        info, tracks = DQ.load_seqinfo(sd), DQ.load_tracks(sd)
        features = PR.item_features(seq, tracks, info["fps"])
        for model, _ in PR.MODELS:
            path = PR.find_parsed(seq, "full", model)
            if not path:
                continue
            preds = read(Path(path) / "predictions.json")
            preds = preds if isinstance(preds, list) else preds.get("predictions", [])
            for _, fam in T28.FAMS:
                pairs = []
                for r in preds:
                    feat = features.get(PR.key(r))
                    if not feat or feat["fam"] != fam or feat["v3"] is None:
                        continue
                    try:
                        p = float(r["vlm_answer"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(p):
                        pairs.append((p, feat["v1"], feat["v3"]))
                if not pairs:
                    continue
                p, g1, g3 = map(np.array, zip(*pairs))
                r1, r3 = rho(p, g1), rho(p, g3)
                rows.append({"sequence": seq, "model": model, "family": fam, "n_common": len(p),
                             "rho_v1": r1, "rho_v3": r3,
                             "delta_rho": r3-r1 if r1 is not None and r3 is not None else None})
    return {"scope": "Direct full-frame prediction rank comparison against both GT versions on each common finite item set; seven TeamTrack clips. No hypothesis test/equivalence claim.", "cells": rows}


def contacts():
    import cv2
    sheets = []
    for seq in ("Q1_side_0-30", "Q1_top_0-30"):
        src = next((ROOT / "data/external_validation/teamtrack").rglob(seq)) / "img1.mp4"
        cap = cv2.VideoCapture(str(src))
        # Local packed MP4s use 30 fps even when seqinfo.ini gives 23.976 fps.
        # Compare original sequence time by source frame number, not MP4 time.
        original_fps = float(DQ.load_seqinfo(src.parent)["fps"])
        tiles = []
        for sec in (0, 5, 10, 15, 20, 25):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(sec * original_fps))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {src}@{sec}")
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(frame, f"{seq}: {sec}s", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            tiles.append(frame)
        cap.release()
        canvas = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
        path = OUT / ("evidence_" + seq + "_contact.jpg")
        cv2.imwrite(str(path), canvas)
        sheets.append(str(path.relative_to(ROOT)))
    return sheets


def report():
    stats_data = read(OUT / "evidence_paired_statistics.json")
    rows = [r for r in stats_data["cells"] if r["model"] in ("sft", "grpo", "tsft", "cdnative")]
    names = {"sft": "source-SFT", "grpo": "GRPO-v3", "tsft": "target-SFT-token", "cdnative": "CourtDyn-native (legacy)"}
    lines = ["# Paper2 数据、坐标与统计 double check（2026-09-12）", "",
             "本报告重放冻结输入/预测及本地 processor，未训练或推理模型。审后区间属于探索性敏感性分析，不是预注册等效检验。", "",
             "## 已确认的错误与修复", "",
             "1. **Q1 事件重叠已获得原视频与轨迹证据。** 六个原始时间点的联排显示同一开场争球、同篮进攻与转换。58 个半秒采样点、旋转 top 球场方向后，群体质心 x 相关 0.999928，y 相关 0.933668；质心中位距离 0.783 m。精确相机时钟偏移与个体身份未认证，以上量用于事件对应，不是额外标定精度认证。", 
             "   默认 builder 已排除 Q1_side_0-30，生成新的 event_holdout 1120 QA（3侧视+Q4_top，每段280）；原1400 QA SHA未变。新文件仅为训练输入，尚无事件隔离重训结果。legacy-cross-view 显式模式标注事件污染、写独立新文件。", 
             "2. **CourtDyn-native 的1400训练标签全部原样来自 v1。** 逐项问题和答案完全匹配，不是 v3 标签训练。", 
             "3. **Human-M3 的原帧确有重叠。** paper1的800 QA有712个独立(camera,frame)；paper2正文四个basketball1/split1机位的224源帧中73个与之重合（19/15/18/21），映射到实际原图路径逐一相同。两篇问题、渲染形式与研究目标不同，但不能宣称零数据重叠。", 
             "4. **597×336不是本地 processor 实测尺寸。** 当前 transformers 5.14.1 / torch 2.11.0+cu128、同一Qwen3.5-4B快照、max_pixels=200704，按run_bench完整apply_chat_template，四幅960×540图得到每幅grid=[1,20,36]，patch=16，即576×320。历史运行没保存grid，只能称当前环境重放。原oracle确实固定用597×336，原分数不能直接改称576×320结果。", 
             "5. **rho完全不下降是错误。** legacy-native/Q2/speed 从0.76409到0.70405；5s块Δrho区间[-0.08888,-0.01071]、10s块[-0.09316,-0.00733]。可称仍有正相关，不能称不变。三继承adapter的full v3范围为0.34–0.80，旧0.53–0.80漏了target-SFT Q1/speed。", "",
             "## 统计口径与边界", "",
             "输入为两条top片段冻结qa_dyn_v1及full/ruler/pxunit已解析预测。配对键是(category,window_index,track)，取两臂与对应GT共同有效且有限的预测；比值另要求full>0。四个adapter每家族每臂N=140。full/ruler按重新计算v3 GT；pxunit按960像素宽渲染坐标GT；另存两臂都相对v3的Δrho。IQR用NumPy线性分位数。", "",
             "时间块由原始window起点/seqinfo fps落入固定5秒或10秒区间，同一块所有球员一起抽样，两臂使用相同抽样下标。每片段仅6或3块、2000次、seed=20260912。边界仍可能共享帧，轨迹也跨块；这些是小样本内敏感性区间，不能给独立比赛推广提供保证，也未预设等效界。部分中位比区间塌为一个值源于离散输出和大量并列，不代表零不确定性。", "",
             "base有效覆盖很低：配对交集N=5–77；Q1像素speed仅6条。不能将这些条件交集的rho与原本全有效题目的rho混用。常数输出rho保持未定义，不替换为0。", "",
             "## 配对比值与分位数", "",
             "R = median(pred_arm_i / pred_full_i)，[]为逐题比值的第1/第3四分位；不是预测中位数之比。下表各行两个干预臂的N均为140。", "",
             "|片段|adapter|家族|N|ruler R [Q1,Q3]|pxunit R [Q1,Q3]|", "|---|---|---|---:|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering\scriptsize", r"\caption{Paired prediction ratios and interquartile ranges. Each cell uses 140 common finite predictions with positive full-arm denominator. Ratios are arm/full itemwise ratios; the native adapter is the historical cross-view model with Q1 event overlap.}",
           r"\label{tab:audit_ratios}", r"\begin{tabular}{lllrrr}", r"\toprule", r"Clip & Adapter & Family & $N$ & Ruler $R$ [Q1,Q3] & Pixel $R$ [Q1,Q3] \\", r"\midrule"]
    for p in [r for r in rows if r["arm"] == "pxunit"]:
        r = next(x for x in rows if x["arm"] == "ruler" and all(x[k] == p[k] for k in ("sequence", "model", "family")))
        def rq(x):
            return f"{x['ratio_median']:.2f} [{x['ratio_q1_q3'][0]:.2f},{x['ratio_q1_q3'][1]:.2f}]"
        clip = "Q1" if p["sequence"].startswith("Q1") else "Q2"
        vals = [clip, names[p["model"]], p["family"], str(p["n_positive_full_ratio"]), rq(r), rq(p)]
        lines.append("|" + "|".join(vals) + "|")
        tex.append(" & ".join(vals) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    lines += ["", "## 两种GT的直接逐格核对", "", "以下为同一完整预测/共同有效样本的原始rho，避免以GT彼此高相关代替重算。全七片段结果见evidence_rank_versions.json；下面补全核心四个adapter。", "",
              "|片段|adapter|家族|N|full rho(v1)|full rho(v3)|Δ|", "|---|---|---|---:|---:|---:|---:|"]
    for r in [r for r in rows if r["arm"] == "pxunit"]:
        lines.append(f"|{r['sequence']}|{names[r['model']]}|{r['family']}|{r['n_common']}|{r['full_rho_v1_same_items']:.4f}|{r['full_rho_v3']:.4f}|{r['full_rho_v3']-r['full_rho_v1_same_items']:+.4f}|")
    lines += ["", "## 仍需新增实验的边界", "",
              "- 用新1120 QA事件隔离清单从base重新训练独立checkpoint，再完整评估两条top片段；旧native结果保留历史交叉机位解释，不能改名替代重训。",
              "- 明确960×540像素坐标的提示与当前processor网格，加入米/厘米转换、给定数字/K的算术阳性控制；这些新结果尚未生成。",
              "- 若要报告真正模型输入分辨率下的oracle，按记录的真实processor尺寸重新运行；此次没有重算原597×336 oracle分数。",
              "- 需要更多独立事件/比赛及预设等效界才能支持不下降/跨比赛推广；现有3或6个块的探索区间只支持有限敏感性检查。", "",
              "## 文件与重放", "", "- `tools/audit_paper2_evidence.py`：CPU审计脚本，主运行包括2000次敏感性重采样。`--skip-bootstrap`复用已有统计文件重建报告，`--processor`额外重放processor。",
              "- `evidence_native_split.json` / `evidence_event_alignment.json` / Q1原帧联排：旧训练来源与同事件证据。",
              "- `evidence_human_m3_overlap.json`：逐机位交集源帧编号及实际路径相等校验。",
              "- `evidence_processor.json`：版本、快照、原始尺寸、processor完整配置与四图grid。",
              "- `evidence_paired_statistics.json`：48格覆盖率、配对N/IQR、两种时间块区间及有效bootstrap数。",
              "- `evidence_rank_versions.json`：七片段、公开模型与source/GRPO逐格v1/v3 rank比较。",
              "- `evidence_paired_ratio_table.tex`：可插入附录的16行比值表。", ""]
    (OUT / "evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "evidence_paired_ratio_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--processor", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true", help="reuse existing paired statistics; do not overwrite them")
    ap.add_argument("--model", default=r"E:/models/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("native_split", native), ("human_m3_overlap", human_m3), ("event_alignment", event_alignment), ("rank_versions", rank_versions)]
    if not args.skip_bootstrap:
        jobs.append(("paired_statistics", lambda: stats(args.bootstrap_reps)))
    for name, fn in jobs:
        print(save(name, fn()), flush=True)
    print(json.dumps(contacts()), flush=True)
    if args.processor:
        print(save("processor", processor(args.model)), flush=True)
    report()
    print("paper2/audit/evidence_summary.md", flush=True)


if __name__ == "__main__":
    main()
