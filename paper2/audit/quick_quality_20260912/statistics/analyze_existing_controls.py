"""Post-hoc CPU-only analyses of frozen CourtDyn predictions.

Run from any directory with the existing research Python (numpy/scipy).
Writes only beside this script. No training, inference, or frozen-file edits.
The two analyses are auxiliary no-prior decomposition and within-trajectory
speed/path consistency; neither supplies independent-event replication.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime

import numpy as np
import scipy
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "engine/dynamics_qa.py").exists())
sys.path.insert(0, str(ROOT))
from tools import summarize_courtdyn_t28 as T
from tools.build_courtdyn_noprior import PRIOR_RE
from tools.build_courtdyn_ruler import tt_split
from engine import dynamics_qa as DQ
from engine.court_homography import CourtPlane

CD = ROOT / "results/courtdyn"
SPEED, PATH = "dynamics_speed_player", "dynamics_path_player"
INPUTS = {}


def record(path):
    p = Path(path)
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(path):
    return json.loads(record(path).read_text(encoding="utf-8"))


def summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"n": len(a), "median": float(np.median(a)) if len(a) else None,
            "q1_q3": np.quantile(a, [.25, .75]).tolist() if len(a) else None,
            "min_max": [float(a.min()), float(a.max())] if len(a) else None}


def rho(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def load_preds(directory):
    directory = Path(directory)
    raw = read(directory / "predictions.json")
    rows = raw if isinstance(raw, list) else raw["predictions"]
    rows = [x for x in rows if x["category"] in (SPEED, PATH)]
    keyed = {T.ikey(x): x for x in rows}
    assert len(keyed) == len(rows), "duplicate category/window/track keys"
    finite = {}
    for k, row in keyed.items():
        try:
            v = float(row["vlm_answer"])
            if math.isfinite(v):
                finite[k] = v
        except (KeyError, ValueError, TypeError):
            pass
    config_path = Path(str(directory).removesuffix("_parsed")) / "run_config.json"
    config = read(config_path) if config_path.exists() else None
    return keyed, finite, config


def image_ids(row):
    return row.get("inference_image_ids") or row.get("image_ids")


def common_score(pred, gt, keys, family, k=1):
    return {"n": len(keys), "rho": rho([pred[x] for x in keys], [gt[x] for x in keys]),
            "tmra": T.tmra([(pred[x], gt[x]) for x in keys], family, k)}


def remove_prior(question):
    return re.sub(r"\s{2,}", " ", PRIOR_RE.sub(" ", question)).strip()


def main():
    records = []
    trajectories = []
    truth_checks = []
    configs = []
    inventory = {"native_prediction_directories": sorted(p.parent.name for p in CD.glob("*cdnative*/predictions.json")),
                 "native_noprior_prediction_files": sorted(str(p.relative_to(ROOT)) for p in CD.rglob("*noprior*/*predictions*.json") if "cdnative" in str(p)),
                 "auxiliary_noprior_directories": sorted(p.parent.name for p in CD.glob("*noprior*_parsed/predictions.json"))}
    for seq in T.TOPS:
        pool = read(CD / f"seq_{seq}/qa_dyn_v1.json")
        items = {T.ikey(x): x for x in pool if x["category"] in (SPEED, PATH)}
        ctx = T.seq_ctx(seq)
        px_pool = {T.ikey(x): x for x in read(CD / f"seq_{seq}/qa_dyn_v1_pxunit.json")}
        read(CD / f"seq_{seq}/qa_dyn_v1_ruler.manifest.json")
        read(CD / f"homography/H_{seq}.json")
        tt = Path(tt_split(seq))
        record(tt / "seqinfo.ini")
        record(tt / "gt/gt.txt")
        tracks = DQ.load_tracks(tt)
        fps = float(DQ.load_seqinfo(tt)["fps"])
        plane = CourtPlane.load(seq)
        pairs = sorted((k[1], k[2]) for k in items if k[0] == SPEED and (PATH, k[1], k[2]) in items)
        exact_errors, rounded_errors, v1_errors, v1_exact_errors, pixel_errors, durations = [], [], [], [], [], []
        images_match = []
        for w, tr in pairs:
            sk, pk = (SPEED, w, tr), (PATH, w, tr)
            si, pi = items[sk], items[pk]
            assert si["meta"]["window"] == pi["meta"]["window"]
            assert si["meta"]["frames"] == pi["meta"]["frames"]
            images_match.append(si["image_ids"] == pi["image_ids"])
            f0, f1 = si["meta"]["window"]
            dur = (f1 - f0) / fps
            durations.append(dur)
            d = plane.path_length_m(tracks[tr], f0, f1)
            v = plane.mean_speed(tracks[tr], f0, f1, fps)
            exact_errors.append(abs(d - dur * v))
            v1_d = DQ.path_m(tracks[tr], f0, f1)
            v1_v = DQ.mean_speed(tracks[tr], f0, f1, fps)
            v1_exact_errors.append(abs(v1_d - dur * v1_v))
            rounded_errors.append(abs(ctx["v3"][pk] - dur * ctx["v3"][sk]))
            v1_errors.append(abs(float(pi["answer"]) - dur * float(si["answer"])))
            pixel_errors.append(abs(ctx["px"][pk] - dur * ctx["px"][sk]))
        rounded_px_errors = [abs(float(px_pool[k]["answer"]) - ctx["px"][k]) for k in px_pool]
        truth_checks.append({"sequence": seq, "n_speed": sum(k[0] == SPEED for k in items),
            "n_path": sum(k[0] == PATH for k in items), "paired_trajectories": len(pairs),
            "union_speed_path_trajectories": len({k[1:] for k in items}),
            "all_speed_path_input_images_identical": all(images_match), "duration_seconds": summary(durations),
            "v3_unrounded_abs_path_minus_duration_speed": summary(exact_errors),
            "v1_unrounded_abs_path_minus_duration_speed": summary(v1_exact_errors),
            "v3_rounded_abs_path_minus_duration_speed": summary(rounded_errors),
            "v1_rounded_abs_path_minus_duration_speed": summary(v1_errors),
            "pixel_unrounded_abs_path_minus_duration_speed": summary(pixel_errors),
            "pixel_stored_answer_vs_recomputed_same_960_canvas_abs_error": summary(rounded_px_errors),
            "rounding_error_bound": .05 * (1 + max(durations)),
            "all_exact_v1_v3_and_pixel_identities_hold": max(exact_errors + v1_exact_errors + pixel_errors) < 1e-10,
            "all_rounded_v1_and_v3_identities_within_rounding": max(v1_errors + rounded_errors) <= .05 * (1 + max(durations)) + 1e-9,
            "all_stored_pixel_answers_agree_to_rounding": max(rounded_px_errors) <= .05000001})

        # Auxiliary models only: common finite triplets, no extrapolation to native.
        for model in ("sft", "grpo"):
            no_prefix = "t31" if seq.startswith("Q1") else "t26"
            paths = {"full": T.full_parsed(seq, model),
                     "noprior": CD / f"{no_prefix}_{seq}_noprior_full_{model}_parsed",
                     "pxunit": T.t28_parsed(seq, "pxunit", "full", model)}
            loaded = {a: load_preds(p) for a, p in paths.items()}
            config_fields = ["model_path", "adapter", "load_4bit", "max_pixels", "generation", "frame_mode", "blank_images", "reference_tag", "reference_text"]
            config_check = {f: len({json.dumps(x[2].get(f), sort_keys=True) for x in loaded.values()}) == 1 for f in config_fields}
            configs.append({"sequence": seq, "model": model, "all_three_configs_available": all(x[2] for x in loaded.values()), "matching_non_prompt_fields": config_check})
            for category, family in ((SPEED, "speed"), (PATH, "path")):
                keys = sorted(k for k in items if k[0] == category and k in ctx["v3"] and k in ctx["px"] and all(k in x[1] for x in loaded.values()))
                full, no, px = (loaded[a][1] for a in ("full", "noprior", "pxunit"))
                metadata_checks = []
                for k in keys:
                    f, n, p = (loaded[a][0][k] for a in ("full", "noprior", "pxunit"))
                    metadata_checks.append({"key": list(k), "same_images": image_ids(f) == image_ids(n) == image_ids(p),
                       "noprior_only_removes_height_sentence": remove_prior(f["question"]) == n["question"],
                       "same_windows_and_tracks": all(f["meta"].get(a) == n["meta"].get(a) == p["meta"].get(a) for a in ["window", "frames", "track", "fps"]),
                       "height_sentence_absent_in_noprior_and_pxunit": not PRIOR_RE.search(n["question"]) and not PRIOR_RE.search(p["question"])})
                full_score = common_score(full, ctx["v3"], keys, family)
                no_score = common_score(no, ctx["v3"], keys, family)
                records.append({"sequence": seq, "model": model, "family": family, "n_common": len(keys),
                    "source_directories": {a: str(Path(p).relative_to(ROOT)) for a, p in paths.items()},
                    "all_inputs_verified": all(all(v for name, v in c.items() if name != "key") for c in metadata_checks),
                    "failed_input_checks": [c for c in metadata_checks if not all(v for name, v in c.items() if name != "key")],
                    "full_v3": full_score, "noprior_v3": no_score,
                    "noprior_minus_full_v3_tmra": no_score["tmra"] - full_score["tmra"],
                    "pxunit_pixel": common_score(px, ctx["px"], keys, family, ctx["K"]),
                    "pxunit_v3_same_truth_rank": rho([px[k] for k in keys], [ctx["v3"][k] for k in keys]),
                    "noprior_over_full": summary([no[k]/full[k] for k in keys if full[k] > 0]),
                    "pxunit_over_full": summary([px[k]/full[k] for k in keys if full[k] > 0]),
                    "pxunit_over_noprior": summary([px[k]/no[k] for k in keys if no[k] > 0]),
                    "reference_K_rendered_px_per_m": ctx["K"]})

        native_paths = {"full": T.full_parsed(seq, "cdnative"),
                        "ruler": T.t28_parsed(seq, "ruler", "full", "cdnative"),
                        "pxunit": T.t28_parsed(seq, "pxunit", "full", "cdnative"),
                        "static4": CD / f"t34_{seq}_main_static4_cdnative_parsed"}
        native_cohorts = {}
        native_loaded = {}
        for arm, directory in native_paths.items():
            rows, pred, config = load_preds(directory)
            native_loaded[arm] = (rows, pred, config)
            paired = [(w, tr) for w, tr in pairs if (SPEED, w, tr) in pred and (PATH, w, tr) in pred]
            native_cohorts[arm] = set(paired)
            quotients, errors, within, speeds, paths_, nominal_ratios, nominal_within = [], [], [], [], [], [], []
            for w, tr in paired:
                sk, pk = (SPEED, w, tr), (PATH, w, tr)
                s, p = pred[sk], pred[pk]
                f0, f1 = items[sk]["meta"]["window"]
                dur = (f1-f0)/fps
                assert image_ids(rows[sk]) == image_ids(rows[pk])
                err = abs(p-dur*s)
                ratio = p/(dur*s) if s > 0 else None
                if ratio is not None: quotients.append(ratio)
                errors.append(err); within.append(err <= .05 * (1+dur) + 1e-9)
                nominal_within.append(abs(p - 2.0*s) <= .05 * (1+2.0) + 1e-9)
                if s > 0: nominal_ratios.append(p/(2.0*s))
                speeds.append(s); paths_.append(p)
            trajectories.append({"sequence": seq, "model": "cdnative", "arm": arm, "n_matched_trajectories": len(paired),
                "n_pool_speed_path_pairs": len(pairs),
                "n_finite_speed_in_pool": sum(k[0] == SPEED and k in items for k in pred),
                "n_finite_path_in_pool": sum(k[0] == PATH and k in items for k in pred),
                "source": str(Path(directory).relative_to(ROOT)), "units": "pixels" if arm == "pxunit" else "metres",
                "path_over_duration_speed": summary(quotients), "abs_path_minus_duration_speed": summary(errors),
                "within_one_decimal_rounding_envelope_count": sum(within),
                "within_one_decimal_rounding_envelope_fraction": sum(within)/len(within),
                "nominal_2s_sensitivity": {"path_over_duration_speed": summary(nominal_ratios),
                    "rounding_envelope": .15, "rounding_consistent_count": sum(nominal_within),
                    "same_itemwise_consistency_classification_as_actual_duration": within == nominal_within},
                "cross_question_spearman": rho(speeds, paths_), "all_paired_images_identical": True})
        assert native_cohorts["full"] == native_cohorts["pxunit"], "full/pxunit cohorts differ"
        for w, tr in native_cohorts["full"]:
            for cat in (SPEED, PATH):
                k = (cat, w, tr)
                f, p = native_loaded["full"][0][k], native_loaded["pxunit"][0][k]
                assert image_ids(f) == image_ids(p) == items[k]["image_ids"]
                assert all(f["meta"][a] == p["meta"][a] == items[k]["meta"][a] for a in ["window", "frames", "track", "fps"])
        assert all(native_loaded["full"][2][f] == native_loaded["pxunit"][2][f] for f in config_fields)
        for x in trajectories:
            if x["sequence"] == seq:
                x["full_and_pxunit_use_identical_trajectory_cohort"] = True
                x["full_pxunit_source_images_metadata_and_non_prompt_config_verified"] = True

    existing = read(ROOT / "paper2/audit/evidence_paired_statistics.json")
    native_ci = [x for x in existing["cells"] if x["model"] == "cdnative" and x["arm"] == "pxunit"]
    for path in ["tools/summarize_courtdyn_t28.py", "tools/build_courtdyn_noprior.py", "tools/build_courtdyn_ruler.py", "engine/dynamics_qa.py", "engine/court_homography.py", "eval/evaluate.py", "results/courtdyn/teamtrack_probe_hits.json"]:
        record(ROOT / path)
    result = {"schema": "paper2-existing-controls-posthoc-v1", "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "PASS", "scope": "Post-hoc descriptive CPU-only analysis of existing predictions. No new independent events, model runs, bootstrap analyses, or causal mechanism identification.",
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__},
        "inventory": inventory, "auxiliary_configs": configs, "auxiliary_noprior_decomposition": records,
        "truth_and_pairing_checks": truth_checks, "native_cross_question_consistency": trajectories,
        "existing_native_pixel_temporal_intervals": native_ci,
        "existing_ci_method": existing["bootstrap"], "input_sha256": INPUTS,
        "cautions": ["No native no-prior prediction exists in the examined results; auxiliary deletion controls cannot establish native robustness.",
            "No-prior-to-pixel removes the height-deletion confound in the auxiliary paired comparison, but still changes requested quantity/units and omits coordinate canvas information.",
            "Speed and path are two questions over the same trajectories, not independent replications. Exact durations follow source fps and are 2.002 s, while the prompt rounds to 2.0 s.",
            "Prediction consistency is not accuracy or temporal reasoning: constant answers can also be consistent.",
            "Ratios use positive denominators; N is reported separately. IQR is descriptive dispersion, not a confidence interval.",
            "Existing 5/10-s intervals are copied with provenance, not recomputed: only 6/3 bins, linked players and boundary windows remain, and no independent-match uncertainty is supported."]}
    assert all(x["all_inputs_verified"] for x in records)
    assert all(all(x["matching_non_prompt_fields"].values()) for x in configs)
    assert all(x["all_stored_pixel_answers_agree_to_rounding"] for x in truth_checks)
    assert all(x["all_exact_v1_v3_and_pixel_identities_hold"] and x["all_rounded_v1_and_v3_identities_within_rounding"] for x in truth_checks)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "quick_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    public = json.loads(json.dumps(result))
    public["runtime"].pop("executable", None)
    public["public_scope"] = "Aggregated results and source-file digests only; no per-item predictions, trajectory IDs or source imagery. Reproduction requires authorized local reconstruction of source QA/H and frozen-format predictions."
    (OUT / "quick_analysis_aggregate.json").write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "findings.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"native_noprior": inventory["native_noprior_prediction_files"], "auxiliary": [{k:x[k] for k in ["sequence", "model", "family", "n_common", "noprior_over_full", "pxunit_over_noprior", "noprior_minus_full_v3_tmra"]} for x in records], "native": [{k:x[k] for k in ["sequence", "arm", "n_matched_trajectories", "path_over_duration_speed", "within_one_decimal_rounding_envelope_count"]} for x in trajectories]}, indent=2))


def render_report(r):
    def fmt(s):
        return f"{s['median']:.3f} [{s['q1_q3'][0]:.3f}, {s['q1_q3'][1]:.3f}]"
    lines = ["# Paper2 现有预测的快速补强分析", "", "状态：PASS。审后新增、描述性 CPU 分析；没有新训练、推理、独立事件或预注册检验。",
        "", "## 可用证据与优先级", "",
        "1. **native 跨问题自洽性是可新增的主要敏感性分析。** 在同一轨迹、相同图片下，速度与路程应满足 path = 实际时长 × speed。这个恒等式在同一坐标体系中不受统一像素缩放因子影响，但模型可能对两个问题使用不同约定，因此不证明测量机制缺失。",
        "2. **no-prior 分解仅能补充辅助模型。** native 没有 no-prior 预测。source-SFT 和 GRPO-v3 各有 Q1/Q2 的米制 full、米制 no-prior 与 pxunit 三臂，可分离这些辅助模型的删身高句影响；不能声称 native 已通过同一控制。",
        "3. **既有 5/10 秒区间不重复生成。** 已有结果保留在 JSON 并记录原文件 SHA；每片段仅 6/3 个块，未提供独立比赛不确定性。新增跨问题配对队列与原逐家族 N=140 队列不同，不混用其区间。",
        "", "## 同轨迹配对与 GT", "",
        "|片段|speed 题|path 题|共同轨迹|轨迹并集|实际时长|", "|---|---:|---:|---:|---:|---:|"]
    for x in r["truth_and_pairing_checks"]:
        lines.append(f"|{x['sequence']}|{x['n_speed']}|{x['n_path']}|{x['paired_trajectories']}|{x['union_speed_path_trajectories']}|{x['duration_seconds']['median']:.3f} s|")
    lines += ["", "共同轨迹按 (window_index, track) 匹配，逐对验证 window、frames、图片相同；不能把各家族 140 题误称为 140 对完全相同的轨迹。两类任务部分共享题目轨迹，且都来自同一个短事件，不是独立复制。",
        "未舍入的 v1、v3 与像素 GT 均满足恒等式（最大绝对误差 < 1e-10）；已存 v1/v3 一位小数答案均落在舍入允许误差内。这是对已有配对轨迹的定义与代码一致性核对，不是新增训练数据、独立样本或训练实验。像素 GT 重算沿用同一 960×540 渲染画布，与冻结 pxunit 答案之差均不超过 0.05。实际时长 2.002 秒来自 60 帧 / 29.97002997 fps，题面写 2.0 秒属于显示舍入。",
        "", "## Native 预测的跨问题自洽性", "",
        "下表 S = pred_path / (duration × pred_speed)，理想自洽值为 1；[] 是逐项 IQR。N 为有两种有限预测的共同轨迹数；比值 N+ 另要求速度预测 > 0。自洽计数按一位小数舍入误差界 0.05×(1+duration)=0.1501 判定，不是任务准确率或预设等效界。",
        "", "|片段|条件|N|N+|S 中位数 [Q1,Q3]|舍入自洽计数|", "|---|---|---:|---:|---|---:|"]
    for x in r["native_cross_question_consistency"]:
        lines.append(f"|{x['sequence']}|{x['arm']}|{x['n_matched_trajectories']}|{x['path_over_duration_speed']['n']}|{fmt(x['path_over_duration_speed'])}|{x['within_one_decimal_rounding_envelope_count']}/{x['n_matched_trajectories']}|")
    lines += ["", "Q2 的 100 条共同轨迹中，full 的中位 S≈0.999，pxunit≈0.500；舍入自洽计数由 42/100 降至 4/100。该现象无法仅用两个答案共同乘上一个未知画布尺度因子来解释，但仍可能来自问题措辞、答案先验或单位解释差异。它补充了提示敏感性的描述，不能认定内部推理机制。Q1 有训练事件重叠；static4 存在大量零速度，正分母比值 N+ 明显小于共同 N，且常数答案也可能自洽，不能把自洽指标当成运动理解验证。",
        "额外的题面时长敏感性核对改用 2.0 秒和相应 0.15 舍入界；各格计数与逐项分类是否不变均记录在 JSON 的 nominal_2s_sensitivity 中。这是同一预测的时长约定敏感性，不是额外数据或新独立实验。",
        "", "## 辅助模型 no-prior 分解", "",
        "每格三臂共同有限 N=140，所有比值分母均正。原始图片、窗口、轨迹、模型/适配器、量化、像素预算、生成配置和 frame mode 相同；no-prior 问题逐字验证为 full 仅删去身高句。full/no-prior 同按 v3 真值评分；pxunit 沿用 960×540 像素真值，同时保存相对于相同 v3 真值的排序相关。",
        "", "|片段|模型|家族|no-prior/full R [Q1,Q3]|pxunit/no-prior R [Q1,Q3]|no-prior−full T-MRA(v3)|", "|---|---|---|---|---|---:|"]
    for x in r["auxiliary_noprior_decomposition"]:
        lines.append(f"|{x['sequence']}|{x['model']}|{x['family']}|{fmt(x['noprior_over_full'])}|{fmt(x['pxunit_over_noprior'])}|{x['noprior_minus_full_v3_tmra']:+.2f}|")
    lines += ["", "pxunit/no-prior 的八格中位比为 1.00–1.79，仍远小于名义 K≈27；因此在这两种辅助模型中，删去身高句不足以单独解释弱像素响应。删句本身并非没有作用：v3 T-MRA 变化从 −3.50 到 +4.93。像素臂仍改变问题措辞与所求量，且未说明画布坐标，不能将此结果写成纯单位转换因果实验。Q1/source-SFT/path 有一个较大的逐项比值尾部（最大 100），已在 JSON 保留，未裁剪或以中位数隐藏原始分布。",
        "", "## 仍须新实验的工作顺序", "",
        "- 第一优先：native 同图同题的 no-prior、固定提示的单位/坐标控制，以及已知数字换算阳性控制；直接针对当前最便宜但关键的未分离解释。",
        "- 第二优先：使用事件隔离切分重新训练，并在同一切分比较 v1/v3 监督；新 1,120 条清单不等于已有重训结果。",
        "- 第三优先：更多独立事件/比赛与具备任务能力的独立骨干；据此才能做有充分事件簇的区间估计，不能靠当前 3/6 个时间块补出外部泛化。",
        "", "## 重现", "",
        "运行本目录 analyze_existing_controls.py；使用现有研究 Python（NumPy、SciPy），默认只写本目录的聚合 JSON 与 findings.md，可用 --output-dir 指定新输出目录。JSON 保存全部输入 SHA-256、运行版本、三臂共同样本数、配置一致性与原始区间来源，不包含逐项预测或轨迹 ID。公开候选仅纳入脚本、quick_analysis_aggregate.json 和本报告。公开候选不附 QA、单应矩阵或逐项预测，复现者须在获得数据授权后按原格式在本地重建 QA/H 与预测输入；缺输入会明确报错，不能把缺输入伪称数值复现失败。主稿与冻结结果均未改动。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUT, help="New aggregate-output directory; frozen inputs are read only.")
    args = ap.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Required local input is absent: {exc.filename}. Reconstruct the authorized source QA/H and frozen-format predictions first; no model inference or training is performed by this script.") from exc
