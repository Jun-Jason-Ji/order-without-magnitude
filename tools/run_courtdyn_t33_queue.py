#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t33_queue.py — CourtDyn T33 (N8): CourtDyn 原生适配器的递尺子 / 换单位臂。

背景
----
主结论 (§一 E1b-T28 "给了不用, 换了不动") 此前只有 base + 论文 1 的三个适配器 + 公开 7B,
而 base 被 ruler 提示扰乱、7B 是退化输出 —— **"有序无量"里肯定的那半没有外部复现**。
盘上又没有第四个能跑的多模态模型 (9B 装不下, SmolVLM2 答字母)。

于是自己训了一个: `models/courtdyn-native-sft`, **从 base 起** (`--allow_base_init`),
训练片段 = 4 条侧视 + Q4_top, **Q1_top / Q2_top 完全留出**。

筛查已通过 (2026-09-08, 留出片段 Q1_top, v3 口径):
    speed ρ = 0.706 (distinct 11), path ρ = 0.680 (distinct 13)
    —— 高于论文 1 的两个适配器的 speed (0.59), 远离 7B 的退化区 (ρ 0.22/0.06, distinct 1-2)。
所以它满足"非退化独立臂"的前提, 可以拿来问同样的问题。

格子: Q1_top / Q2_top × {main(full 基线), ruler, pxunit} × cdnative = 6 格 (~1.5 GPU-h)。
main 臂是 ruler/pxunit 比值的分母, 必须同模型同片段自己跑一份。
Q1_top 的 main 臂已有 partial (筛查时跑的), --resume 会接着跑完。

汇总沿用 summarize_courtdyn_t28.py (已把 cdnative 加进 MODELS, t28_parsed 认 t33_ 前缀)。
状态 results/courtdyn_t33_state.json; 日志 results/_t33_*.log。
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_courtdyn_t26_queue as Q   # noqa: E402

RES = os.path.join(ROOT, "results")
CD = os.path.join(RES, "courtdyn")
TOPS = ["Q1_top_0-30", "Q2_top_480-510"]


def log(msg):
    print(f"[t33 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t33_{seq}_{pool}_{mode}_{model}"


Q.log = log
Q.cell = cell
Q.STATE = os.path.join(RES, "courtdyn_t33_state.json")
Q.POOLS = dict(
    Q.POOLS,
    ruler=("qa_dyn_v1_ruler.json", None, None),
    pxunit=("qa_dyn_v1_pxunit.json", None, None),
)

# main 先跑 (它是比值的分母), 再跑两条干预臂
Q.CELLS = [(s, "main", "full", ["cdnative"]) for s in TOPS]
Q.CELLS += [(s, arm, "full", ["cdnative"]) for s in TOPS for arm in ("ruler", "pxunit")]

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})


def summary_stage():
    """强制重算 T28 汇总表, 否则 cdnative 的行进不去。删旧表 + done_file 指真产物 (勿用哨兵)。"""
    summ = os.path.join(TOOLS, "summarize_courtdyn_t28.py")
    table = os.path.join(CD, "courtdyn_t28_table.md")
    if os.path.isfile(table):
        os.remove(table)
        log("summary_t33: 已删除旧的 courtdyn_t28_table.md, 强制重算 (cdnative 行才会进表)")
    return Q.Stage("summary_t33", [Q.PY, "-u", summ], table, "_t33_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t33_queue", log)
    gpu = Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t33_")
    log(f"T33 启动: {len(gpu)} 个评测/解析阶段 ({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), 1 个汇总; "
        f"模型 = courtdyn-native-sft (从 base 训, 非论文 1 适配器)")
    Q.save_state()
    for sg in gpu + [summary_stage()]:
        try:
            Q.run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            Q.set_stage(sg.name, status="failed", note=repr(e))
    Q.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Q.save_state()
    log("T33 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
