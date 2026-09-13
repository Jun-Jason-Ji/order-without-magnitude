#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t32_queue.py — CourtDyn T32: 给主结论补一条**不属于论文 1 的**模型臂。

为什么必须补
------------
T28 (递尺子 / 换单位) 是本篇最强的一格, 但它的模型只有 base / source-SFT / GRPO-v3 /
target-SFT-token —— **后三个全是论文 1 的适配器**, 而 base 在 ruler 臂里是被提示扰乱的
(ρ 从 +0.11 掉到 −0.19), 不能算独立复现。也就是说主结论目前完全建立在姊妹篇的产物上。
审稿人一句 "你最强的结论只讲了三个来自未发表姊妹篇的适配器" 就能戳中。

本队列把公开权重 Qwen2.5-VL-7B 放进同样两条臂: 2 片段 x {ruler, pxunit} = 4 格 (~1 GPU-h)。
它的 full 臂在 T26/T27 已落盘, 直接复用, 不重跑。

预期与如实写法
--------------
7B 在 2x 缩放臂里是**退化**的 (两臂都恒定输出 1.5, 比值 1.00, 见红线 12), 换单位臂大概率同样退化。
那不是坏消息, 但证据强度要写清楚:
  * 若米制答 ~1.5、像素制仍答 ~1.5  ⇒ 比值 ~1, 同样说明没做单位换算 —— 主结论多一条独立臂,
    但必须注明这是**退化输出下的 ~1**, 与适配链"有变化但只挪 1-2 倍"不是同一强度的证据;
  * 若像素制明显抬高 (接近 K≈27) ⇒ 公开模型会换算而适配链不会, 那是个更有意思的结果, 论点要改写。
无论哪种, 都不能把 7B 的 ~1 直接当作"它也不换算"的强证据 —— 判据只在预测确有变化时才满格有效。

格子: Q1_top_0-30 / Q2_top_480-510 x {ruler, pxunit} x qwen25vl7b。
汇总沿用 summarize_courtdyn_t28.py (已把 7B 加进 MODELS, 且 t28_parsed 同时认 t28_ / t32_ 前缀)。
排在 T31 之后, 不抢卡。状态 results/courtdyn_t32_state.json; 日志 results/_t32_*.log。
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
    print(f"[t32 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t32_{seq}_{pool}_{mode}_{model}"


Q.log = log
Q.cell = cell
Q.STATE = os.path.join(RES, "courtdyn_t32_state.json")
Q.T25_STATE = os.path.join(RES, "courtdyn_t31_state.json")   # 等 T31 让出卡
Q.T25_LAST_STAGE = "summary_t31"
Q.POOLS = dict(
    Q.POOLS,
    ruler=("qa_dyn_v1_ruler.json", None, None),
    pxunit=("qa_dyn_v1_pxunit.json", None, None),
)

Q.CELLS = [(s, arm, "full", ["qwen25vl7b"]) for s in TOPS for arm in ("ruler", "pxunit")]

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})
_t31_done_orig = Q.t25_done


def _t31_done():
    ok, why = _t31_done_orig()
    return ok, why.replace("T25", "T31")


Q.t25_done = _t31_done


def summary_stage():
    """T28 汇总表已存在, 必须强制重算才能把 7B 的新行并进去。

    做法与 T30 相同: done_file 指向真产物, 但先删掉旧表 —— 不要用永不存在的哨兵当 done_file,
    那样前置检查过了、后置校验必挂, 汇总脚本会被空跑 MAX_RESTARTS 次然后判 failed。
    """
    summ = os.path.join(TOOLS, "summarize_courtdyn_t28.py")
    table = os.path.join(CD, "courtdyn_t28_table.md")
    if os.path.isfile(table):
        os.remove(table)
        log("summary_t32: 已删除旧的 courtdyn_t28_table.md, 强制重算 (7B 行才会进表)")
    return Q.Stage("summary_t32", [Q.PY, "-u", summ], table, "_t32_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t32_queue", log)
    gpu = Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t32_")
    log(f"T32 启动: {len(gpu)} 个评测/解析阶段 ({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), "
        f"1 个汇总; 目标 = 公开 7B 的 ruler / pxunit 臂; 先等 T31 结束")
    Q.save_state()
    if "--no_wait" not in sys.argv:
        Q.wait_for_t25()
    for sg in gpu + [summary_stage()]:
        try:
            Q.run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            Q.set_stage(sg.name, status="failed", note=repr(e))
    Q.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Q.save_state()
    log("T32 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
