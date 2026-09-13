#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t30_queue.py — CourtDyn T30 (N6): 把首帧x4 补在第二场地 ρ 最高的那一格上。

为什么只有 2 格却值得单独排一次队:
  T29 里首帧x4 只跑在 split1 的 camera_0 / camera_2 上, 而那两台相机的 full ρ 只有 0.17-0.40 ——
  按 T26 在 CourtDyn 上总结的规律, 这是 "本来就没有东西可掉" 的区间, Δρ 无论多少都没有判别力
  (实测 -0.21 ... +0.24, 正负都有)。
  T29 全表 ρ 最高的一格是 **basketball2 / camera_0** (SFT 0.48/0.50, GRPO 0.44/0.46), 恰恰没跑
  首帧x4。把对照补在这一格上, 才是在第二场地上真正检验 "多帧确实被读到"。

预注册判据 (写在跑之前, 见 docs/courtdyn_results_summary_2026-09-05.md 的 N6):
  Δρ(full − static4) > 0.3  => E1a 的核心干预跨场地成立, 红线 8 (单一场地) 可以放宽;
  Δρ ≈ 0                    => E1a 只在 CourtDyn 俯视上成立, 红线 8 维持, 必须写进 Limitations。

格子: hm3_basketball2_camera_0 x {source-SFT, GRPO-v3} static4, 共 2 个评测格 (~20 min GPU)。
full 臂已在 T29 落盘, 不重跑。汇总沿用 summarize_courtdyn_hm3.py —— 它的 C 节按目录存在与否
枚举, 新格子落盘后自动进表, 不需要改汇总脚本。

状态 results/courtdyn_t30_state.json; 日志 results/_t30_*.log。
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
SEQ = "hm3_basketball2_camera_0"


def log(msg):
    print(f"[t30 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t30_{seq}_{pool}_{mode}_{model}"


Q.log = log
Q.cell = cell
Q.STATE = os.path.join(RES, "courtdyn_t30_state.json")
Q.T25_STATE = os.path.join(RES, "courtdyn_t29_state.json")   # 等待目标: T29 (已结束, 立即放行)
Q.T25_LAST_STAGE = "summary_t29"
Q.POOLS = dict(Q.POOLS, main=("qa_dyn_v1.json", None, None))  # hm3 池没有成对家族

Q.CELLS = [(SEQ, "main", "static4", ["sft", "grpo"])]

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})
_t29_done_orig = Q.t25_done


def _t29_done():
    ok, why = _t29_done_orig()
    return ok, why.replace("T25", "T29")


Q.t25_done = _t29_done


def summary_stage():
    """汇总产物 (courtdyn_hm3_table.md) 已由 T29 写过, 这里必须强制重跑, 否则新格子进不了 C 节。

    做法: done_file 仍指向**真正的产物**, 但在建阶段时把 T29 留下的旧表删掉 ——
    这样 run_stage 的前置 "产物已存在, 跳过" 不触发, 而跑完后的产物校验依旧有效。
    (不要拿一个永不存在的哨兵当 done_file: 前置检查是过了, 但后置校验必然失败,
     结果是汇总脚本被空跑 MAX_RESTARTS 次然后判 failed —— 9/6 就是这么踩的。)
    """
    summ = os.path.join(TOOLS, "summarize_courtdyn_hm3.py")
    table = os.path.join(CD, "courtdyn_hm3_table.md")
    if os.path.isfile(table):
        os.remove(table)
        log("summary_t30: 已删除 T29 留下的 courtdyn_hm3_table.md, 强制重算")
    return Q.Stage("summary_t30", [Q.PY, "-u", summ], table, "_t30_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t30_queue", log)
    gpu = Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t30_")
    log(f"T30 (N6) 启动: {len(gpu)} 个评测/解析阶段 ({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), "
        f"1 个汇总; 目标 = {SEQ} 的首帧x4 对照 (full 臂已在 T29 落盘)")
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
    log("T30 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
