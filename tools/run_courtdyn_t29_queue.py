#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t29_queue.py — CourtDyn T29 (N1): 第二场地 + 同一段运动的多相机视角。

题池由 tools/build_courtdyn_hm3.py 从 Human-M3 篮球序列建出 (另一个场地, 真三维世界 GT,
零下载)。它比 "再找一条片段" 强的地方在于: **同一 (窗口, 人物) 在同一序列的多台相机下是
同一段世界运动, GT 逐字相同, 只有相机几何和像素不同**。因此 E1a "视角决定序" 不再依赖
跨片段比较 —— 可以在同一批 item 上直接测。

格子 (按决定性排序):
  A. basketball1/split1 的 **全部 4 台相机** x {source-SFT, GRPO} full
     => 世界量固定, 只有视角变: ρ 若随相机大幅变化, E1a 就被同批 item 证实。
  B. split1 的 camera_0 / camera_2 x {SFT, GRPO} 首帧x4
     => 在第二场地上复现 "多帧确实被读到" 的对照。
  C. basketball2 (第二条序列, 3 相机里取 0 / 2) x {SFT, GRPO} full  => 跨序列复现。
  D. split1/camera_0 x {base, Qwen2.5-VL-7B} full => 参照臂。

复用 T26 队列模块的全部机制; 序列名形如 hm3_basketball1_split1_camera_0, 目录约定与
CourtDyn 一致 (results/courtdyn/seq_<name>/, data/courtdyn/frames_<name>/), 因此
T26 的 seq_dir / img_root 默认实现可直接用。开跑前等 T28 结束, 不抢卡。
状态 results/courtdyn_t29_state.json; 日志 results/_t29_*.log。
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
S1 = "hm3_basketball1_split1_camera_%d"
B2 = "hm3_basketball2_camera_%d"
TSFT = os.path.join(ROOT, "models", "target-sft-v3-token")


def log(msg):
    print(f"[t29 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t29_{seq}_{pool}_{mode}_{model}"


Q.log = log
Q.cell = cell
Q.STATE = os.path.join(RES, "courtdyn_t29_state.json")
Q.T25_STATE = os.path.join(RES, "courtdyn_t28_state.json")   # 等待目标: T28
Q.T25_LAST_STAGE = "summary_t28"
Q.MODELS = dict(Q.MODELS, tsft=(Q.QWEN, TSFT))
Q.POOLS = dict(Q.POOLS, main=("qa_dyn_v1.json", None, None))  # hm3 池没有成对家族

Q.CELLS = []
Q.CELLS += [(S1 % c, "main", "full", ["sft", "grpo"]) for c in (0, 1, 2, 3)]   # A
Q.CELLS += [(S1 % c, "main", "static4", ["sft", "grpo"]) for c in (0, 2)]      # B
Q.CELLS += [(B2 % c, "main", "full", ["sft", "grpo"]) for c in (0, 2)]         # C
Q.CELLS += [(S1 % 0, "main", "full", ["base", "qwen25vl7b"])]                  # D

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})
_t28_done_orig = Q.t25_done


def _t28_done():
    ok, why = _t28_done_orig()
    return ok, why.replace("T25", "T28")


Q.t25_done = _t28_done


def summary_stage():
    summ = os.path.join(TOOLS, "summarize_courtdyn_hm3.py")
    return Q.Stage("summary_t29", [Q.PY, "-u", summ],
                   os.path.join(CD, "courtdyn_hm3_table.md"), "_t29_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t29_queue", log)
    gpu = Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t29_")
    log(f"T29 启动: {len(gpu)} 个评测/解析阶段 ({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), "
        f"1 个汇总; 题池已由 build_courtdyn_hm3.py 建好; 先等 T28 结束再占卡")
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
    log("T29 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
