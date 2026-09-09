#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t27_queue.py — CourtDyn T27: 2× 缩放干预臂 + 侧视片段的首帧×4 对照 (paper 2)。

T26 判决: 俯视片段上 SFT / GRPO 的 speed / path ρ 0.5–0.8, 首帧×4 归零 ⇒ 多帧运动确实被读到了。
T27-A (CPU, tools/courtdyn_pixel_reader.py) 显示侧视上"控制像素位移后没有剩余的世界信息",
但侧视的像素量与世界量本身相关 0.8–0.95, 偏相关分辨力有限。T27 用干预定案:
  A. **zoom2** (tools/build_courtdyn_zoom2.py): 同 item / 同帧 / 同题面 / 同 GT, 画面换成半幅裁剪再缩回
     同一输出宽度 ⇒ 像素位移 ×2、框高 ×2、世界量不变。pred(zoom2)/pred(full) ≈ 2 ⇒ 读像素; ≈ 1 ⇒ 用尺子。
     3 条俯视片段 + 原侧视片段 × {base, SFT, GRPO, 7B}。
  B. 侧视三条新片段的首帧×4 × 4 模型 (+ 7B 的 full), 让 Δρ(full − static-4) 在两种视角上都有。
复用 T26 队列的全部机制 (Stage / 看门狗 / 锁 / 状态), 只换格子、前缀与等待目标 (等 T26 结束)。
状态 results/courtdyn_t27_state.json; 日志 results/_t27_*.log; 汇总 tools/summarize_courtdyn_t27.py。
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
MAIN = "Q4_side_480-510"
TOPS = ["Q2_top_480-510", "Q1_top_0-30", "Q4_top_0-30"]
SIDES = ["Q1_side_0-30", "Q2_side_300-330", "Q4_side_570-600"]
ALL4 = ["base", "sft", "grpo", "qwen25vl7b"]


def log(msg):
    print(f"[t27 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t27_{seq}_{pool}_{mode}_{model}"


def seq_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def img_root(seq):
    return os.path.join(ROOT, "data", "courtdyn", "frames" if seq == MAIN else f"frames_{seq}")


# ---- 把 T26 模块改装成 T27 ------------------------------------------------
Q.log = log
Q.cell = cell
Q.seq_dir = seq_dir
Q.img_root = img_root
Q.STATE = os.path.join(RES, "courtdyn_t27_state.json")
Q.T25_STATE = os.path.join(RES, "courtdyn_t26_state.json")   # 等待目标: T26
Q.T25_LAST_STAGE = "summary_t26"
Q.POOLS = dict(Q.POOLS, zoom2=("qa_dyn_v1_zoom2.json", None, None))
Q.CELLS = [(s, "zoom2", "full", ALL4) for s in TOPS + [MAIN]]          # A
for _s in SIDES:                                                       # B
    Q.CELLS += [(_s, "main", "static4", ALL4), (_s, "main", "full", ["qwen25vl7b"])]
Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})
_t26_done_orig = Q.t25_done


def _t26_done():
    ok, why = _t26_done_orig()
    return ok, why.replace("T25", "T26")


Q.t25_done = _t26_done


def prep_stages():
    st = []
    for s in TOPS + [MAIN]:
        st.append(Q.Stage(f"zoom2_{s}", [Q.PY, "-u", os.path.join(TOOLS, "build_courtdyn_zoom2.py"), "--seq", s],
                          os.path.join(seq_dir(s), "qa_dyn_v1_zoom2.json"), f"_t27_zoom2_{s}.log",
                          needs=[os.path.join(seq_dir(s), "qa_dyn_v1.json"), os.path.join(Q.tt_split(s), "img1.mp4")]))
    return st


def summary_stage():
    summ = os.path.join(TOOLS, "summarize_courtdyn_t27.py")
    return Q.Stage("summary_t27", [Q.PY, "-u", summ], os.path.join(CD, "courtdyn_t27_table.md"),
                   "_t27_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t27_queue", log)
    prep, gpu = prep_stages(), Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t27_")
    log(f"T27 启动: {len(prep)} 个 CPU 准备阶段 (zoom2 建池), {len(gpu)} 个评测/解析阶段 "
        f"({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), 1 个汇总; 先等 T26 结束再占卡")
    Q.save_state()
    for sg in prep:
        try:
            Q.run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续")
            Q.set_stage(sg.name, status="failed", note=repr(e))
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
    log("T27 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
