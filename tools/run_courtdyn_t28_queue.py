#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t28_queue.py — CourtDyn T28: N2 (尺子/像素单位两臂) + N3 (target-SFT 臂)。

接 docs/courtdyn_results_summary_2026-09-05.md §五。T26/T27 已确立: 俯视片段上"序"被读到
(ρ 0.45–0.80, 首帧×4 归零), 但"尺"停在先验上 (高 ρ 不换来高 T-MRA; 删掉尺度句预测不变)。
T28 直接测"尺"这一侧, 并补上论文 1 最强的那个适配器:

  **N2-A ruler**  给一把**正确**的尺子: 身高句换成 "one meter on the floor spans about K pixels",
      K 由球场单应导出的片段级常数 (Q1_top 27, Q2_top 28; IQR 1.5–2.4%)。GT 同步换成 v3 (单应) 答案 ——
      递过去的尺子与计分必须在同一个世界里, 否则"模型真的用了尺子"反而会被判错。
      不变 ⇒ 模型不做这次乘除法 (E1b 从"尺度句惰性"升级为"给了也不用"); 改善 ⇒ 瓶颈是尺子的提取。
  **N2-B pxunit** 直接按图像平面提问 (单位 = 渲染像素, GT = 像素位移, 不给尺子)。
      若"读的是图像平面位移"成立, 这应当是模型的**最好**条件; 分数跳起来就把负结果变成正的机制结果。
  **N3 tsft** 论文 1 的 target-SFT (token 切片) 在原片段 accel_vis 上是最好的一臂 (paired 32.8)。
      在俯视片段补 main full / static-4 / zoom2 / ruler / pxunit, 看"序"会不会更高、"尺"会不会出现。
      若 tsft 的 zoom2 比值明显低于 GRPO/SFT ⇒ 逐 token 标签监督比标量奖励更容易学到归一化。

机制 (Stage / 看门狗 / 实例锁 / done-file 幂等 / 状态文件) 全部复用 T26 队列模块, 只换格子、前缀与等待目标。
开跑前轮询 results/courtdyn_t27_state.json 等 T27 结束, 不抢卡。
状态 results/courtdyn_t28_state.json; 日志 results/_t28_*.log; 汇总 tools/summarize_courtdyn_t28.py。
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
TOPS = ["Q1_top_0-30", "Q2_top_480-510"]          # 决定性片段在前
TSFT = os.path.join(ROOT, "models", "target-sft-v3-token")


def log(msg):
    print(f"[t28 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t28_{seq}_{pool}_{mode}_{model}"


def seq_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def img_root(seq):
    return os.path.join(ROOT, "data", "courtdyn",
                        "frames" if seq == MAIN else f"frames_{seq}")


# ---- 把 T26 模块改装成 T28 ------------------------------------------------
Q.log = log
Q.cell = cell
Q.seq_dir = seq_dir
Q.img_root = img_root
Q.STATE = os.path.join(RES, "courtdyn_t28_state.json")
Q.T25_STATE = os.path.join(RES, "courtdyn_t27_state.json")   # 等待目标: T27
Q.T25_LAST_STAGE = "summary_t27"
Q.MODELS = dict(Q.MODELS, tsft=(Q.QWEN, TSFT))
Q.POOLS = dict(
    Q.POOLS,
    ruler=("qa_dyn_v1_ruler.json", None, None),
    pxunit=("qa_dyn_v1_pxunit.json", None, None),
    zoom2=("qa_dyn_v1_zoom2.json", None, None),
)

N2_MODELS = ["sft", "grpo", "tsft", "base"]       # 两条适配链在前
Q.CELLS = []
for _s in TOPS:                                    # N2: 决定性片段 Q1_top 先跑完两臂
    Q.CELLS += [(_s, "ruler", "full", N2_MODELS),
                (_s, "pxunit", "full", N2_MODELS)]
for _s in TOPS:                                    # N3: target-SFT 补齐俯视的既有臂
    Q.CELLS += [(_s, "main", "full", ["tsft"]),
                (_s, "main", "static4", ["tsft"]),
                (_s, "zoom2", "full", ["tsft"])]

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})
_t27_done_orig = Q.t25_done


def _t27_done():
    ok, why = _t27_done_orig()
    return ok, why.replace("T25", "T27").replace("T26", "T28")


Q.t25_done = _t27_done


def prep_stages():
    """CPU 准备: 两个题池 (幂等; 已建则跳过)。"""
    st = []
    for s in TOPS:
        for v in ("ruler", "pxunit"):
            st.append(Q.Stage(
                f"build_{v}_{s}",
                [Q.PY, "-u", os.path.join(TOOLS, "build_courtdyn_ruler.py"),
                 "--seq", s, "--variant", v],
                os.path.join(seq_dir(s), f"qa_dyn_v1_{v}.json"),
                f"_t28_build_{v}_{s}.log",
                needs=[os.path.join(seq_dir(s), "qa_dyn_v1.json"),
                       os.path.join(CD, "homography", f"H_{s}.json")]))
    return st


def summary_stage():
    summ = os.path.join(TOOLS, "summarize_courtdyn_t28.py")
    return Q.Stage("summary_t28", [Q.PY, "-u", summ],
                   os.path.join(CD, "courtdyn_t28_table.md"), "_t28_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t28_queue", log)
    prep, gpu = prep_stages(), Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t28_")
    log(f"T28 启动: {len(prep)} 个 CPU 准备阶段, {len(gpu)} 个评测/解析阶段 "
        f"({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), 1 个汇总; 先等 T27 结束再占卡")
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
    log("T28 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
