#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t31_queue.py — CourtDyn T31: 拆开裁剪干预 (N5-1) + 补齐无先验句臂 (N5-2)。

A. declut 臂 —— 把 zoom2 的复合干预拆成两半
   T27 的 zoom2 同时改了"表观尺寸"和"周边内容", 正文只能报一个复合干预 (§一 E2-C, 红线 11)。
   declut 用**与 zoom2 逐字相同的裁剪矩形**, 但按 full 臂的比例渲染后居中贴在黑底画布上:
       full   : 整帧      -> 表观 1x, 周边全在
       zoom2  : 裁剪块    -> 表观 2x, 周边已去
       declut : 同一裁剪块 -> 表观 1x, 周边已去
   于是 full->zoom2 = Δ尺寸+Δ周边, full->declut = Δ周边, 相减得 Δ尺寸。
   4 条片段 (3 俯视 + zoom2 用的那条侧视) x {source-SFT, GRPO} = 8 格。
   公开 7B 不跑 —— 拆解论证只需要两条适配链, 表里会写明未跑。

B. noprior 臂补齐 —— E1b(b) 目前只有 Q2_top 一条俯视片段
   "删掉身高句预测不变"这条结论现在靠 Q2_top + 原侧视片段撑着 (§一 E1b(b))。
   给 Q1_top / Q4_top 各补一条, 让它在三条俯视上都成立。2 片段 x {source-SFT, GRPO} = 4 格。

共 12 个评测格 (~2-3 GPU-h)。全部队列 T26-T30 已冻结, 本队列不与任何东西抢卡。
状态 results/courtdyn_t31_state.json; 日志 results/_t31_*.log。
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
TOPS = ["Q1_top_0-30", "Q2_top_480-510", "Q4_top_0-30"]
DECLUT_SEQS = TOPS + [MAIN]
NOPRIOR_SEQS = ["Q1_top_0-30", "Q4_top_0-30"]        # Q2_top 已有
ARMS = ["sft", "grpo"]


def log(msg):
    print(f"[t31 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t31_{seq}_{pool}_{mode}_{model}"


def seq_dir(seq):
    return CD if seq == MAIN else os.path.join(CD, f"seq_{seq}")


def img_root(seq):
    return os.path.join(ROOT, "data", "courtdyn", "frames" if seq == MAIN else f"frames_{seq}")


Q.log = log
Q.cell = cell
Q.seq_dir = seq_dir
Q.img_root = img_root
Q.STATE = os.path.join(RES, "courtdyn_t31_state.json")
Q.POOLS = dict(Q.POOLS, declut=("qa_dyn_v1_declut.json", None, None))

Q.CELLS = [(s, "declut", "full", ARMS) for s in DECLUT_SEQS]
Q.CELLS += [(s, "noprior", "full", ARMS) for s in NOPRIOR_SEQS]

Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})


def prep_stages():
    """CPU 准备: 题池 (幂等, 已建则跳过)。declut 要解一遍 mp4, 每条约 1-2 min。"""
    st = []
    for s in DECLUT_SEQS:
        st.append(Q.Stage(
            f"build_declut_{s}",
            [Q.PY, "-u", os.path.join(TOOLS, "build_courtdyn_declut.py"), "--seq", s],
            os.path.join(seq_dir(s), "qa_dyn_v1_declut.json"),
            f"_t31_build_declut_{s}.log",
            needs=[os.path.join(seq_dir(s), "qa_dyn_v1.json"),
                   os.path.join(Q.tt_split(s), "img1.mp4")]))
    for s in NOPRIOR_SEQS:
        st.append(Q.Stage(
            f"build_noprior_{s}",
            [Q.PY, "-u", os.path.join(TOOLS, "build_courtdyn_noprior.py"),
             "--qa_json", os.path.join(seq_dir(s), "qa_dyn_v1.json")],
            os.path.join(seq_dir(s), "qa_dyn_v1_noprior.json"),
            f"_t31_build_noprior_{s}.log",
            needs=[os.path.join(seq_dir(s), "qa_dyn_v1.json")]))
    return st


def summary_stage():
    """汇总脚本可能还没写完 —— needs 缺失时 run_stage 会跳过而不是判失败, 事后手动补跑即可。

    done_file 指向真正的产物; 若产物已存在会被跳过, 想强制重算就先删掉它。
    """
    summ = os.path.join(TOOLS, "summarize_courtdyn_t31.py")
    return Q.Stage("summary_t31", [Q.PY, "-u", summ],
                   os.path.join(CD, "courtdyn_t31_table.md"), "_t31_summary.log", needs=[summ])


def main():
    Q.acquire_instance_lock("courtdyn_t31_queue", log)
    prep, gpu = prep_stages(), Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t31_")
    log(f"T31 启动: {len(prep)} 个 CPU 准备阶段, {len(gpu)} 个评测/解析阶段 "
        f"({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), 1 个汇总")
    Q.save_state()
    for sg in prep + gpu + [summary_stage()]:
        try:
            Q.run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            Q.set_stage(sg.name, status="failed", note=repr(e))
    Q.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Q.save_state()
    log("T31 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
