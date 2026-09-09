#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t34_queue.py — CourtDyn T34: 给 E1a ("序确实来自多帧") 补一条非论文 1 的臂。

缺口
----
T33 之后, 主结论的"**量**从不换算"那半已经独立 (cdnative 一个模型就能立起来)。
但"**序**被读到"那半还没有: §一 E1a 的表格四列全是 source-SFT / GRPO-v3, 都是论文 1 的适配器;
base 与公开 7B 在那一节太弱 (Δρ 仅 0.06-0.28), 撑不起"抽掉运动 ρ 就归零"。
而 T33 只跑了 cdnative 的 full / ruler / pxunit 三条臂, **没跑 static-4** ——
所以它证明了"量不换算", 没证明"序确实来自多帧"。

本队列补上: cdnative × {Q1_top, Q2_top} × static-4 (首帧×4, token 预算不变、运动信息为零)。
full 臂已在 T33 落盘, 直接复用。两条片段都是 cdnative 训练时**一帧未见**的留出片段。

判据 (开跑前写死)
-----------------
参照: 论文 1 的两个适配器在这两条片段上 Δρ(full − static4) 是 0.46-0.93 (§一 E1a)。
  * cdnative 也明显塌 (Δρ 与之同量级) ⇒ **E1a 也有了非论文 1 的臂**, 两半都不再只靠姊妹篇;
  * cdnative 不塌 ⇒ 它读的"序"另有来源 (可能是单帧站位), E1a 的适用范围要收窄, 如实写。
无论哪种都要报, 不许只报好看的那个。

状态 results/courtdyn_t34_state.json; 日志 results/_t34_*.log。
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
TOPS = ["Q1_top_0-30", "Q2_top_480-510"]


def log(msg):
    print(f"[t34 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cell(seq, pool, mode, model):
    return f"t34_{seq}_{pool}_{mode}_{model}"


Q.log = log
Q.cell = cell
Q.STATE = os.path.join(RES, "courtdyn_t34_state.json")
Q.CELLS = [(s, "main", "static4", ["cdnative"]) for s in TOPS]
Q.state.update({"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None})


def main():
    Q.acquire_instance_lock("courtdyn_t34_queue", log)
    gpu = Q.gpu_stages()
    for sg in gpu:
        sg.log_path = sg.log_path.replace("_t26_", "_t34_")
    log(f"T34 启动: {len(gpu)} 个阶段; cdnative 的首帧×4 臂 (full 臂已在 T33 落盘, 不重跑)")
    Q.save_state()
    for sg in gpu:
        try:
            Q.run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            Q.set_stage(sg.name, status="failed", note=repr(e))
    Q.state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    Q.save_state()
    log("T34 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in Q.state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
