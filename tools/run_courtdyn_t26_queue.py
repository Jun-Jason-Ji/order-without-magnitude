#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_courtdyn_t26_queue.py — T26: 俯视片段的对照臂 + 再加两条俯视片段 (2026-09-05)。

T21 的逐序列表里, 俯视片段 Q2_top_480-510 上 SFT / GRPO 的 speed / path ρ 到了 0.5–0.67
(CI 不含 0), 而那里 ρ(GT, 框高) 只有 0.10, 读框高的捷径解释不了。这与主片段 "ρ ≤ 0.23 /
变化探测器不是测量器" 的 E1 表述冲突, 必须在下笔前弄清楚:
  A. 现有俯视片段补齐审计对照臂 (T20 只在主片段跑过):
     static-4 (首帧 ×4, token 预算 = full, 运动信息 = 0): ρ 还在 0.6 ⇒ 读的是站位/间距, 不是运动;
                                                          ρ 掉到 0 ⇒ 多帧运动真的被读到了 (本文第一个正面结果)。
     灰板 (blank): 先验地板。   no-prior: 删身高句 (G3 逐序列版)。
     公开 7B 臂 (Qwen2.5-VL-7B): full / static-4 / blank + accel_vis full, 让俯视结论不依赖上一篇的 adapter。
  B. 再取两条俯视片段 (Q1_top_0-30, Q4_top_0-30: 不同节次), 从 Kaggle 逐文件下载 (~67 MB/条),
     建池 (v1 主池 + accel 视觉对) + 球场单应, 跑 full + static-4 × {base, SFT, GRPO, 7B} + accel_vis full,
     让俯视发现不是 n = 1。
CPU 准备阶段 (下载 / 建池 / 单应 / no-prior 池) 立即执行; 第一个 GPU 阶段前等 T25
(results/precision_seeds_state.json) 彻底结束 (finished 且最后阶段有终态, 宽限 3 h), 之后每个 GPU 阶段
仍过 wait_gpu_free + wait_commit_free。每个评测格后接 llm_extract 解析 + 成对打分, 与 T21 逐字相同。
收尾 CPU: tools/summarize_courtdyn_t26.py -> results/courtdyn/courtdyn_t26_table.md。
机制 (实例锁 / done-file 幂等 / 45 min 停滞看门狗 / 重试 8 次) 与 T15–T25 相同。

状态 results/courtdyn_t26_state.json; 日志 results/_t26_*.log;
仪表盘 http://127.0.0.1:8021/courtdyn_t26_dashboard.html
"""
import json
import os
import subprocess
import sys
import time

from queue_guard import (acquire_instance_lock, replace_state,
                         wait_commit_free, wait_gpu_free)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
CD = os.path.join(RES, "courtdyn")
STATE = os.path.join(RES, "courtdyn_t26_state.json")
T25_STATE = os.path.join(RES, "precision_seeds_state.json")
T25_LAST_STAGE = "analysis_seeds"
T25_GRACE_H = 3.0
PY = sys.executable

# 底座权重的位置随机器而异 —— 用环境变量覆盖, 默认值是本项目的开发机布局。
# (发布到公开仓库时这两个默认值只是示例, 换机器请设 QWEN35_4B / QWEN25_VL_7B。)
QWEN = os.environ.get("QWEN35_4B") or (
    r"E:\models\hf\hub\models--Qwen--Qwen3.5-4B"
    r"\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
SFT_ADAPTER = os.path.join(ROOT, "models", "courtsi-qwen3.5-4b-lora-final")
GRPO_V3 = os.path.join(ROOT, "models", "courtsi-grpo-v3")
QWEN25VL7B = os.environ.get("QWEN25_VL_7B") or r"E:\models\Qwen2.5-VL-7B-Instruct"
CDNATIVE = os.path.join(ROOT, "models", "courtdyn-native-sft")
MODELS = {
    "base": (QWEN, None),
    "sft": (QWEN, SFT_ADAPTER),
    "grpo": (QWEN, GRPO_V3),
    "qwen25vl7b": (QWEN25VL7B, None),
    # 2026-09-08 (T33): 直接在 CourtDyn 上从 base 训出的 LoRA, **不是**从论文 1 的 source-SFT 续训。
    # 训练片段 = 4 条侧视 + Q4_top; Q1_top / Q2_top 完全留出, 正是 ruler / pxunit 用的那两条。
    "cdnative": (QWEN, CDNATIVE),
}
TT = os.path.join(ROOT, "data", "external_validation", "teamtrack", "teamtrack-mot", "teamtrack-mot")
KAGGLE_HANDLE = "atomscott/teamtrack"
PROBE_HITS = os.path.join(CD, "teamtrack_probe_hits.json")

OLD_TOP = "Q2_top_480-510"                 # T21 已跑 full 的俯视片段
NEW_TOPS = ["Q1_top_0-30", "Q4_top_0-30"]  # 不同节次的两条新俯视片段
# pool -> (qa file inside seq dir, paired family, paired qa file)
POOLS = {
    "main": ("qa_dyn_v1.json", "faster", "qa_dyn_v1_paired_faster.json"),
    "noprior": ("qa_dyn_v1_noprior.json", None, None),
    "vis": ("qa_dyn_v2_paired_accel_vis.json", "accel_vis", "qa_dyn_v2_paired_accel_vis.json"),
}
ADAPTER = "adapter_model.safetensors"
SUMMARY = "summary.json"
STALL_MIN = 45
MAX_RESTARTS = 8
WAIT_POLL_MIN = 10

# GPU 格 (seq, pool, mode, models), 按优先级
CELLS = [
    (OLD_TOP, "main", "static4", ["base", "sft", "grpo", "qwen25vl7b"]),   # A1 决定性对照
    (OLD_TOP, "main", "blank", ["base", "sft", "grpo", "qwen25vl7b"]),     # A2 先验地板
    (OLD_TOP, "main", "full", ["qwen25vl7b"]),                             # A3 公开 7B 臂
    (OLD_TOP, "vis", "full", ["qwen25vl7b"]),
    (OLD_TOP, "vis", "static4", ["base", "sft", "grpo"]),                  # A4
    (OLD_TOP, "noprior", "full", ["base", "sft", "grpo"]),                 # A5
]
for _seq in NEW_TOPS:                                                      # B
    CELLS += [
        (_seq, "main", "full", ["base", "sft", "grpo", "qwen25vl7b"]),
        (_seq, "main", "static4", ["base", "sft", "grpo", "qwen25vl7b"]),
        (_seq, "vis", "full", ["base", "sft", "grpo", "qwen25vl7b"]),
    ]


def log(msg):
    print(f"[t26 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Stage:
    def __init__(self, name, cmd, done_file, log_name, watchdog=False, needs=None):
        self.name, self.cmd, self.done_file = name, cmd, done_file
        self.log_path = os.path.join(RES, log_name)
        self.watchdog = watchdog
        self.needs = needs or []


def cell(seq, pool, mode, model):
    return f"t26_{seq}_{pool}_{mode}_{model}"


def out_dir(seq, pool, mode, model):
    return os.path.join(CD, cell(seq, pool, mode, model))


def seq_dir(seq):
    return os.path.join(CD, f"seq_{seq}")


def img_root(seq):
    return os.path.join(ROOT, "data", "courtdyn", f"frames_{seq}")


def tt_split(seq):
    """teamtrack_probe_hits.json: [family, split, name] -> TeamTrack 序列目录。"""
    try:
        hits = json.load(open(PROBE_HITS, encoding="utf-8"))
        for fam, split, name in hits:
            if name == seq:
                return os.path.join(TT, fam, split, seq)
    except Exception:
        pass
    return os.path.join(TT, "basketball_top", "train", seq)


def bench(seq, pool, mode, model):
    path, adapter = MODELS[model]
    qa_json = os.path.join(seq_dir(seq), POOLS[pool][0])
    cmd = [PY, "-u", os.path.join(ROOT, "eval", "run_bench.py"),
           "--model_path", path, "--load_4bit", "--max_pixels", "200704",
           "--qa_json", qa_json, "--img_root", img_root(seq), "--resume",
           "--metric_version", "floor_v2",
           "--frame_mode", "full" if mode == "blank" else mode,
           "--output_folder", out_dir(seq, pool, mode, model)]
    if adapter:
        cmd += ["--adapter", adapter]
    if mode == "blank":
        cmd += ["--blank_images"]
    return cmd


def prep_stages():
    """CPU 准备: 下载 / 建池 / 单应 / no-prior 池。全部幂等。"""
    st = []
    for seq in NEW_TOPS:
        sd = tt_split(seq)
        st.append(Stage(f"download_{seq}",
                        [PY, "-u", os.path.join(ROOT, "tools", "download_teamtrack_seq.py"),
                         "--seq", seq, "--handle", KAGGLE_HANDLE],
                        os.path.join(sd, "img1.mp4"), f"_t26_download_{seq}.log"))
        st.append(Stage(f"build_{seq}",
                        [PY, "-u", os.path.join(ROOT, "tools", "build_courtdyn_qa.py"),
                         "--seq_dir", sd, "--out_dir", seq_dir(seq), "--frame_dir", img_root(seq)],
                        os.path.join(seq_dir(seq), "qa_dyn_v1.json"), f"_t26_build_{seq}.log",
                        needs=[os.path.join(sd, "img1.mp4"), os.path.join(sd, "gt", "gt.txt")]))
        st.append(Stage(f"build_vis_{seq}",
                        [PY, "-u", os.path.join(ROOT, "tools", "build_courtdyn_qa.py"),
                         "--seq_dir", sd, "--out_dir", seq_dir(seq), "--frame_dir", img_root(seq),
                         "--accel_visual"],
                        os.path.join(seq_dir(seq), "qa_dyn_v2_paired_accel_vis.json"),
                        f"_t26_build_vis_{seq}.log",
                        needs=[os.path.join(seq_dir(seq), "qa_dyn_v1.json")]))
        st.append(Stage(f"homography_{seq}",
                        [PY, "-u", os.path.join(ROOT, "tools", "calibrate_court_homography.py"),
                         "--seq_dir", sd],
                        os.path.join(CD, "homography", f"H_{seq}.json"), f"_t26_homography_{seq}.log",
                        needs=[os.path.join(sd, "img1.mp4")]))
    for seq in [OLD_TOP]:
        st.append(Stage(f"noprior_{seq}",
                        [PY, "-u", os.path.join(ROOT, "tools", "build_courtdyn_noprior.py"),
                         "--qa_json", os.path.join(seq_dir(seq), "qa_dyn_v1.json")],
                        os.path.join(seq_dir(seq), "qa_dyn_v1_noprior.json"), f"_t26_noprior_{seq}.log",
                        needs=[os.path.join(seq_dir(seq), "qa_dyn_v1.json")]))
    return st


def gpu_stages():
    st = []
    for seq, pool, mode, models in CELLS:
        qa_json = os.path.join(seq_dir(seq), POOLS[pool][0])
        for m in models:
            od = out_dir(seq, pool, mode, m)
            pd = od + "_parsed"
            adapter = MODELS[m][1]
            needs = [qa_json] + ([os.path.join(adapter, ADAPTER)] if adapter else [])
            if m == "qwen25vl7b":
                needs.append(os.path.join(QWEN25VL7B, "config.json"))
            c = cell(seq, pool, mode, m)
            st.append(Stage(f"eval_{c}", bench(seq, pool, mode, m), os.path.join(od, SUMMARY),
                            f"_t26_eval_{c}.log", watchdog=True, needs=needs))
            st.append(Stage(f"parse_{c}",
                            [PY, "-u", os.path.join(ROOT, "eval", "llm_extract.py"),
                             "--pred", os.path.join(od, "predictions.json"),
                             "--output_folder", pd, "--load_4bit", "--metric_version", "floor_v2"],
                            os.path.join(pd, SUMMARY), f"_t26_parse_{c}.log", watchdog=True,
                            needs=[os.path.join(od, "predictions.json")]))
            fam, paired_qa = POOLS[pool][1], POOLS[pool][2]
            if fam:
                st.append(Stage(f"paired_{c}",
                                [PY, "-u", os.path.join(ROOT, "eval", "paired_relational_scorer.py"),
                                 "score", "--qa_json", os.path.join(seq_dir(seq), paired_qa),
                                 "--pred_json", os.path.join(pd, "predictions.json"),
                                 "--out", os.path.join(pd, f"paired_{fam}_score.json")],
                                os.path.join(pd, f"paired_{fam}_score.json"), f"_t26_paired_{c}.log",
                                needs=[os.path.join(pd, "predictions.json")]))
    return st


def summary_stage():
    summ = os.path.join(ROOT, "tools", "summarize_courtdyn_t26.py")
    return Stage("summary_t26", [PY, "-u", summ], os.path.join(CD, "courtdyn_t26_table.md"),
                 "_t26_summary.log", needs=[summ])


state = {"stages": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None}


def save_state():
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    replace_state(tmp, STATE, log)


def set_stage(name, **kw):
    state["stages"].setdefault(name, {}).update(kw)
    save_state()


def kill_tree(proc):
    subprocess.call(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=60)
    except Exception:
        pass


def t25_done():
    try:
        with open(T25_STATE, encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        return True, "T25 状态文件读不到, 视为未运行"
    stages = s.get("stages", {})
    fin = s.get("finished")
    last = stages.get(T25_LAST_STAGE, {}).get("status")
    if last in ("done", "skipped", "failed") and fin:
        return True, f"T25 已于 {fin} 结束"
    if fin and T25_LAST_STAGE not in stages:
        try:
            age_h = (time.time() - time.mktime(time.strptime(fin, "%Y-%m-%d %H:%M:%S"))) / 3600
        except Exception:
            age_h = 0.0
        if age_h > T25_GRACE_H:
            return True, f"T25 finished 已 {age_h:.1f} h, 最后阶段未登记 -> 放行"
        return False, f"T25 上一轮已结束 ({fin}), 等它重拉跑完最后阶段 (宽限 {T25_GRACE_H} h)"
    done = sum(1 for d in stages.values() if d.get("status") in ("done", "skipped"))
    return False, f"T25 运行中 ({done}/{len(stages) or '?'} 阶段, 最后阶段 {last or '未开始'})"


def wait_for_t25():
    announced = False
    while True:
        ok, why = t25_done()
        if ok:
            log(why + ", T26 开始占卡")
            state.pop("waiting_for", None)
            save_state()
            return
        if not announced:
            log(why + f", T26 等待中, 每 {WAIT_POLL_MIN} 分钟查一次")
            announced = True
        state["waiting_for"] = why
        save_state()
        time.sleep(WAIT_POLL_MIN * 60)


def run_stage(sg):
    if os.path.isfile(sg.done_file):
        log(f"{sg.name}: 产物已存在, 跳过")
        set_stage(sg.name, status="done", rc=0, note="已有产物")
        return True
    for need in sg.needs:
        if not os.path.isfile(need):
            log(f"{sg.name}: 前置缺失 {os.path.basename(need)}, 跳过")
            set_stage(sg.name, status="skipped", note=f"缺 {os.path.basename(need)}")
            return False
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    restarts = 0
    while True:
        if sg.watchdog:
            wait_gpu_free(log)
            if not wait_commit_free(log):
                set_stage(sg.name, status="failed", note="提交内存不足 (重试无效)")
                return False
        set_stage(sg.name, status="running", restarts=restarts,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"),
                  log=os.path.basename(sg.log_path))
        log(f"{sg.name}: 启动 (第 {restarts+1} 次)")
        with open(sg.log_path, "a" if restarts else "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(sg.cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=ROOT, env=env)
            stalled = False
            while proc.poll() is None:
                time.sleep(60)
                if sg.watchdog:
                    try:
                        age = (time.time() - os.path.getmtime(sg.log_path)) / 60
                    except OSError:
                        age = 0
                    if age > STALL_MIN:
                        log(f"{sg.name}: 日志 {age:.0f} 分钟无更新 -> 杀进程树重启")
                        kill_tree(proc)
                        stalled = True
                        break
        if stalled:
            restarts += 1
            if restarts > MAX_RESTARTS:
                set_stage(sg.name, status="failed", note=f"停滞重启超过 {MAX_RESTARTS} 次")
                return False
            continue
        rc = proc.returncode
        if rc == 0 and os.path.isfile(sg.done_file):
            set_stage(sg.name, status="done", rc=0, ended=time.strftime("%Y-%m-%d %H:%M:%S"))
            log(f"{sg.name}: 完成 (rc=0)")
            return True
        restarts += 1
        if restarts > MAX_RESTARTS:
            set_stage(sg.name, status="failed", rc=rc, note=f"失败重试超过 {MAX_RESTARTS} 次")
            log(f"{sg.name}: 失败 rc={rc}, 放弃")
            return False
        log(f"{sg.name}: rc={rc} 或产物缺失, 60 s 后重试 ({restarts}/{MAX_RESTARTS})")
        time.sleep(60)


def main():
    acquire_instance_lock("courtdyn_t26_queue", log)
    prep, gpu = prep_stages(), gpu_stages()
    log(f"T26 启动: {len(prep)} 个 CPU 准备阶段, {len(gpu)} 个评测/解析/成对阶段 "
        f"({sum(1 for s in gpu if s.watchdog)} 个吃 GPU), 1 个汇总")
    save_state()
    for sg in prep:
        try:
            run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续")
            set_stage(sg.name, status="failed", note=repr(e))
    if "--no_wait" not in sys.argv:
        wait_for_t25()
    for sg in gpu + [summary_stage()]:
        try:
            run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            set_stage(sg.name, status="failed", note=repr(e))
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state()
    log("T26 结束: " + ", ".join(f"{n}={d.get('status')}" for n, d in state["stages"].items()))


if __name__ == "__main__":
    sys.exit(main())
