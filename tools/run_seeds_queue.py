#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_seeds_queue.py — T22: v3 协议多 seed 方差 (审稿护甲, 2026-09-02).

pr_sprint 门控 (T15) 未通过, 故 seeds 43/44 当时按设计跳过。投稿前补上, 目的只有一个:
让 tab:directional 里 prompt-matched target-SFT (61.7) 与 GRPO (60.0) 的 1.7 分差距
有三 seed 方差可依。全部沿用 T15 的命令行, 仅 --seed / 输出目录不同。

顺序 (便宜的先跑, 中途停掉也有可交付物):
  1. target-SFT prompt_matched s43 (~1.2h)  → eval (~1h)
  2. target-SFT prompt_matched s44          → eval
  3. GRPO-v3 s43 (~6.5h)                    → eval
  4. GRPO-v3 s44                            → eval
  5. summarize_variance ×2 → results/protocol_v3/variance_targetsft_prompt_v3.md
                              results/protocol_v3/variance_v3.md
状态 results/seeds_state.json; 日志 results/_t22_*.log。机制 (锁 / GPU 空闲门 / 45 分钟
停滞看门狗 / 重试) 与 T15–T21 相同。
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
BR = os.path.join(RES, "basketball_bench_real")
STATE = os.path.join(RES, "seeds_state.json")
PY = sys.executable
BASE_MODEL = (r"E:\models\hf\hub\models--Qwen--Qwen3.5-4B"
              r"\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
SFT_ADAPTER = os.path.join(ROOT, "models", "courtsi-qwen3.5-4b-lora-final")
IMG_ROOT = os.path.join(ROOT, "data", "_kagglehub", "datasets",
                        "gabrielvanzandycke", "deepsport-dataset", "versions", "8")
QA_TRAIN_V3 = os.path.join(BR, "qa_grpo_train_v3_instant.json")
QA_EVAL_V3 = os.path.join(BR, "qa_eval_v3_instant.json")
EV_GRPO_42 = os.path.join(RES, "protocol_v3_eval_grpo")
EV_TS_42 = os.path.join(RES, "protocol_v3_eval_targetsft_prompt")
ADAPTER = "adapter_model.safetensors"
SUMMARY = "summary.json"
SEEDS = [43, 44]

STALL_MIN = 45
MAX_RESTARTS = 8


def log(msg):
    print(f"[t22 {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Stage:
    def __init__(self, name, cmd, done_file, log_name, watchdog=False, needs=None):
        self.name, self.cmd, self.done_file = name, cmd, done_file
        self.log_path = os.path.join(RES, log_name)
        self.watchdog = watchdog
        self.needs = needs or []


def bench(out_dir, adapter):
    return [PY, "-u", os.path.join(ROOT, "eval", "run_bench.py"),
            "--model_path", BASE_MODEL, "--load_4bit", "--max_pixels", "200704",
            "--qa_json", QA_EVAL_V3, "--img_root", IMG_ROOT, "--resume",
            "--metric_version", "floor_v2",
            "--output_folder", out_dir, "--adapter", adapter]


def targetsft_stages(seed):
    out = os.path.join(ROOT, "models", f"target-sft-v3-prompt-s{seed}")
    ev = os.path.join(RES, f"protocol_v3_eval_targetsft_prompt_s{seed}")
    return [
        Stage(f"targetsft_prompt_s{seed}",
              [PY, "-u", os.path.join(ROOT, "train", "train_qlora.py"),
               "--qa_json", QA_TRAIN_V3, "--img_root", IMG_ROOT,
               "--init_adapter", SFT_ADAPTER, "--budget_slice", "prompt_matched",
               "--num_samples", "0", "--num_train_epochs", "1", "--seed", str(seed),
               "--max_pixels", "200704", "--output_dir", out],
              os.path.join(out, "training_completion.json"),
              f"_t22_targetsft_s{seed}.log", watchdog=True),
        Stage(f"eval_targetsft_s{seed}", bench(ev, out),
              os.path.join(ev, SUMMARY), f"_t22_eval_targetsft_s{seed}.log",
              watchdog=True, needs=[os.path.join(out, ADAPTER)]),
    ]


def grpo_stages(seed):
    out = os.path.join(ROOT, "models", f"courtsi-grpo-v3-s{seed}")
    ev = os.path.join(RES, f"protocol_v3_eval_grpo_s{seed}")
    return [
        Stage(f"grpo_s{seed}",
              [PY, "-u", os.path.join(ROOT, "train", "train_grpo.py"),
               "--output_dir", out, "--seed", str(seed), "--max_hours", "40",
               "--qa_train", QA_TRAIN_V3],
              os.path.join(out, ADAPTER), f"_t22_grpo_s{seed}.log", watchdog=True),
        Stage(f"eval_grpo_s{seed}", bench(ev, out),
              os.path.join(ev, SUMMARY), f"_t22_eval_grpo_s{seed}.log",
              watchdog=True, needs=[os.path.join(out, ADAPTER)]),
    ]


def variance_stage(name, title, runs, out_md, log_name):
    cmd = [PY, "-u", os.path.join(ROOT, "eval", "summarize_variance.py")]
    for tag, d in runs:
        cmd += ["--run", f"{tag}={d}"]
    cmd += ["--title", title, "--out", out_md]
    return Stage(name, cmd, out_md, log_name,
                 needs=[os.path.join(d, SUMMARY) for _, d in runs])


STAGES = (targetsft_stages(43) + targetsft_stages(44)
          + grpo_stages(43) + grpo_stages(44)
          + [variance_stage(
                 "summary_targetsft_seeds",
                 "target-SFT (prompt-matched) v3 多 seed 方差 (instant-disjoint)",
                 [("seed42", EV_TS_42)] + [
                     (f"seed{s}", os.path.join(RES, f"protocol_v3_eval_targetsft_prompt_s{s}"))
                     for s in SEEDS],
                 os.path.join(RES, "protocol_v3", "variance_targetsft_prompt_v3.md"),
                 "_t22_sum_targetsft_seeds.log"),
             variance_stage(
                 "summary_grpo_seeds",
                 "GRPO v3 多 seed 方差 (instant-disjoint)",
                 [("seed42", EV_GRPO_42)] + [
                     (f"seed{s}", os.path.join(RES, f"protocol_v3_eval_grpo_s{s}"))
                     for s in SEEDS],
                 os.path.join(RES, "protocol_v3", "variance_v3.md"),
                 "_t22_sum_grpo_seeds.log")])

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
            # 提交内存耗尽 (lsass 泄漏等) 时重试是白跑, 直接放弃并说明原因
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
                log(f"{sg.name}: 重启超限, 放弃")
                set_stage(sg.name, status="failed", note="重启超限")
                return False
            time.sleep(30)
            continue
        rc = proc.returncode
        if rc == 0 and os.path.isfile(sg.done_file):
            log(f"{sg.name}: 完成 (rc=0)")
            set_stage(sg.name, status="done", rc=0,
                      ended=time.strftime("%Y-%m-%d %H:%M:%S"))
            return True
        restarts += 1
        if restarts > MAX_RESTARTS:
            set_stage(sg.name, status="failed", rc=rc)
            log(f"{sg.name}: rc={rc}, 重试超限, 放弃")
            return False
        log(f"{sg.name}: rc={rc}, 30s 后重试 ({restarts}/{MAX_RESTARTS})")
        set_stage(sg.name, status="running", rc=rc, restarts=restarts)
        time.sleep(30)


def main():
    acquire_instance_lock("seeds_queue", log)
    log(f"T22 多 seed 队列启动, {len(STAGES)} 个阶段 (预计 ~20h)")
    for sg in STAGES:
        try:
            run_stage(sg)
        except Exception as e:
            log(f"{sg.name}: 队列级异常 {e!r}, 继续下一阶段")
            set_stage(sg.name, status="failed", note=repr(e))
    state["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state()
    log("队列结束: " + ", ".join(
        f"{n}={d.get('status')}" for n, d in state["stages"].items()))


if __name__ == "__main__":
    main()
