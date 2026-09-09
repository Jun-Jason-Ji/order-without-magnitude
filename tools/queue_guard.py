#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
queue_guard.py — 队列运行的两道硬保护 (2026-08-26 双蓝屏事故后加)。

事故复盘: 8-26 晚 nvlddmkm (UVM 路径) 连续两次蓝屏。第二次的直接诱因是 GPU
双占用 —— T9 消融训练在跑的同时, 旧版 T12 反驳队列把「状态文件缺失」当放行
信号, 直接压上 blank_eval 评测; 8GB 卡被两个 CUDA 进程挤爆, WDDM/UVM 换页
路径触发驱动崩溃 (事件 14, \\Device\\UVMLiteProcess*)。此外蓝屏重启后人工
连开了 3 个消融队列实例 (21:49/21:53/22:03), 各自都会启动训练。

两道保护:
  1. acquire_instance_lock(name): 同名队列全机唯一。用 msvcrt 文件区域锁,
     进程死亡 (含蓝屏) 时 OS 自动释放, 不会留陈旧锁文件问题。
  2. wait_gpu_free(): GPU 阶段启动前确认卡上没有别的 compute 进程且显存
     接近空闲 (混合显卡本 dGPU 空闲时应为 ~0 MiB), 忙则等待。这是状态文件
     协调之外的兜底 —— 无论哪个队列的等待逻辑再出 bug, 都不会双占 GPU。
"""
import msvcrt
import os
import subprocess
import sys
import time

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

# 句柄必须活到进程结束, 锁才一直有效
_lock_handles = []

# 空闲判定: dGPU 显存占用低于此值且无 compute 进程 (桌面走 iGPU, dGPU 平时 0 MiB)
IDLE_MB = 500


def acquire_instance_lock(name, log=print):
    """同名队列只允许一个实例; 抢不到锁就退出 (码 3)。"""
    path = os.path.join(RES, f"_{name}.lock")
    f = open(path, "a", encoding="utf-8")
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log(f"另一个 {name} 实例已持有 {os.path.basename(path)}, 本实例退出")
        f.close()
        sys.exit(3)
    _lock_handles.append(f)
    log(f"实例锁已取得: {os.path.basename(path)} (pid={os.getpid()})")


def replace_state(tmp, dst, log=print):
    """os.replace 带重试。

    队列间靠轮询对方的 *_state.json 协调, 而 Python 打开文件不带
    FILE_SHARE_DELETE —— 读方恰好持有 dst 的瞬间, 写方的原子替换会吃
    PermissionError (8-27 00:45 消融队列被 T11 的 10 分钟轮询撞死在启动注册期)。
    读一次 json 只有毫秒级窗口, 短退避重试即可。"""
    for i in range(10):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            time.sleep(0.2 * (i + 1))
    log(f"replace_state: {os.path.basename(dst)} 持续被占用, 放弃本次写入")
    try:
        os.remove(tmp)
    except OSError:
        pass


def _query(args):
    out = subprocess.run(["nvidia-smi"] + args + ["--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    return out.stdout


def _gpu_busy():
    """返回 (是否忙, 原因)。

    nvidia-smi 不可用/报错 → 视为**忙** (等待)。8-27 驱动重装期间的教训: 设备
    无驱动时 nvidia-smi 消失, 若按空闲放行, 训练每次启动都在 CUDA init 处
    失败, ~7 分钟一轮把 8 次重启额度烧光, 整条队列被标 failed。GPU 用不了时
    启动阶段必失败, 等到 nvidia-smi 恢复 (驱动装好) 再放行严格更优。
    """
    try:
        # 只把 python 的 compute 进程当硬阻塞 —— 队列的训练/评测都是 python。
        # ChatGPT 桌面版等应用的 on-device-model 服务会 24/7 挂一个 ~13 MiB 的
        # compute 上下文, 按「任意 compute 进程=忙」判会永远等下去; 大块非
        # python CUDA 占用仍被下面的 500 MiB 显存门拦住。
        pids = []
        for ln in _query(["--query-compute-apps=pid,process_name"]).splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("N/A"):
                continue
            pid, _, pname = ln.partition(", ")
            if "python" in pname.lower():
                pids.append(pid)
        if pids:
            return True, f"python compute 进程 pid={','.join(pids)}"
        mem = _query(["--query-gpu=memory.used"]).strip().splitlines()
        used = int(mem[0]) if mem and mem[0].strip().isdigit() else 0
        if used >= IDLE_MB:
            return True, f"显存占用 {used} MiB (>= {IDLE_MB})"
        return False, ""
    except Exception as e:
        return True, f"nvidia-smi 不可用 ({e!r}) — GPU 驱动可能未就绪, 等待恢复"


def wait_gpu_free(log=print, poll_s=60):
    """阻塞到 GPU 空闲。每 10 分钟报一次还在等什么。"""
    waited = 0
    while True:
        busy, why = _gpu_busy()
        if not busy:
            if why:
                log(f"GPU 检查: {why}")
            if waited:
                log(f"GPU 已空闲 (等了 {waited // 60} 分钟)")
            return
        if waited % 600 == 0:
            log(f"GPU 忙 ({why}), 等待空闲后再启动本阶段")
        time.sleep(poll_s)
        waited += poll_s


# ---------------------------------------------------------------------------
# 第三道保护 (2026-09-02 加): 系统提交内存 (commit) 余量。
#
# 事故: T22 的 eval_grpo_s44 连崩 9 次 —— 8 次 `OSError: The paging file is too
# small ... (os error 1455)`, 最后一次直接 0xC0000005 访问违例, 全部死在
# safetensors mmap 加载 8.7 GB 权重分片的那一步。GPU 当时完全空闲, 不是显存
# 问题: lsass.exe 泄漏到 44.6 GB 私有提交内存 (开机 4.5 天), 把 81.6 GB 的
# commit limit 吃到只剩 14 GB, 模型加载申请不到内存。
#
# 关键教训: 这类故障重试无用 —— 队列 90 秒一次白跑 8 次, 12 分钟耗尽重试预算,
# 然后把 summary_grpo_seeds 也连带跳过。所以这里等待 + 点名最大占用者, 让日志
# 直接说清楚"谁吃了内存", 而不是刷一屏一模一样的 traceback。
COMMIT_FREE_GB = 20.0   # 4B 模型 4-bit 加载实测需要 >14 GB, 留出余量
COMMIT_WAIT_MIN = 30    # 等这么久还不够就放弃, 交给人处理 (通常要重启清 lsass)


def _commit_status():
    """-> (free_gb, limit_gb)。用 GlobalMemoryStatusEx 的 PageFile 字段,
    即性能计数器里的 Commit Limit / Committed Bytes。"""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError()
    return m.ullAvailPageFile / 2**30, m.ullTotalPageFile / 2**30


def _top_commit_hog():
    """占用私有提交内存最多的进程, 用来在日志里点名 (如泄漏的 lsass)。"""
    ps = ("Get-Process | Sort-Object PagedMemorySize64 -Descending | "
          "Select-Object -First 1 Name,Id,PagedMemorySize64 | ForEach-Object "
          "{ \"$($_.Name) (pid $($_.Id)) \" + "
          "[math]::Round($_.PagedMemorySize64/1GB,1) + ' GB' }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                              "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return "未知"
    return (out or "").strip() or "未知"


def wait_commit_free(log=print, need_gb=COMMIT_FREE_GB, poll_s=60,
                     max_min=COMMIT_WAIT_MIN):
    """阻塞到系统提交内存余量足够。不够则等待, 超时返回 False (调用方应放弃
    而不是重试 —— 重试解决不了 commit 耗尽)。"""
    waited = 0
    while True:
        try:
            free, limit = _commit_status()
        except Exception as e:          # 拿不到就别挡路
            log(f"commit 检查失败 ({e!r}), 跳过该保护")
            return True
        if free >= need_gb:
            if waited:
                log(f"commit 已回落到 {free:.1f} GB (等了 {waited // 60} 分钟)")
            return True
        if waited % 600 == 0:
            log(f"提交内存不足: 剩 {free:.1f} GB / 上限 {limit:.1f} GB "
                f"(需要 {need_gb:.0f} GB), 最大占用: {_top_commit_hog()}")
        if waited >= max_min * 60:
            log(f"提交内存等待超过 {max_min} 分钟仍不足 (剩 {free:.1f} GB, "
                f"最大占用: {_top_commit_hog()})。这类故障重试无效, 通常需要重启"
                f"释放泄漏进程, 或加大页面文件。放弃本阶段。")
            return False
        time.sleep(poll_s)
        waited += poll_s
